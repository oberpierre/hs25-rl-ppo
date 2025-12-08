import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PPOAgent:
    def __init__(self, model, tokenizer, lr=1e-5, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, value_coef=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def get_action_and_value(self, state_text, max_new_tokens=10):
        # Tokenize state
        inputs = self.tokenizer(state_text, return_tensors="pt").to(self.model.base_model.device)
        
        # Generate action (response)
        # We need to manually sample to get log probs, or use generate and then re-evaluate.
        # Re-evaluation is standard for PPO updates, but for rollout we just need the action.
        # However, to store log_prob_old, we might want to compute it now.
        # Simplest: Generate, then forward pass to get log prob.
        
        with torch.no_grad():
            # Get value of the state
            _, value = self.model(inputs.input_ids, inputs.attention_mask)
            value = value[:, -1, 0] # Value of the last token of prompt
            
            # Generate
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Extract generated tokens
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            action_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Compute log prob of the generated sequence
            # We need to run forward pass on the full sequence (prompt + gen)
            logits, _ = self.model(outputs)
            logits = logits[:, :-1, :] # Shift right
            target_ids = outputs[:, 1:]
            
            # We only care about log probs of the NEW tokens
            gen_start_idx = inputs.input_ids.shape[1] - 1
            
            gen_logits = logits[:, gen_start_idx:, :]
            gen_targets = target_ids[:, gen_start_idx:]
            
            log_probs = F.log_softmax(gen_logits, dim=-1)
            
            # Gather log probs of selected tokens
            selected_log_probs = torch.gather(log_probs, -1, gen_targets.unsqueeze(-1)).squeeze(-1)
            
            # Sum log probs for the whole action
            action_log_prob = selected_log_probs.sum(dim=-1)

            # Compute Reference Log Prob (for KL)
            with self.model.base_model.disable_adapter():
                ref_logits, _ = self.model(outputs)
                ref_logits = ref_logits[:, :-1, :]
                ref_gen_logits = ref_logits[:, gen_start_idx:, :]
                
                ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
                ref_selected_log_probs = torch.gather(ref_log_probs, -1, gen_targets.unsqueeze(-1)).squeeze(-1)
                ref_action_log_prob = ref_selected_log_probs.sum(dim=-1)
            
        return action_text, action_log_prob.item(), value.item(), ref_action_log_prob.item()

    def compute_advantages(self, rewards, values, next_value, dones):
        # rewards: list of floats
        # values: list of floats
        # next_value: float
        # dones: list of bools
        
        advantages = []
        last_gae_lam = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[step])
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(dones[step])
                next_val = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)
            
        return np.array(advantages)

    def update(self, rollouts, batch_size=4, epochs=4):
        # rollouts is a list of dicts: {state, action_text, log_prob, reward, value, advantage, return}
        
        # Flatten data
        states = [r['state'] for r in rollouts]
        actions = [r['action_text'] for r in rollouts]
        old_log_probs = torch.tensor([r['log_prob'] for r in rollouts], dtype=torch.float32).to(self.model.base_model.device)
        returns = torch.tensor([r['return'] for r in rollouts], dtype=torch.float32).to(self.model.base_model.device)
        advantages = torch.tensor([r['advantage'] for r in rollouts], dtype=torch.float32).to(self.model.base_model.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_len = len(states)
        indices = np.arange(dataset_len)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_len, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = [states[i] for i in batch_indices]
                batch_actions = [actions[i] for i in batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Re-evaluate
                # Construct full input: state + action
                # We need to handle padding if batching
                # For simplicity, let's process one by one or assume padding is handled by tokenizer
                
                # Tokenize batch
                # "state + action"
                full_texts = [s + a for s, a in zip(batch_states, batch_actions)]
                inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.model.base_model.device)
                
                # We need to know where the action starts to mask out the prompt for loss
                # This is tricky with batching if lengths differ. 
                # A simple way is to tokenize states separately to get lengths.
                state_inputs = self.tokenizer(batch_states, return_tensors="pt", padding=True) # Just to get lengths? No, padding messes up lengths.
                # Better: tokenize state individually to get length, then create mask.
                
                # Let's do a simpler loop or careful masking.
                # Given the complexity of batching variable length sequences in PPO for LLMs, 
                # iterating or small batches is safer.
                
                # Forward pass
                logits, values = self.model(inputs.input_ids, inputs.attention_mask)
                
                # Values: we want value of the state (end of prompt).
                # This is hard to pinpoint in a padded batch without careful indexing.
                # Simplified: Value of the FIRST token (or specific token). 
                # Actually, we stored value of the state.
                # Let's use the value head output at the position corresponding to the last token of the state.
                
                # Log probs
                # Shift logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                
                # Calculate log probs
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                # Gather
                selected_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Mask out prompt (we only care about action likelihood)
                # We need a mask that is 0 for prompt tokens and 1 for action tokens.
                # We can construct this by tokenizing states.
                
                action_log_probs = []
                new_values = []
                
                for i in range(len(batch_states)):
                    # Re-tokenize single item to find split point
                    s_ids = self.tokenizer(batch_states[i], return_tensors="pt").input_ids
                    s_len = s_ids.shape[1]
                    
                    # Action log prob is sum of log probs from s_len-1 to end
                    # selected_log_probs[i] has length seq_len-1
                    # Corresponds to prediction at 0 (for token 1), ..., prediction at N-2 (for token N-1)
                    # We want predictions for tokens starting at s_len.
                    # The token at s_len is predicted by s_len-1.
                    
                    # Indices in shift_labels:
                    # 0 -> label is token 1
                    # ...
                    # s_len-1 -> label is token s_len (first token of action)
                    
                    item_log_prob = selected_log_probs[i, s_len-1:].sum()
                    action_log_probs.append(item_log_prob)

                    # Value: at s_len-1 (last token of state)
                    # values[i] has shape (seq_len, 1)
                    # We want value predicted at s_len-1
                    new_values.append(values[i, s_len-1, 0])
                    
                action_log_probs = torch.stack(action_log_probs)
                new_values = torch.stack(new_values)

                # PPO Loss
                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(new_values, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        return loss.item()
