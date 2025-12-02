import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class ActorCritic(nn.Module):
    """
    Actor-Critic model using a Causal LM as the backbone.
    The Actor is the Causal LM itself (generating tokens).
    The Critic is a value head on top of the last hidden state.
    """
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        super().__init__()
        
        # Load Base Model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"] # Common for attention models
        )
        
        self.base_model = get_peft_model(self.base_model, peft_config)
        
        # Value Head
        # We'll use the hidden size from the config
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize value head
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        # We need the hidden states to compute value
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Last hidden state of the last token? 
        # Or pooling? usually last token for causal LM.
        # hidden_states: (batch, seq_len, hidden)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Value estimate (batch, seq_len, 1) or just for the last token?
        # For PPO, we usually want value for every token if we do token-level RL,
        # but here we are doing action-level RL (one text response = one action?).
        # If we treat the whole generation as one action, we just need value at the start.
        # But standard LLM PPO (like TRL) does token-level.
        # Let's stick to a simpler approach: 
        # The model outputs text. We treat the entire response as the action.
        # So we need the value of the state (input prompt).
        # We take the value from the last token of the INPUT prompt.
        
        values = self.value_head(last_hidden_state)
        
        return outputs.logits, values

    def generate(self, input_ids, **kwargs):
        return self.base_model.generate(input_ids, **kwargs)
    
    def save_pretrained(self, path):
        self.base_model.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, model_name="Qwen/Qwen3-0.6B"):
        model = cls(model_name)
        model.base_model.load_adapter(path, "default")
        model.value_head.load_state_dict(torch.load(f"{path}/value_head.pt"))
        return model
