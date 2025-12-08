import os
# Set this before any other imports or tokenizer usage to suppress warnings and avoid deadlocks in forks (e.g. video recording)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import logging
import time
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo

from src.env_wrapper import BreakoutTextWrapper
from src.model import ActorCritic
from src.ppo import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def train(
    lr=1e-5,
    target_kl=0.1,
    steps_per_epoch=128,
    max_steps=128 * 20, # Default to ~20 epochs
):
    # Hyperparameters
    STEPS_PER_EPOCH = steps_per_epoch
    BATCH_SIZE = 4
    LR = lr
    GAMMA = 0.99
    TARGET_KL = target_kl
    INIT_KL_COEF = 0.05
    SAVE_INTERVAL = STEPS_PER_EPOCH * 4
    MAX_STEPS = max_steps
    
    # Setup paths
    now = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    run_name = f"{timestamp}_lr{LR}_kl{TARGET_KL}"
    log_dir = f"runs/{run_name}"
    video_dir = f"videos/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # Environment
    # We use a lambda to create env for video recording
    def make_env(render_mode=None):
        return BreakoutTextWrapper(render_mode=render_mode)
    
    env = make_env(render_mode="rgb_array")
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    
    # Model
    logger.info("Loading model...")
    model = ActorCritic()
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Pass tokenizer to environment wrapper for chat template formatting
    # env is RecordVideo(BreakoutTextWrapper), so we need to access the inner wrapper
    # RecordVideo -> BreakoutTextWrapper
    try:
        env.set_tokenizer(model.tokenizer)
    except AttributeError:
        current_env = env
        while hasattr(current_env, 'env'):
            if isinstance(current_env, BreakoutTextWrapper):
                current_env.set_tokenizer(model.tokenizer)
                break
            current_env = current_env.env
    
    agent = PPOAgent(model, model.tokenizer, lr=LR, gamma=GAMMA)
    
    # Initialize Adaptive KL Coefficient
    kl_coef = INIT_KL_COEF
    
    # Training Loop
    obs, info = env.reset()
    total_steps = 0
    current_episode_reward = 0
    
    from tqdm import tqdm
    from collections import deque
    
    pbar = tqdm(total=MAX_STEPS, desc="Training Steps")
    
    # Stats for tqdm
    recent_scores = deque(maxlen=10)
    last_score = 0.0
    avg_score = 0.0
    last_loss = 0.0
    
    while total_steps < MAX_STEPS:
        rollouts = []
        
        # Rollout Phase
        for _ in range(STEPS_PER_EPOCH):
            action_text, log_prob, value, ref_log_prob = agent.get_action_and_value(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action_text)
            
            # KL Penalty
            kl = log_prob - ref_log_prob
            reward_penalized = reward - kl_coef * kl
            
            rollouts.append({
                "state": obs,
                "action_text": action_text,
                "log_prob": log_prob,
                "reward": reward_penalized, # PPO optimizes this
                "value": value,
                "done": terminated or truncated,
                "kl": kl
            })
            
            current_episode_reward += reward
            obs = next_obs
            total_steps += 1
            pbar.update(1)
            
            # Update progress bar description immediately
            pbar.set_postfix_str(f"Avg: {avg_score:.2f} | Curr: {current_episode_reward:.2f} | Last: {last_score:.2f} | Loss: {last_loss:.4f}\n")
            
            if terminated or truncated:
                # Log episode reward immediately
                writer.add_scalar("Reward/episode_reward", current_episode_reward, total_steps)
                writer.flush() # Ensure it's written to disk
                
                last_score = current_episode_reward
                recent_scores.append(last_score)
                avg_score = sum(recent_scores) / len(recent_scores)
                
                writer.add_scalar("Reward/avg_score", avg_score, total_steps)
                writer.flush()
                
                logger.info(f"Episode finished at step {total_steps}. Reward: {current_episode_reward:.2f}")
                
                current_episode_reward = 0
                obs, info = env.reset()
                
        # Compute Advantages and Returns
        # We need next value for the last step
        _, _, next_value, _ = agent.get_action_and_value(obs) # Just to get value
        # Actually get_action_and_value returns (action, log_prob, value)
        # We discard action/log_prob
        next_value = next_value
        
        rewards = [r['reward'] for r in rollouts]
        values = [r['value'] for r in rollouts]
        dones = [r['done'] for r in rollouts]
        
        advantages = agent.compute_advantages(rewards, values, next_value, dones)
        
        # Add advantages and returns to rollouts
        for i, r in enumerate(rollouts):
            r['advantage'] = advantages[i]
            r['return'] = advantages[i] + values[i]
            
        # Update Phase
        loss = agent.update(rollouts, batch_size=BATCH_SIZE)
        last_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # Logging
        avg_step_reward = np.mean([r['reward'] for r in rollouts]) # This is penalized reward
        avg_raw_reward = np.mean(rewards)
        avg_kl = np.mean([r['kl'] for r in rollouts])
        
        # Adaptive KL Logic
        if avg_kl > TARGET_KL * 1.5:
            kl_coef *= 2.0
        elif avg_kl < TARGET_KL / 1.5:
            kl_coef /= 2.0
            
        # Linear Learning Rate Decay
        # Standard PPO schedule: Linear decay from LR to 0
        frac = 1.0 - (total_steps - 1.0) / MAX_STEPS
        new_lr = frac * LR
        agent.update_lr(new_lr)
        
        writer.add_scalar("Loss/policy_value", last_loss, total_steps)
        writer.add_scalar("Reward/average_step_reward", avg_step_reward, total_steps)
        writer.add_scalar("Reward/average_raw_reward", avg_raw_reward, total_steps)
        writer.add_scalar("KL/mean_divergence", avg_kl, total_steps)
        writer.add_scalar("KL/coefficient", kl_coef, total_steps)
        writer.add_scalar("Training/learning_rate", new_lr, total_steps)
        writer.flush()
        
        logger.info(f"Step {total_steps}: Loss {last_loss:.4f}, Avg Reward {avg_step_reward:.4f}, KL {avg_kl:.4f}, KL Coef {kl_coef:.4f}, LR {new_lr:.2e}")
        
        # Save Checkpoint
        # Check if we crossed a save interval or finished
        if total_steps // SAVE_INTERVAL > (total_steps - STEPS_PER_EPOCH) // SAVE_INTERVAL:
            save_path = f"checkpoints/{run_name}/step_{total_steps}"
            model.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    pbar.close()
    env.close()
    writer.close()

if __name__ == "__main__":
    train()
