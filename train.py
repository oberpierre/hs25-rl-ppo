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

def train():
    # Hyperparameters
    MAX_STEPS = 1000 # Small for testing, increase for real training
    STEPS_PER_EPOCH = 128
    BATCH_SIZE = 4
    LR = 1e-5
    GAMMA = 0.99
    SAVE_INTERVAL = 500
    
    # Setup paths
    now = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    run_name = f"{timestamp}_breakout_ppo_qwen"
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
    if torch.backends.mps.is_available():
        model.to("mps")
    elif torch.cuda.is_available():
        model.to("cuda")
    
    agent = PPOAgent(model, model.tokenizer, lr=LR, gamma=GAMMA)
    
    # Training Loop
    obs, info = env.reset()
    total_steps = 0
    
    while total_steps < MAX_STEPS:
        rollouts = []
        episode_rewards = []
        
        # Rollout Phase
        for _ in range(STEPS_PER_EPOCH):
            action_text, log_prob, value = agent.get_action_and_value(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action_text)
            
            rollouts.append({
                "state": obs,
                "action_text": action_text,
                "log_prob": log_prob,
                "reward": reward,
                "value": value,
                "done": terminated or truncated
            })
            
            obs = next_obs
            total_steps += 1
            
            if terminated or truncated:
                obs, info = env.reset()
                
        # Compute Advantages and Returns
        # We need next value for the last step
        _, _, next_value = agent.get_action_and_value(obs) # Just to get value
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
        
        # Logging
        avg_reward = np.mean(rewards)
        writer.add_scalar("Loss/policy_value", loss, total_steps)
        writer.add_scalar("Reward/average_step", avg_reward, total_steps)
        
        logger.info(f"Step {total_steps}: Loss {loss:.4f}, Avg Reward {avg_reward:.4f}")
        
        # Save Checkpoint
        # Check if we crossed a save interval or finished
        if total_steps // SAVE_INTERVAL > (total_steps - STEPS_PER_EPOCH) // SAVE_INTERVAL:
            save_path = f"checkpoints/{run_name}_{total_steps}"
            model.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    env.close()
    writer.close()

if __name__ == "__main__":
    train()
