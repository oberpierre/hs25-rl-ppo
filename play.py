import argparse
import torch
import time
import logging
from src.env_wrapper import BreakoutTextWrapper
from src.model import ActorCritic

import os
# Mute SDL audio for Atari
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def play(checkpoint_path=None, model_name="Qwen/Qwen3-0.6B", num_episodes=1):
    logger.info(f"Loading model (Checkpoint: {checkpoint_path})...")
    
    if checkpoint_path:
        model = ActorCritic.from_pretrained(checkpoint_path, model_name=model_name)
    else:
        model = ActorCritic(model_name=model_name)
        
    if torch.backends.mps.is_available():
        model.to("mps")
    elif torch.cuda.is_available():
        model.to("cuda")
        
    env = BreakoutTextWrapper(render_mode="human", use_reward_shaping=False)
    env.set_tokenizer(model.tokenizer)
    
    episode_rewards = []
    
    try:
        for episode in range(num_episodes):
            logger.info(f"Starting Episode {episode + 1}/{num_episodes}...")
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            start_time = time.time()
            step_count = 0
            while not done:
                # Render is handled by gym environment in "human" mode
                
                # Get action
                inputs = model.tokenizer(obs, return_tensors="pt").to(model.base_model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=10,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=model.tokenizer.pad_token_id
                    )
                    
                generated_ids = outputs[:, inputs.input_ids.shape[1]:]
                action_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Clean up action text for logging/env
                action_text = action_text.strip().split('\n')[0]
                
                logger.debug(f"Observation: {obs}")
                logger.debug(f"Action: {action_text}")
                
                obs, reward, terminated, truncated, info = env.step(action_text)
                total_reward += reward
                done = terminated or truncated
                
                # Slow down for visibility
                # time.sleep(0.1) 
                
                step_count += 1

            elapsed_time = time.time() - start_time
            episode_rewards.append(total_reward)
            logger.info(f"Episode {episode + 1} Finished. Reward: {total_reward} ({step_count} steps, {elapsed_time:.2f}s)")
        
        # Statistics
        if episode_rewards:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            min_reward = min(episode_rewards)
            max_reward = max(episode_rewards)
            
            print("\n" + "="*40)
            print(f"SESSION STATISTICS ({len(episode_rewards)} Episodes)")
            print("="*40)
            print(f"Min Reward: {min_reward}")
            print(f"Max Reward: {max_reward}")
            print(f"Avg Reward: {avg_reward:.2f}")
            print("="*40 + "\n")
            
    except KeyboardInterrupt:
        logger.info("Game stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play")
    args = parser.parse_args()
    
    play(args.checkpoint, args.base_model, args.episodes)
