import argparse
import torch
import time
import logging
from src.env_wrapper import BreakoutTextWrapper
from src.model import ActorCritic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def play(checkpoint_path=None, model_name="Qwen/Qwen3-0.6B"):
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
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    logger.info("Starting game...")
    
    try:
        while not done:
            # Render is handled by gym environment in "human" mode
            
            # Get action
            inputs = model.tokenizer(obs, return_tensors="pt").to(model.base_model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=model.tokenizer.pad_token_id
                )
                
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            action_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up action text for logging/env
            action_text = action_text.strip().split('\n')[0]
            
            logger.info(f"Observation: {obs}")
            logger.info(f"Action: {action_text}")
            
            obs, reward, terminated, truncated, info = env.step(action_text)
            total_reward += reward
            done = terminated or truncated
            
            # Slow down for visibility
            # time.sleep(0.1) 
            
        logger.info(f"Game Over. Total Reward: {total_reward}")
        
    except KeyboardInterrupt:
        logger.info("Game stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    args = parser.parse_args()
    
    play(args.checkpoint, args.base_model)
