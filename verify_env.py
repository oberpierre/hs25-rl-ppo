from src.env_wrapper import BreakoutTextWrapper
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def verify_env():
    logger.info("Initializing environment...")
    env = BreakoutTextWrapper()
    
    logger.info("Resetting environment...")
    obs, info = env.reset()
    logger.info(f"Initial Observation: {obs}")
    
    assert isinstance(obs, str)
    assert "Paddle X" in obs
    
    logger.info("Stepping environment...")
    # Test different actions
    actions = ["LEFT", "RIGHT", "FIRE", "NOOP"]
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"Action: {action}, Reward: {reward}, Done: {terminated or truncated}")
        logger.info(f"Observation: {obs}")
        
    logger.info("Environment verification passed!")
    env.close()

if __name__ == "__main__":
    verify_env()
