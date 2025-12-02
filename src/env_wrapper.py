import gymnasium as gym
import numpy as np
import ale_py
from collections import deque
from .reward_shaping import BreakoutRewardShaper
import logging

logger = logging.getLogger(__name__)

class BreakoutTextWrapper(gym.Wrapper):
    """
    Wraps Atari Breakout to provide text observations and accept text actions.
    Also integrates reward shaping.
    """
    def __init__(self, env_id="ALE/Breakout-v5", render_mode=None, use_reward_shaping=True):
        # We need RAM to extract features easily.
        # If env_id is the visual one, we can still access RAM via env.unwrapped.ale.getRAM()
        env = gym.make(env_id, render_mode=render_mode, obs_type="ram")
        super().__init__(env)
        
        self.reward_shaper = BreakoutRewardShaper() if use_reward_shaping else None
        self.observation_space = gym.spaces.Text(max_length=512)
        
        # Action mapping
        self.action_map = {
            "NOOP": 0,
            "FIRE": 1,
            "RIGHT": 2,
            "LEFT": 3
        }
        
        # Track history for velocity
        self.ball_history = deque(maxlen=2)
        
    def _get_info_from_ram(self, ram):
        # Breakout RAM map
        # 72: Paddle X
        # 99: Ball X
        # 101: Ball Y
        # 57: Lives
        
        paddle_x = ram[72]
        ball_x = ram[99]
        ball_y = ram[101]
        lives = ram[57]
        
        return {
            "paddle_x": paddle_x,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "lives": lives
        }

    def _get_text_obs(self, info):
        # Calculate approximate velocity if possible
        ball_vel_desc = "stationary"
        if len(self.ball_history) == 2:
            prev_x, prev_y = int(self.ball_history[0][0]), int(self.ball_history[0][1])
            curr_x, curr_y = int(self.ball_history[1][0]), int(self.ball_history[1][1])
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            
            if dy > 0: v_y = "down"
            elif dy < 0: v_y = "up"
            else: v_y = "flat"
            
            if dx > 0: v_x = "right"
            elif dx < 0: v_x = "left"
            else: v_x = "straight"
            
            ball_vel_desc = f"moving {v_y} and {v_x}"
            
        obs_text = (
            f"Paddle X: {info['paddle_x']}. "
            f"Ball X: {info['ball_x']}, Ball Y: {info['ball_y']}. "
            f"Ball is {ball_vel_desc}. "
            f"Lives: {info['lives']}. "
            "Goal: Keep the ball alive and break bricks. "
            "Available actions: LEFT, RIGHT, FIRE. "
            "What is your next move?"
        )
        return obs_text

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # obs is RAM
        
        ram_info = self._get_info_from_ram(obs)
        self.ball_history.clear()
        self.ball_history.append((ram_info['ball_x'], ram_info['ball_y']))
        
        if self.reward_shaper:
            self.reward_shaper.reset(ram_info)
            
        text_obs = self._get_text_obs(ram_info)
        return text_obs, info

    def step(self, action_text):
        # Parse action
        if isinstance(action_text, str):
            action_text = action_text.strip().upper()
            # Simple heuristic matching
            if "LEFT" in action_text:
                action = 3
            elif "RIGHT" in action_text:
                action = 2
            elif "FIRE" in action_text:
                action = 1
            else:
                action = 0 # NOOP
        elif isinstance(action_text, int) and action_text in [0, 1, 2, 3]:
            action = action_text
        else:
            logger.warning(f"Invalid action: {action_text}")
            action = 0 # NOOP
        
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        ram_info = self._get_info_from_ram(obs)
        self.ball_history.append((ram_info['ball_x'], ram_info['ball_y']))
        
        if self.reward_shaper:
            reward = self.reward_shaper.shape_reward(reward, ram_info, terminated or truncated)
            
        text_obs = self._get_text_obs(ram_info)
        
        return text_obs, reward, terminated, truncated, info
