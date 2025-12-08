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
        self.tokenizer = None
        
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

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _get_text_obs(self, info):
        # Dynamic observation
        obs_content = (
            f"Paddle X: {info['paddle_x']}.\n"
            f"Ball X: {info['ball_x']}, Ball Y: {info['ball_y']}.\n"
            f"Lives: {info['lives']}."
        )
        
        # Add velocity info
        if len(self.ball_history) == 2:
            prev_x, prev_y = int(self.ball_history[0][0]), int(self.ball_history[0][1])
            curr_x, curr_y = int(self.ball_history[1][0]), int(self.ball_history[1][1])
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            
            if dy > 0: v_y = "DOWN"
            elif dy < 0: v_y = "UP"
            else: v_y = "FLAT"
            
            if dx > 0: v_x = "RIGHT"
            elif dx < 0: v_x = "LEFT"
            else: v_x = "STRAIGHT"
            
            obs_content += f"\nBall is moving {v_y} and {v_x}."
        else:
            obs_content += "\nBall is stationary."

        # Add Relative Position (Semantic Perception)
        # Paddle width is approx 16 pixels in Breakout (standard)
        # Center alignment check
        paddle_center = info['paddle_x'] + 8 # Approx center
        ball_center = info['ball_x']
        
        if ball_center < paddle_center - 4:
            rel_pos = "to your LEFT"
        elif ball_center > paddle_center + 4:
            rel_pos = "to your RIGHT"
        else:
            rel_pos = "ALIGNED with you"
            
        obs_content += f"\nThe Ball is {rel_pos}."

        if info['ball_y'] == 0:
            obs_content += "\nBall is not in play. You must FIRE to start."

        # System Prompt
        system_prompt = (
            "You are an expert Atari Breakout player.\n"
            "GAME RULES:\n"
            "1. The paddle moves horizontally at a fixed Y position of 189.\n"
            "2. The ball bounces off walls and bricks.\n"
            "3. Your goal is to intercept the ball with the paddle to break bricks.\n"
            "4. If the ball passes the paddle (Y > 189), you lose a life.\n"
            "5. Output ONLY one word: LEFT, RIGHT, FIRE, or NOOP."
        )

        # Apply Chat Template if tokenizer is available
        if self.tokenizer:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obs_content + "\nAction:"}
            ]
            # apply_chat_template returns a string
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            return full_prompt
        else:
            # Fallback
            return f"{system_prompt}\n\nOBSERVATION:\n{obs_content}\nAction:"

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
            cleaned_text = action_text.strip().upper()
            # Simple heuristic matching
            if "LEFT" in cleaned_text:
                action = 3
            elif "RIGHT" in cleaned_text:
                action = 2
            elif "FIRE" in cleaned_text:
                action = 1
            elif "NOOP" in cleaned_text:
                action = 0
            else:
                action = 0 # NOOP
                logger.warning(f"Invalid action: {action_text}")
                
        elif isinstance(action_text, int) and action_text in [0, 1, 2, 3]:
            action = action_text
        else:
            logger.warning(f"Invalid action: {action_text}")
            action = 0 # NOOP
        
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        ram_info = self._get_info_from_ram(obs)
        self.ball_history.append((ram_info['ball_x'], ram_info['ball_y']))
        
        if self.reward_shaper:
            # Shape game reward
            reward = self.reward_shaper.shape_reward(reward, ram_info, terminated or truncated)
            # Add format reward
            reward += self.reward_shaper.calculate_format_reward(action_text)
            
        text_obs = self._get_text_obs(ram_info)
        
        return text_obs, reward, terminated, truncated, info
