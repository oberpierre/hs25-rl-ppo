import numpy as np

class BreakoutRewardShaper:
    """
    Potential-based reward shaping for Atari Breakout.
    
    The potential function Phi(s) is designed to guide the agent towards:
    1. Keeping the paddle close to the ball (especially when the ball is moving down).
    2. Hitting the ball (implied by keeping it alive).
    3. Breaking bricks (standard reward, but we can amplify it).
    
    Formula:
    R_shaped = r + gamma * Phi(s') - Phi(s)
    """
    
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.last_potential = 0.0
        
    def get_potential(self, info: dict) -> float:
        """
        Calculate potential Phi(s) based on game info extracted from RAM.
        
        Expected keys in info:
        - paddle_x: int (0-255 approx)
        - ball_x: int
        - ball_y: int
        - ball_y_vel: int or float (positive = down, negative = up usually, depends on coordinate system)
        """
        # In Atari, y=0 is usually top. So increasing y is moving down.
        # We want to be close to the ball when it's low (high y).
        
        paddle_x = info.get('paddle_x', 0)
        ball_x = info.get('ball_x', 0)
        ball_y = info.get('ball_y', 0)
        
        # Normalize roughly to 0-1
        # Screen width is 160, but RAM values might differ. 
        # Usually paddle x is 0-200ish.
        
        dist = abs(int(paddle_x) - int(ball_x))
        max_dist = 160.0 # Approx screen width
        
        # Potential 1: Proximity to ball. 
        # We care more about this when the ball is lower (closer to paddle).
        # ball_y goes from ~0 (top) to ~200 (bottom).
        normalized_y = ball_y / 210.0
        normalized_dist = dist / max_dist
        
        # If ball is low (normalized_y close to 1), we want dist to be small.
        # Phi = (1 - dist) * (ball_y^2) 
        # This gives high potential when ball is close to paddle AND ball is near the paddle.
        potential = (1.0 - normalized_dist) * (normalized_y ** 2)
        
        return potential

    def shape_reward(self, reward: float, info: dict, done: bool) -> float:
        """
        Calculate shaped reward.
        """
        current_potential = self.get_potential(info)
        
        if done:
            # If episode ends, next potential is 0 (or we can treat it differently).
            # Usually set next potential to 0 for terminal states.
            next_potential = 0.0
            # Reset last potential for next episode
            shaped_reward = reward + self.gamma * next_potential - self.last_potential
            self.last_potential = 0.0
        else:
            shaped_reward = reward + self.gamma * current_potential - self.last_potential
            self.last_potential = current_potential
            
        return shaped_reward
        
    def reset(self, info: dict):
        self.last_potential = self.get_potential(info)
