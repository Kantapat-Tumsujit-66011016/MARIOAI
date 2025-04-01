import gym
import numpy as np
import cv2
from collections import deque
from gym import spaces
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
import gc

class MarioRLEnv:
    def __init__(self, stack_frames=4):
        self.env = None
        self.stack_frames = stack_frames
        self.frame_buffer = deque(maxlen=stack_frames)
        self.create_env()
        
        self.last_x = 0
        self.last_y = 0
        self.last_status = ''
        self.last_coins = 0
        self.max_position_ever = 0 
        self.position_milestones = set()  
        
    def create_env(self):
        try:
            if self.env is not None:
                try:
                    self.env.close()
                except:
                    pass
                
            gc.collect()
            time.sleep(0.1)
                
            self.env = gym_super_mario_bros.make("SuperMarioBros-v0")
            self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
            
            self.last_x = 0
            self.last_y = 0
            self.last_status = ''
            self.last_coins = 0
            self.frame_buffer.clear()
            
            return True
        except Exception as e:
            print(f"Error creating environment: {e}")
            time.sleep(0.5)
            return False
        
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return resized
        
    def reset(self):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                obs = self.env.reset()
                
                self.frame_buffer.clear()
                
                processed_frame = self.preprocess_frame(obs)
                
                for _ in range(self.stack_frames):
                    self.frame_buffer.append(processed_frame)
                    
                return self._get_stacked_frames()
            except Exception as e:
                print(f"Error during reset (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    print("Recreating environment and trying again...")
                    self.create_env()
                    time.sleep(0.5)
                else:
                    print("Max reset attempts reached, creating empty observation")
                    return np.zeros(16 * 16 * self.stack_frames, dtype=np.float32)
    
    def step(self, action):
        try:
            obs, reward, done, info = self.env.step(action)
            
            processed_frame = self.preprocess_frame(obs)
            self.frame_buffer.append(processed_frame)
            
            custom_reward = self._shape_reward(reward, info, done)
            
            return self._get_stacked_frames(), custom_reward, done, info
            
        except Exception as e:
            print(f"Error during step: {e}, recreating environment")
            success = self.create_env()
            
            if not success:
                print("Could not recreate environment, returning terminal state")
                return np.zeros(16 * 16 * self.stack_frames, dtype=np.float32), 0, True, {"error": str(e)}
            
            try:
                obs = self.env.reset()
                processed_frame = self.preprocess_frame(obs)
                for _ in range(self.stack_frames):
                    self.frame_buffer.append(processed_frame)
                return self._get_stacked_frames(), 0, False, {"reset": True}
            except Exception as e:
                print(f"Error after recreation: {e}")
                return np.zeros(16 * 16 * self.stack_frames, dtype=np.float32), 0, True, {"error": str(e)}
    
    def _get_stacked_frames(self):
        return np.array(self.frame_buffer).reshape(-1) / 255.0
    
    def _shape_reward(self, reward, info, done):
        progress_reward = 0
        milestone_bonus = 0
        
        # Position-based rewards
        if 'x_pos' in info:
            current_x = info['x_pos']
            
            progress = current_x - self.last_x
            progress_reward = progress * 0.2
            
            if current_x > self.max_position_ever:
                new_progress = current_x - self.max_position_ever
                milestone_bonus = new_progress * 0.5
                self.max_position_ever = current_x
                
                # Additional milestone bonuses at specific intervals
                for milestone in range(int(self.max_position_ever / 100) * 100, 
                                      int(current_x / 100) * 100 + 100, 
                                      100):
                    if milestone > 0 and milestone not in self.position_milestones:
                        self.position_milestones.add(milestone)
                        milestone_bonus += 25  # Fixed bonus for each 100-unit milestone
            
            self.last_x = current_x
        
        if 'y_pos' in info:
            current_y = info['y_pos']
            y_diff = self.last_y - current_y
            if y_diff > 10:  # Reward significant jumps
                progress_reward += 2
            self.last_y = current_y
        
        # Flag capture bonus
        if 'flag_get' in info and info['flag_get']:
            progress_reward += 500
        
        # Reduce death penalty to encourage exploration
        if done and info.get('life', 1) <= 0:
            if self.last_status != info.get('status', ''):
                progress_reward -= 40  # Reduced from 80
            else:
                progress_reward -= 20  # Reduced from 40
            self.last_status = info.get('status', '')
        
        # Coin collection bonus
        if 'coins' in info:
            if info['coins'] > self.last_coins:
                coin_bonus = (info['coins'] - self.last_coins) * 15  # Increased from 10
                progress_reward += coin_bonus
            self.last_coins = info['coins']
            
        return reward + progress_reward + milestone_bonus
    
    def render(self):
        """Render the current frame"""
        try:
            return self.env.render()
        except Exception as e:
            print(f"Error during render: {e}")
            return None
            
    def close(self):
        """Close the environment"""
        try:
            self.env.close()
        except:
            pass
            
def create_mario_env(stack_frames=4, render_mode=None):
    """Create a more stable Mario environment"""
    # Ignore render_mode in older versions
    return MarioRLEnv(stack_frames=stack_frames)