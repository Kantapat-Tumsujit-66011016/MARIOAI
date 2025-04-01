import numpy as np
import cv2
from collections import deque
import time
import gc
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


RIGHT_ONLY_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['LEFT'],
]

class ImprovedMarioEnv:
    def __init__(self, stack_frames=4, restrict_left=True, stuck_penalty=True):
        self.env = None
        self.stack_frames = stack_frames
        self.frame_buffer = deque(maxlen=stack_frames)
        
        # Options
        self.restrict_left = restrict_left  
        self.stuck_penalty = stuck_penalty  
        
        # Tracking variables
        self.last_x = 0
        self.last_y = 0
        self.last_status = ''
        self.last_coins = 0
        self.max_position_ever = 0
        self.position_milestones = set()
        
        # Stuck detection
        self.position_history = deque(maxlen=100)
        self.no_progress_counter = 0
        
        self.create_env()
    
    def create_env(self):
        """Create a fresh environment instance"""
        try:
            if self.env is not None:
                try:
                    self.env.close()
                except:
                    pass
            
            gc.collect()
            time.sleep(0.1)
            
            self.env = gym_super_mario_bros.make("SuperMarioBros-v0")
            
            if self.restrict_left:
                self.env = JoypadSpace(self.env, RIGHT_ONLY_MOVEMENT)
            else:
                self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
            
            # Reset tracking variables
            self.last_x = 0
            self.last_y = 0
            self.last_status = ''
            self.last_coins = 0
            self.position_history.clear()
            self.no_progress_counter = 0
            self.frame_buffer.clear()
            
            return True
        except Exception as e:
            print(f"Error creating environment: {e}")
            time.sleep(0.5)
            return False
    
    def reset(self):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                obs = self.env.reset()
                
                # Clear tracking variables
                self.frame_buffer.clear()
                self.position_history.clear()
                self.no_progress_counter = 0
                self.last_x = 0
                self.last_y = 0
                
                # Process the initial frame
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
            # Execute action in environment
            obs, reward, done, info = self.env.step(action)
            
            # Process frame
            processed_frame = self.preprocess_frame(obs)
            self.frame_buffer.append(processed_frame)
            
            # Apply custom reward shaping
            custom_reward = self._shape_reward(reward, info, done, action)
            
            return self._get_stacked_frames(), custom_reward, done, info
        except Exception as e:
            print(f"Error during step: {e}, recreating environment")
            success = self.create_env()
            
            if not success:
                print("Could not recreate environment, returning terminal state")
                return np.zeros(16 * 16 * self.stack_frames, dtype=np.float32), -10, True, {"error": str(e)}
            
            try:
                obs = self.env.reset()
                processed_frame = self.preprocess_frame(obs)
                for _ in range(self.stack_frames):
                    self.frame_buffer.append(processed_frame)
                return self._get_stacked_frames(), -10, False, {"reset": True}
            except Exception as e:
                print(f"Error after recreation: {e}")
                return np.zeros(16 * 16 * self.stack_frames, dtype=np.float32), -10, True, {"error": str(e)}
    
    def _shape_reward(self, reward, info, done, action):
        progress_reward = 0
        milestone_bonus = 0
        directional_bonus = 0
        stuck_penalty = 0
        exploration_bonus = 0  
        survival_bonus = 0.01 
        
    
        if action == 0:  
            survival_bonus = 0.005  
        
       
        if 'x_pos' in info:
            current_x = info['x_pos']
            
            
            self.position_history.append(current_x)
            
            #Directional reward
            progress = current_x - self.last_x
            if progress > 0:
                progress_reward = progress * 1.5 
                directional_bonus = progress * 0.8  
            elif progress < 0 and not self.restrict_left:
                directional_bonus = max(-2.0, progress * 0.2)
            
            # Stuck detection 
            if self.stuck_penalty and len(self.position_history) >= 50:
                pos_range = max(self.position_history) - min(self.position_history)
                if pos_range < 10: 
                    self.no_progress_counter += 1
                    if self.no_progress_counter > 20:  
                        stuck_penalty = max(-5.0, -0.05 * min(self.no_progress_counter, 50))
                else:
                    self.no_progress_counter = 0  

            #milestone bonus
            if current_x > self.max_position_ever:
                new_progress = current_x - self.max_position_ever
                
                # Add exploration bonus that increases with distance
                if new_progress > 50: 
                    exploration_bonus = 100 
                else:
                    exploration_bonus = new_progress * 1.5  
                
                milestone_bonus = new_progress * 3.0  
                self.max_position_ever = current_x
                
                for milestone in range(int(self.max_position_ever / 100) * 100, 
                                      int(current_x / 100) * 100 + 100, 
                                      100):
                    if milestone > 0 and milestone not in self.position_milestones:
                        self.position_milestones.add(milestone)
                        base_milestone_reward = 250 
                        milestone_multiplier = 1.0 + (milestone / 1000)
                        milestone_bonus += base_milestone_reward * milestone_multiplier
            
            self.last_x = current_x
            
        
        if 'y_pos' in info:
            current_y = info['y_pos']
            y_diff = self.last_y - current_y
            if y_diff > 10: 
                progress_reward += 5 
            self.last_y = current_y
        
        # Flag capture bonus
        if 'flag_get' in info and info['flag_get']:
            progress_reward += 2000  
        
       
        if done and info.get('life', 1) <= 0:
            if self.last_status != info.get('status', ''):
                progress_reward += -15  
            else:
                progress_reward += -10  
            self.last_status = info.get('status', '')
        
        # Coin collection bonus
        if 'coins' in info:
            if info['coins'] > self.last_coins:
                coin_bonus = (info['coins'] - self.last_coins) * 40 
                progress_reward += coin_bonus
            self.last_coins = info['coins']
        
        # Combine all reward components but clamp to reasonable range
        total_reward = reward + progress_reward + milestone_bonus + directional_bonus + stuck_penalty + survival_bonus + exploration_bonus
        
        return max(-50.0, min(2000.0, total_reward))
    
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return resized
    
    def _get_stacked_frames(self):
        return np.array(self.frame_buffer).reshape(-1) / 255.0
    
    def render(self):
        try:
            return self.env.render()
        except Exception as e:
            print(f"Error during render: {e}")
            return None
    
    def close(self):
        try:
            self.env.close()
        except:
            pass

def create_improved_mario_env(stack_frames=4, restrict_left=True, stuck_penalty=True):
    return ImprovedMarioEnv(stack_frames=stack_frames, 
                           restrict_left=restrict_left,
                           stuck_penalty=stuck_penalty)