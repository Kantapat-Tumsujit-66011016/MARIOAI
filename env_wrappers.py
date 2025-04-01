import numpy as np
import cv2
from collections import deque
import gym
from gym import spaces

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkipWrapper, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, done, info

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super(FrameStackWrapper, self).__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(16, 16, num_stack),  
            dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        processed_frame = self._process_frame(obs)
        
        for _ in range(self.num_stack):
            self.frames.append(processed_frame)
            
        return self._get_observation()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed_frame = self._process_frame(obs)
        self.frames.append(processed_frame)
        
        return self._get_observation(), reward, done, info
    
    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return resized
    
    def _get_observation(self):
        return np.array(self.frames).reshape(-1) / 255.0

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)
        self.last_x = 0
        self.last_y = 0
        self.last_status = ''
        self.last_coins = 0
        
    def reset(self):
        self.last_x = 0
        self.last_y = 0
        self.last_status = ''
        self.last_coins = 0
        return self.env.reset()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        progress_reward = 0
        
        if 'x_pos' in info:
            current_x = info['x_pos']
            
            progress = current_x - self.last_x
            progress_reward = progress * 0.2
            self.last_x = current_x
        
        if 'y_pos' in info:
            current_y = info['y_pos']
            y_diff = self.last_y - current_y
            if y_diff > 10:  
                progress_reward += 2
            self.last_y = current_y
        
        if 'flag_get' in info and info['flag_get']:
            progress_reward += 500
        
        if done and info.get('life', 1) <= 0:
            if self.last_status != info.get('status', ''):
                progress_reward -= 80
            else:
                progress_reward -= 40
            self.last_status = info.get('status', '')
        
        if 'coins' in info:
            if info['coins'] > self.last_coins:
                progress_reward += 10
            self.last_coins = info['coins']
            
        custom_reward = reward + progress_reward
        
        return obs, custom_reward, done, info

def make_mario_env(env_id="SuperMarioBros-v0", stack_frames=4, skip_frames=4):
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    
    try:
        env = gym_super_mario_bros.make(env_id)
        
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        if skip_frames > 1:
            env = FrameSkipWrapper(env, skip=skip_frames)
        
        env = RewardWrapper(env)
        
        if stack_frames > 1:
            env = FrameStackWrapper(env, num_stack=stack_frames)
        
        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        import time
        time.sleep(1.0)
        
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        if skip_frames > 1:
            env = FrameSkipWrapper(env, skip=skip_frames)
        env = RewardWrapper(env)
        if stack_frames > 1:
            env = FrameStackWrapper(env, num_stack=stack_frames)
        return env