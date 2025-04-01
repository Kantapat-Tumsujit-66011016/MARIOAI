import numpy as np
import random
from collections import deque
import torch

class PrioritizedReplayBuffer:
    
    def __init__(self, buffer_size=10000, batch_size=64, alpha=0.7, beta=0.4, beta_increment=0.001, seed=42):
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha  
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.experience_count = 0
        random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        self.priorities.append(self.max_priority)
        self.experience_count += 1
    
    def sample(self):
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)
        
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize weights
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in indices:
            experience = self.memory[i]
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(experience[3])
            dones.append(experience[4])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones, weights, indices)
    
    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            # Add small constant to avoid zero priority
            priority = abs(td_error) + 1e-5
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.memory)