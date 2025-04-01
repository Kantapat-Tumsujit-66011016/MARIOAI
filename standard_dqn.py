import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, output_size=7, dropout_rate=0.2):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0) 
        
        return self.network(x)

class DQNAgent:
    def __init__(self, input_size=1024, hidden_size=256, output_size=7, learning_rate=0.0001, 
                 gamma=0.99, tau=0.001, entropy_beta=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        
        self.output_size = output_size
        
        self.q_network = QNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_network = QNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # 'none' to apply weights per sample
        
        self.gamma = gamma 
        self.tau = tau 
        self.entropy_beta = entropy_beta 
        
        self.train_count = 0
        self.update_count = 0
        
        self.action_counts = np.zeros(output_size)
        self.total_actions = 0
    
    def get_action(self, state, epsilon=0.0, enforce_diversity=False):
        self.total_actions += 1
        
        if random.random() < epsilon:
            action = random.randint(0, 6)
            self.action_counts[action] += 1
            return action
        
        if enforce_diversity and self.total_actions > 100:
            action_probs = self.action_counts / self.total_actions
            if np.max(action_probs) > 0.8: 
                if random.random() < 0.2:
                    action = random.randint(0, 6)
                    self.action_counts[action] += 1
                    return action
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        self.action_counts[action] += 1
        return action
    
    def train(self, experiences):
        if len(experiences) == 5:  
            states, actions, rewards, next_states, dones = experiences
            weights = torch.ones_like(torch.FloatTensor(rewards)).to(self.device)
            indices = None
        else:  
            states, actions, rewards, next_states, dones, weights, indices = experiences
            weights = torch.FloatTensor(weights).to(self.device)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        
        q_values = self.q_network(states).gather(1, actions)
        
       
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(1, keepdim=True)
            
            target_next_q_values = self.target_network(next_states).gather(1, next_actions)

            target_q_values = rewards + (1 - dones) * self.gamma * target_next_q_values
        
        td_errors = (q_values - target_q_values).detach().cpu().numpy()
        
        action_penalty = torch.zeros_like(q_values)
        if actions.size(0) > 0 and self.output_size > 6:  
            left_mask = (actions == 6)
            if left_mask.any():
                action_penalty[left_mask] = 0.5 
        
        losses = self.loss_fn(q_values, target_q_values)
        loss = (losses * weights).mean() + action_penalty.mean()
        
        if self.entropy_beta > 0:
            action_probs = F.softmax(self.q_network(states), dim=1)
            
            if self.output_size > 6: 
                left_probs = action_probs[:, 6]
                left_entropy_penalty = torch.mean(left_probs) * 0.5 
            else:
                left_entropy_penalty = 0
                
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1).mean()
            loss -= self.entropy_beta * entropy  
            loss += left_entropy_penalty  

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_count += 1
        
        if self.train_count % 4 == 0:
            self.soft_update_target_network()
            self.update_count += 1
        
        return loss.item(), td_errors
    
    def soft_update_target_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def hard_update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filename='saved_agents/dqn_agent.pth'):
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_count': self.train_count,
            'update_count': self.update_count,
            'action_counts': self.action_counts,
            'total_actions': self.total_actions
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename='saved_agents/dqn_agent.pth'):
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'train_count' in checkpoint:
                self.train_count = checkpoint['train_count']
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']
            if 'action_counts' in checkpoint:
                self.action_counts = checkpoint['action_counts']
            if 'total_actions' in checkpoint:
                self.total_actions = checkpoint['total_actions']
                
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False