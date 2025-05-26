#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
import rospy
import json

class DQNNetwork(nn.Module):
    """Neural network architecture designed for transfer learning between stages"""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        # Feature extraction layers (shared between stages)
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        
        # Navigation layers (can be fine-tuned for each stage)
        self.navigation_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Action output layer
        self.action_layer = nn.Linear(64, action_size)
        
        # Value stream for better action selection
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream for better action selection
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, state):
        features = self.feature_layers(state)
        nav_features = self.navigation_layers(features)
        
        # Dueling DQN architecture
        value = self.value_stream(nav_features)
        advantages = self.advantage_stream(nav_features)
        
        # Combine value and advantages using dueling architecture
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class DQNAgent:
    """DQN Agent with transfer learning capabilities"""
    def __init__(self, state_size, action_size, stage=1, device="cuda"):
        self.state_size = state_size
        self.action_size = action_size
        self.stage = stage
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.memory_size = 100000
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001   # Soft update parameter
        self.learning_rate = 0.0001
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Prioritized Experience Replay
        self.memory = deque(maxlen=self.memory_size)
        self.priority_scale = 0.7  # For prioritized sampling
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                  lr=self.learning_rate,
                                  weight_decay=1e-5)  # L2 regularization
                                  
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'losses': [],
            'avg_q_values': []
        }

    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy with noisy exploration"""
        if training and random.random() < self.epsilon:
            # Add noise to random actions for better exploration
            action = random.randrange(self.action_size)
            return action
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)
            self.policy_net.train()
            
            # Add small noise to Q-values for exploration
            if training:
                noise = torch.randn_like(q_values) * 0.1
                q_values += noise
                
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory with priority"""
        # Calculate TD error for priority
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            current_q = self.policy_net(state_tensor)[0][action]
            next_q = self.target_net(next_state_tensor).max(1)[0]
            target_q = reward + (1 - done) * self.gamma * next_q
            
            priority = abs(target_q - current_q).item() ** self.priority_scale
            
        self.memory.append((state, action, reward, next_state, done, priority))

    def replay(self):
        """Train the network using prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample batch based on priorities
        priorities = np.array([exp[5] for exp in self.memory])
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.memory), 
                                 self.batch_size, 
                                 p=probabilities,
                                 replace=False)
        batch = [self.memory[idx] for idx in indices]
        
        # Prepare batch tensors
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp[4] for exp in batch]).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            # Double DQN
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network (soft update)
        self._update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Track metrics
        self.training_metrics['losses'].append(loss.item())
        self.training_metrics['avg_q_values'].append(current_q_values.mean().item())
        
        return loss.item()

    def _update_target_network(self):
        """Soft update of target network"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + 
                                  (1 - self.tau) * target_param.data)

    def save_model(self, path, episode):
        """Save model with additional information for transfer learning"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stage': self.stage,
            'episode': episode,
            'training_metrics': self.training_metrics
        }
        
        torch.save(save_dict, path)
        
        # Save training metrics separately for analysis
        metrics_path = path.replace('.pth', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f)

    def load_model(self, path, transfer_learning=False):
        """Load model with support for transfer learning"""
        if not os.path.exists(path):
            rospy.logwarn(f"Model file not found: {path}")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            if transfer_learning:
                # Load only feature extraction layers for transfer learning
                policy_dict = self.policy_net.state_dict()
                target_dict = self.target_net.state_dict()
                
                # Filter feature extraction layers
                for key in checkpoint['policy_net_state_dict']:
                    if 'feature_layers' in key:
                        policy_dict[key] = checkpoint['policy_net_state_dict'][key]
                        target_dict[key] = checkpoint['target_net_state_dict'][key]
                        
                self.policy_net.load_state_dict(policy_dict)
                self.target_net.load_state_dict(target_dict)
                
                # Reset navigation layers for fine-tuning
                self.epsilon = 1.0  # Reset exploration for new stage
            else:
                # Load complete model
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                
            rospy.loginfo(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error loading model: {str(e)}")
            return False

    def get_model_summary(self):
        """Get model architecture and training summary"""
        return {
            'architecture': str(self.policy_net),
            'parameters': sum(p.numel() for p in self.policy_net.parameters()),
            'device': str(self.device),
            'metrics': self.training_metrics
        }

    def adjust_learning_rate(self, factor=0.5):
        """Adjust learning rate during training"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor
            rospy.loginfo(f"Learning rate adjusted to: {param_group['lr']}")