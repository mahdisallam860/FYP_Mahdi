#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import rospy
import os
import json

class DDQNAgent(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, 
                 batch_size=64, memory_size=100000):
        super(DDQNAgent, self).__init__()
        
        # Core Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Enhanced Memory Management
        self.memory = deque(maxlen=memory_size)
        self.priority_memory = deque(maxlen=memory_size//4)  # High priority experiences
        
        # Network Architecture - Designed for transfer learning compatibility
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Initialize target network
        self.target_shared_layers = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.target_value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.target_advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Sync target network initially
        self.update_target_network()
        
        # Training components
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Training metrics
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'q_values': []
        }

    def forward(self, state):
        """Forward pass with dueling architecture."""
        x = self.shared_layers(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def target_forward(self, state):
        """Forward pass for target network."""
        with torch.no_grad():
            x = self.target_shared_layers(state)
            value = self.target_value_stream(x)
            advantage = self.target_advantage_stream(x)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def remember(self, state, action, reward, next_state, done, priority=False):
        """Store experience with optional priority."""
        experience = (state, action, reward, next_state, done)
        if priority:
            self.priority_memory.append(experience)
        else:
            self.memory.append(experience)

    def get_action(self, state, testing=False):
        """Select action using epsilon-greedy with noise for exploration."""
        if not testing and np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.eval()
        with torch.no_grad():
            q_values = self(state)
            
            # Add small noise for exploration
            if not testing:
                noise = torch.randn_like(q_values) * 0.1
                q_values += noise
                
        self.train()
        return q_values.argmax().item()

    def replay(self):
        """Enhanced training with better error handling."""
        if len(self.memory) < self.batch_size:
            return None
            
        losses = []
        for _ in range(4):  # Multiple training iterations per replay
            try:
                # Prioritize recent and important experiences
                if len(self.priority_memory) > self.batch_size // 4:
                    priority_batch = random.sample(self.priority_memory, self.batch_size // 4)
                    regular_batch = random.sample(self.memory, self.batch_size - len(priority_batch))
                    mini_batch = priority_batch + regular_batch
                else:
                    mini_batch = random.sample(self.memory, self.batch_size)
                
                # Safe conversion to numpy arrays first
                state_batch = []
                action_batch = []
                reward_batch = []
                next_state_batch = []
                done_batch = []
                
                for transition in mini_batch:
                    state, action, reward, next_state, done = transition
                    state_batch.append(np.array(state, dtype=np.float32))
                    action_batch.append(action)
                    reward_batch.append(reward)
                    next_state_batch.append(np.array(next_state, dtype=np.float32))
                    done_batch.append(done)
                
                # Convert to tensors with proper shape and type
                states = torch.FloatTensor(np.array(state_batch)).to(self.device)
                actions = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
                rewards = torch.FloatTensor(reward_batch).to(self.device)
                next_states = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
                dones = torch.FloatTensor(done_batch).to(self.device)
                
                # Current Q-values
                current_q_values = self(states).gather(1, actions)
                
                # Target Q-values with double DQN
                with torch.no_grad():
                    # Select actions using online network
                    next_actions = self(next_states).argmax(1).unsqueeze(1)
                    # Evaluate actions using target network
                    next_q_values = self.target_forward(next_states).gather(1, next_actions)
                    target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))
                
                # Compute loss with Huber loss for stability
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                losses.append(loss.item())
                
            except Exception as e:
                rospy.logwarn(f"Error during replay: {str(e)}")
                continue
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return np.mean(losses) if losses else None
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_shared_layers.load_state_dict(self.shared_layers.state_dict())
        self.target_value_stream.load_state_dict(self.value_stream.state_dict())
        self.target_advantage_stream.load_state_dict(self.advantage_stream.state_dict())

    def save_model(self, path, episode, metrics=None):
        """Save model with additional training information."""
        try:
            save_dict = {
                'shared_layers_state_dict': self.shared_layers.state_dict(),
                'value_stream_state_dict': self.value_stream.state_dict(),
                'advantage_stream_state_dict': self.advantage_stream.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episode': episode,
                'architecture': {
                    'state_size': self.state_size,
                    'action_size': self.action_size
                }
            }
            
            if metrics:
                save_dict['metrics'] = metrics
                
            # Save temporarily first
            temp_path = path + '.tmp'
            torch.save(save_dict, temp_path)
            os.replace(temp_path, path)
            
            rospy.loginfo(f"Model saved successfully to {path}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error saving model: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def load_model(self, path, training=True):
        """Load model with architecture verification for transfer learning."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Verify architecture compatibility
            loaded_arch = checkpoint.get('architecture', {})
            if loaded_arch.get('state_size') != self.state_size:
                rospy.logwarn(f"State size mismatch: {loaded_arch.get('state_size')} vs {self.state_size}")
                return False
                
            # Load network weights
            self.shared_layers.load_state_dict(checkpoint['shared_layers_state_dict'])
            self.value_stream.load_state_dict(checkpoint['value_stream_state_dict'])
            self.advantage_stream.load_state_dict(checkpoint['advantage_stream_state_dict'])
            
            if training:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                
            self.update_target_network()
            rospy.loginfo(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error loading model: {str(e)}")
            return False

    def get_network_stats(self):
        """Get network statistics for monitoring."""
        with torch.no_grad():
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            # Get gradient statistics
            grad_norms = []
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
                    
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'gradient_norms': grad_norms if grad_norms else None,
                'epsilon': self.epsilon,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }

    def transfer_knowledge(self, target_agent):
        """Transfer learned features to another agent for Stage 2."""
        try:
            # Transfer shared layers
            target_agent.shared_layers.load_state_dict(self.shared_layers.state_dict())
            
            # Initialize new parts of target network with transferred knowledge
            if hasattr(target_agent, 'value_stream'):
                target_agent.value_stream[0].weight.data = self.value_stream[0].weight.data.clone()
                target_agent.value_stream[0].bias.data = self.value_stream[0].bias.data.clone()
                
            if hasattr(target_agent, 'advantage_stream'):
                target_agent.advantage_stream[0].weight.data = self.advantage_stream[0].weight.data.clone()
                target_agent.advantage_stream[0].bias.data = self.advantage_stream[0].bias.data.clone()
            
            # Update target network
            target_agent.update_target_network()
            
            rospy.loginfo("Knowledge transfer completed successfully")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error during knowledge transfer: {str(e)}")
            return False