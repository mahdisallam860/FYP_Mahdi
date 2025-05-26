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

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.00015, gamma=0.98, 
                 epsilon=0.5, epsilon_decay=0.9997, epsilon_min=0.05, 
                 batch_size=128, memory_size=100000):
        super(DQNAgent, self).__init__()
        
        # Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # Neural Networks
        self.model = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_size)
        )
        
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_size)
        )
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def validate_state(self, state):
        """Validate and fix state dimensions."""
        if state is None:
            return np.zeros(self.state_size, dtype=np.float32)
            
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
            
        if len(state) != self.state_size:
            rospy.logwarn(f"Invalid state size: {len(state)}, expected: {self.state_size}")
            # Pad or truncate to correct size
            if len(state) < self.state_size:
                state = np.pad(state, (0, self.state_size - len(state)), 
                             mode='constant', constant_values=0)
            else:
                state = state[:self.state_size]
                
        return state.astype(np.float32)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory with validation."""
        state = self.validate_state(state)
        next_state = self.validate_state(next_state)
        
        if state is not None and next_state is not None:
            self.memory.append([state, action, reward, next_state, done])

    def select_action(self, state):
        """Select action with validated state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = self.validate_state(state)
        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            q_values = self(state)
        self.train()
        return q_values.argmax().item()

    def replay(self):
        """Enhanced training with prioritized sampling and robust validation"""
        if len(self.memory) < self.batch_size:
            return 0
            
        try:
            # Split batch sampling between recent and older experiences
            memory_list = list(self.memory)
            recent_size = min(len(memory_list) // 2, 5000)  # Use last 5000 experiences max
            
            # Get half the batch from recent experiences
            recent_batch_size = self.batch_size // 2
            recent_samples = random.sample(memory_list[-recent_size:], recent_batch_size)
            
            # Get other half from entire memory
            old_batch_size = self.batch_size - recent_batch_size
            old_samples = random.sample(memory_list[:-recent_size], old_batch_size)
            
            # Combine samples
            mini_batch = recent_samples + old_samples
            
            # Validate states with enhanced error checking
            valid_batch = []
            for transition in mini_batch:
                try:
                    state = self.validate_state(transition[0])
                    next_state = self.validate_state(transition[3])
                    
                    if (len(state) == self.state_size and 
                        len(next_state) == self.state_size and 
                        not np.any(np.isnan(state)) and 
                        not np.any(np.isnan(next_state))):
                        valid_batch.append([state, transition[1], transition[2], 
                                        next_state, transition[4]])
                except Exception as e:
                    rospy.logdebug(f"Skipped invalid transition: {str(e)}")
                    continue
            
            # Check for sufficient valid samples
            if len(valid_batch) < self.batch_size * 0.8:  # Allow up to 20% invalid
                rospy.logwarn(f"Insufficient valid transitions: {len(valid_batch)}/{self.batch_size}")
                return 0
                
            # Convert to tensors with error checking
            try:
                states = torch.FloatTensor(np.vstack([x[0] for x in valid_batch])).to(self.device)
                actions = torch.LongTensor([x[1] for x in valid_batch]).to(self.device)
                rewards = torch.FloatTensor([x[2] for x in valid_batch]).to(self.device)
                next_states = torch.FloatTensor(np.vstack([x[3] for x in valid_batch])).to(self.device)
                dones = torch.FloatTensor([x[4] for x in valid_batch]).to(self.device)
                
                # Verify tensor shapes
                batch_size = states.shape[0]
                if (states.shape[1] != self.state_size or 
                    next_states.shape[1] != self.state_size):
                    rospy.logwarn("Invalid tensor shapes detected")
                    return 0
                    
            except Exception as e:
                rospy.logerr(f"Error creating tensors: {str(e)}")
                return 0
                
            # Compute Q values with gradient tracking
            self.train()
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Compute target Q values without gradient tracking
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
                target_q = rewards + (self.gamma * next_q * (1 - dones))
                
            # Compute loss with huber loss for stability
            loss = F.smooth_l1_loss(current_q, target_q)  # Changed from MSE to Huber loss
            
            # Optimize with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            # Increased max norm for potentially larger gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            # Update epsilon with minimum bound check
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, 
                                self.epsilon * self.epsilon_decay)
            
            return loss.item()
            
        except Exception as e:
            rospy.logerr(f"Error in replay: {str(e)}")
            return 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, name)

    def load(self, path):
        """Load model weights from a specified file path."""
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Selectively load model weights
            self.model.load_state_dict({k: v for k, v in checkpoint['model_state_dict'].items() if k in self.model.state_dict()})
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            
            self.update_target_network()  # Update target network with loaded weights
            rospy.loginfo(f"Successfully loaded model weights from {path}")
        else:
            rospy.logwarn(f"Model file not found at {path}. Starting training from scratch.")

    def forward(self, x):
        return self.model(x)