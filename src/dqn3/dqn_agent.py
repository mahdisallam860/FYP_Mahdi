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

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, 
                 batch_size=64, memory_size=100000):
        super(DQNAgent, self).__init__()
        
        # Core Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Memory Management
        self.memory = deque(maxlen=memory_size)
        
        # Network Architecture
        self.network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_size)
        )
        
        # Target Network
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_size)
        )
        
        self.update_target_network()
        
        # Training components
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'q_values': []
        }

    def forward(self, state):
        return self.network(state)

    def target_forward(self, state):
        with torch.no_grad():
            return self.target_network(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, testing=False):
        if not testing and np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.eval()
        with torch.no_grad():
            q_values = self(state)
            if not testing:
                noise = torch.randn_like(q_values) * 0.1
                q_values += noise
        self.train()
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
            
        losses = []
        for _ in range(4):
            try:
                mini_batch = random.sample(self.memory, self.batch_size)
                
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
                
                states = torch.FloatTensor(np.array(state_batch)).to(self.device)
                actions = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
                rewards = torch.FloatTensor(reward_batch).to(self.device)
                next_states = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
                dones = torch.FloatTensor(done_batch).to(self.device)
                
                current_q_values = self(states).gather(1, actions)
                
                with torch.no_grad():
                    next_q_values = self.target_forward(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))
                
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                losses.append(loss.item())
                
            except Exception as e:
                rospy.logwarn(f"Error during replay: {str(e)}")
                continue
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return np.mean(losses) if losses else None

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, path, episode, metrics=None):
        try:
            save_dict = {
                'network_state_dict': self.network.state_dict(),
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
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            loaded_arch = checkpoint.get('architecture', {})
            if loaded_arch.get('state_size') != self.state_size:
                rospy.logwarn(f"State size mismatch: {loaded_arch.get('state_size')} vs {self.state_size}")
                return False
                
            self.network.load_state_dict(checkpoint['network_state_dict'])
            
            if training:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                
            self.update_target_network()
            rospy.loginfo(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error loading model: {str(e)}")
            return False