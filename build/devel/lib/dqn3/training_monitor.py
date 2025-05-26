#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class TrainingMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.epsilons = []
        
        # Moving averages
        self.reward_window = deque(maxlen=window_size)
        self.success_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)

    def add_episode_data(self, reward, success, episode_length, epsilon):
        self.rewards.append(reward)
        self.success_rates.append(float(success))
        self.episode_lengths.append(episode_length)
        self.epsilons.append(epsilon)
        
        self.reward_window.append(reward)
        self.success_window.append(float(success))
        self.length_window.append(episode_length)

    def plot_training_progress(self, save_path=None):
        if len(self.rewards) == 0:
            print("No data to plot yet.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.rewards, alpha=0.3, label='Raw')
        if len(self.rewards) >= self.window_size:
            moving_avg = np.convolve(self.rewards, 
                                   np.ones(self.window_size)/self.window_size, 
                                   mode='valid')
            axes[0, 0].plot(moving_avg, label=f'{self.window_size}-ep Moving Avg')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        
        # Plot success rate
        if len(self.success_rates) > 0:
            running_success = np.cumsum(self.success_rates)/(np.arange(len(self.success_rates))+1)
            axes[0, 1].plot(running_success)
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
        
        # Plot episode lengths
        axes[1, 0].plot(self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) >= self.window_size:
            moving_avg = np.convolve(self.episode_lengths, 
                                   np.ones(self.window_size)/self.window_size, 
                                   mode='valid')
            axes[1, 0].plot(moving_avg, label=f'{self.window_size}-ep Moving Avg')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        
        # Plot epsilon
        if len(self.epsilons) > 0:
            axes[1, 1].plot(self.epsilons)
            axes[1, 1].set_title('Exploration Rate (Epsilon)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def get_current_stats(self):
        """Get current training statistics"""
        if len(self.rewards) == 0:
            return "No training data available yet."
            
        recent_rewards = self.rewards[-100:] if len(self.rewards) > 100 else self.rewards
        recent_success = self.success_rates[-100:] if len(self.success_rates) > 100 else self.success_rates
        
        stats = {
            'episodes_completed': len(self.rewards),
            'recent_avg_reward': np.mean(recent_rewards),
            'recent_success_rate': np.mean(recent_success),
            'best_reward': max(self.rewards) if self.rewards else float('-inf'),
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'current_epsilon': self.epsilons[-1] if self.epsilons else 1.0
        }
        
        return stats