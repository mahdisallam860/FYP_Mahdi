#!/usr/bin/env python3

import torch
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

def load_and_plot_metrics(checkpoint_dir, episode_num):
    # Set style for better visualization
    plt.style.use('default')
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    # Load metrics file
    metrics_file = os.path.join(checkpoint_dir, f"metrics_episode_{episode_num}.json")
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode_num}.pth")
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Load checkpoint for additional info
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - Episode {episode_num}', fontsize=16)
    
    # 1. Plot rewards and moving average
    ax1.plot(metrics['episodes'], metrics['rewards'], color=colors[0], alpha=0.3, label='Episode Reward')
    # Calculate and plot moving average
    window_size = 50
    moving_avg = np.convolve(metrics['rewards'], np.ones(window_size)/window_size, mode='valid')
    ax1.plot(metrics['episodes'][window_size-1:], moving_avg, color=colors[1], label='50-Episode Moving Average')
    ax1.set_title('Rewards over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot average rewards
    ax2.plot(metrics['episodes'], metrics['avg_rewards'], color=colors[1])
    ax2.set_title('Average Reward Progress')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, alpha=0.3)
    
    # 3. Plot steps per episode
    ax3.plot(metrics['episodes'], metrics['steps'], color=colors[2])
    ax3.set_title('Steps per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.grid(True, alpha=0.3)
    
    # 4. Plot training losses
    if metrics['losses']:  # Check if losses exist
        filtered_losses = [x for x in metrics['losses'] if x != 0]  # Remove zero losses
        if filtered_losses:
            ax4.plot(filtered_losses, color=colors[3], alpha=0.7)
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Training Iteration')
            ax4.set_ylabel('Loss')
            ax4.grid(True, alpha=0.3)
    
    # Add epsilon value from checkpoint
    epsilon_text = f"Current ε: {checkpoint['epsilon']:.4f}"
    fig.text(0.02, 0.02, epsilon_text, fontsize=10)
    
    # Add some statistics as text
    stats_text = (
        f"Best Average Reward: {checkpoint['best_average_reward']:.2f}\n"
        f"Latest Average Reward: {metrics['avg_rewards'][-1]:.2f}\n"
        f"Latest Episode Reward: {metrics['rewards'][-1]:.2f}\n"
        f"Average Steps/Episode: {np.mean(metrics['steps']):.1f}"
    )
    fig.text(0.02, 0.95, stats_text, fontsize=10, verticalalignment='top')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    plot_path = os.path.join(checkpoint_dir, f"training_progress_{episode_num}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Display some summary statistics
    print("\nTraining Summary:")
    print("-" * 50)
    print(f"Episodes Completed: {episode_num}")
    print(f"Current Epsilon: {checkpoint['epsilon']:.4f}")
    print(f"Best Average Reward: {checkpoint['best_average_reward']:.2f}")
    print(f"Final Average Reward: {metrics['avg_rewards'][-1]:.2f}")
    print(f"Average Steps per Episode: {np.mean(metrics['steps']):.1f}")
    
    return fig

if __name__ == "__main__":
    checkpoint_dir = "/home/alma/catkin_ws/src/dqn3/checkpoints"
    episode_num = 700
    
    try:
        fig = load_and_plot_metrics(checkpoint_dir, episode_num)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")