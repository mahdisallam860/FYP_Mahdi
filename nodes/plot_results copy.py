#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

def load_all_metrics(directory):
    """Load and combine all metrics files in chronological order."""
    metrics_files = [f for f in os.listdir(directory) if f.startswith('metrics_ep') and f.endswith('.json')]
    
    # Sort files by episode number
    metrics_files.sort(key=lambda x: int(x.split('ep')[1].split('_')[0]))
    
    # Initialize lists to store combined data
    all_rewards = []
    all_successes = []
    all_collisions = []
    all_steps = []
    episode_numbers = []
    best_rewards = []
    success_rates = []
    
    # Load and combine data
    for filename in metrics_files:
        with open(os.path.join(directory, filename), 'r') as f:
            metrics = json.load(f)
            
            # Extract episode number
            episode = metrics['episode']
            
            # Append metrics
            all_rewards.extend(metrics['recent_rewards'])
            all_successes.extend(metrics['recent_successes'])
            all_collisions.extend(metrics['recent_collisions'])
            all_steps.extend(metrics['recent_steps'])
            
            # Record best reward and success rate
            best_rewards.append(metrics['best_reward'])
            success_rates.append(metrics['best_success_rate'])
            
            # Create episode numbers
            episode_numbers.extend(range(episode - len(metrics['recent_rewards']) + 1, episode + 1))
    
    return {
        'episodes': np.array(episode_numbers),
        'rewards': np.array(all_rewards),
        'successes': np.array(all_successes),
        'collisions': np.array(all_collisions),
        'steps': np.array(all_steps),
        'best_rewards': np.array(best_rewards),
        'success_rates': np.array(success_rates),
        'checkpoints': len(metrics_files)
    }

def moving_average(data, window):
    """Calculate moving average with the specified window."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Load all metrics
metrics_dir = "/home/alma/catkin_ws/src/dqn3/logs/stage3ddqn"
data = load_all_metrics(metrics_dir)

# Set window for moving average
window = 50

# 1. Episode Rewards Plot
plt.figure(figsize=(10, 6))
plt.plot(data['episodes'], data['rewards'], alpha=0.3, color='blue', label='Raw Rewards')
plt.plot(data['episodes'][window-1:], moving_average(data['rewards'], window), 
         color='blue', linewidth=2, label=f'{window}-Episode Moving Avg')
plt.title('Episode Rewards Throughout Training')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('episode_rewards.png', dpi=300)
plt.close()

# 2. Success and Collision Rates Plot
plt.figure(figsize=(10, 6))
success_rate = moving_average(data['successes'], window) * 100
collision_rate = moving_average(data['collisions'], window) * 100
plt.plot(data['episodes'][window-1:], success_rate, color='green', label='Success Rate')
plt.plot(data['episodes'][window-1:], collision_rate, color='red', label='Collision Rate')
plt.title('Success and Collision Rates')
plt.xlabel('Episode')
plt.ylabel('Rate (%)')
plt.ylim([0, 100])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('success_collision_rates.png', dpi=300)
plt.close()

# 3. Steps per Episode Plot
plt.figure(figsize=(10, 6))
plt.plot(data['episodes'], data['steps'], alpha=0.3, color='purple', label='Raw Steps')
plt.plot(data['episodes'][window-1:], moving_average(data['steps'], window),
         color='purple', linewidth=2, label=f'{window}-Episode Moving Avg')
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('steps_per_episode.png', dpi=300)
plt.close()

# 4. Learning Efficiency Plot
plt.figure(figsize=(10, 6))
reward_per_step = data['rewards'] / data['steps']
plt.plot(data['episodes'][window-1:], moving_average(reward_per_step, window),
         color='orange', label='Reward per Step')
plt.title('Learning Efficiency')
plt.xlabel('Episode')
plt.ylabel('Average Reward per Step')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_efficiency.png', dpi=300)
plt.close()

# 5. Performance Metrics Summary
plt.figure(figsize=(10, 6))
plt.axis('off')

# Calculate final statistics
final_success_rate = success_rate[-1] if len(success_rate) > 0 else 0
final_collision_rate = collision_rate[-1] if len(collision_rate) > 0 else 0
avg_steps = np.mean(data['steps'])
best_reward = np.max(data['rewards'])

metrics_text = [
    f"Training Summary:",
    f"Total Episodes: {len(data['episodes'])}",
    f"Best Reward: {best_reward:.1f}",
    f"Final Success Rate: {final_success_rate:.1f}%",
    f"Final Collision Rate: {final_collision_rate:.1f}%",
    f"Average Steps/Episode: {avg_steps:.1f}",
    f"Total Successful Episodes: {np.sum(data['successes'])}",
    f"Total Collisions: {np.sum(data['collisions'])}",
    f"Number of Checkpoints: {data['checkpoints']}",
    f"\nProgress:",
    f"Starting Success Rate: {success_rate[0]:.1f}%",
    f"Ending Success Rate: {success_rate[-1]:.1f}%",
    f"Success Rate Improvement: {success_rate[-1] - success_rate[0]:.1f}%"
]

plt.text(0.1, 0.5, '\n'.join(metrics_text), 
         fontsize=10, verticalalignment='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
plt.title('Training Performance Summary')
plt.tight_layout()
plt.savefig('performance_summary.png', dpi=300)
plt.close()

# Print additional statistics
print("Plots saved: episode_rewards.png, success_collision_rates.png, steps_per_episode.png, learning_efficiency.png, performance_summary.png")
print("\nTraining Statistics:")
print(f"Total Episodes: {len(data['episodes'])}")
print(f"Best Reward: {best_reward:.2f}")
print(f"Final Success Rate: {final_success_rate:.2f}%")
print(f"Success Rate Improvement: {success_rate[-1] - success_rate[0]:.2f}%")
print(f"Average Steps per Episode: {avg_steps:.2f}")