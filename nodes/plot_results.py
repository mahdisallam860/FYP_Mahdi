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
metrics_dir = "/home/alma/catkin_ws/src/dqn3/logs/stage1_enhanced_dqn"
data = load_all_metrics(metrics_dir)

# Create visualization
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Complete  DQN Stage 1 enhanced Training Analysis', fontsize=16, y=0.95)

# 1. Episode Rewards
window = 50  # Larger window for complete dataset
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax1.plot(data['episodes'], data['rewards'], alpha=0.3, color='blue', label='Raw Rewards')
ax1.plot(data['episodes'][window-1:], moving_average(data['rewards'], window), 
         color='blue', linewidth=2, label=f'{window}-Episode Moving Avg')
ax1.set_title('Episode Rewards Throughout Training')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True)

# 2. Success and Collision Rates
ax2 = plt.subplot2grid((3, 2), (1, 0))
success_rate = moving_average(data['successes'], window) * 100
collision_rate = moving_average(data['collisions'], window) * 100
ax2.plot(data['episodes'][window-1:], success_rate, color='green', label='Success Rate')
ax2.plot(data['episodes'][window-1:], collision_rate, color='red', label='Collision Rate')
ax2.set_title('Success and Collision Rates')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Rate (%)')
ax2.set_ylim([0, 100])
ax2.legend()
ax2.grid(True)

# 3. Steps per Episode
ax3 = plt.subplot2grid((3, 2), (1, 1))
ax3.plot(data['episodes'], data['steps'], alpha=0.3, color='purple', label='Raw Steps')
ax3.plot(data['episodes'][window-1:], moving_average(data['steps'], window),
         color='purple', linewidth=2, label=f'{window}-Episode Moving Avg')
ax3.set_title('Steps per Episode')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Steps')
ax3.legend()
ax3.grid(True)

# 4. Learning Progress
ax4 = plt.subplot2grid((3, 2), (2, 0))
reward_per_step = data['rewards'] / data['steps']
ax4.plot(data['episodes'][window-1:], moving_average(reward_per_step, window),
         color='orange', label='Reward per Step')
ax4.set_title('Learning Efficiency')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Average Reward per Step')
ax4.legend()
ax4.grid(True)

# 5. Performance Metrics
ax5 = plt.subplot2grid((3, 2), (2, 1))
ax5.axis('off')

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

ax5.text(0, 0.95, '\n'.join(metrics_text), 
         fontsize=10, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'stage1_complete_analysis_{timestamp}.png', 
            bbox_inches='tight', dpi=300)
print(f"Complete analysis saved as 'stage1_complete_analysis_{timestamp}.png'")

# Print additional statistics
print("\nTraining Statistics:")
print(f"Total Episodes: {len(data['episodes'])}")
print(f"Best Reward: {best_reward:.2f}")
print(f"Final Success Rate: {final_success_rate:.2f}%")
print(f"Success Rate Improvement: {success_rate[-1] - success_rate[0]:.2f}%")
print(f"Average Steps per Episode: {avg_steps:.2f}")

plt.show()