#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ----------------------------
# Academic-Style Plotting Setup
# ----------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': False,  # Remove grid lines
    'figure.figsize': (10, 6),
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ----------------------------
# Data Loading Function
# ----------------------------
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
    
    # Load and combine data from each file
    for filename in metrics_files:
        with open(os.path.join(directory, filename), 'r') as f:
            metrics = json.load(f)
            episode = metrics['episode']
            
            # Append the metrics data
            all_rewards.extend(metrics['recent_rewards'])
            all_successes.extend(metrics['recent_successes'])
            all_collisions.extend(metrics['recent_collisions'])
            all_steps.extend(metrics['recent_steps'])
            
            # Record best reward and best success rate
            best_rewards.append(metrics['best_reward'])
            success_rates.append(metrics['best_success_rate'])
            
            # Create episode numbers for each reward entry in this file
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

# ----------------------------
# Partial Moving Average Function
# ----------------------------
def moving_average_partial(data, window):
    """
    Compute a moving average that uses all available data up to each index.
    For indices where a full window is not available, it averages only the existing points.
    """
    out = np.empty(len(data), dtype=float)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out[i] = np.mean(data[start:i+1])
    return out

# ----------------------------
# Helper Function for Academic Styling on Axes
# ----------------------------
def apply_academic_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

# ----------------------------
# Main Script
# ----------------------------
# 1. Load metrics data
metrics_dir = "/home/alma/catkin_ws/src/dqn3/logs/stage1_enhanced"
data = load_all_metrics(metrics_dir)

# 2. Adjust episodes so that the earliest episode becomes 0
min_episode = np.min(data['episodes'])
adjusted_episodes = data['episodes'] - min_episode

# 3. Set the moving average window and define an academic color palette
window = 50
colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442']

# ----------------------------
# 4. Plot: Episode Rewards
# ----------------------------
fig, ax = plt.subplots()

# Plot raw rewards
ax.plot(adjusted_episodes, data['rewards'], alpha=0.3, color=colors[0], label='Raw Rewards')

# Plot moving average rewards using the partial moving average
ma_rewards = moving_average_partial(data['rewards'], window)
ax.plot(adjusted_episodes, ma_rewards, color=colors[0], linewidth=2,
        label=f'{window}-Episode Moving Avg')

ax.set_title('Episode Rewards Throughout Training')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_xlim(left=0)
ax.legend()
apply_academic_style(ax)
plt.tight_layout()
plt.savefig('episode_rewards.png', dpi=300)
plt.close()

# ----------------------------
# 5. Plot: Success and Collision Rates
# ----------------------------
fig, ax = plt.subplots()

# Multiply by 100 to get percentages and compute the partial moving average
ma_success = moving_average_partial(data['successes'] * 100, window)
ma_collision = moving_average_partial(data['collisions'] * 100, window)

ax.plot(adjusted_episodes, ma_success, color=colors[1], label='Success Rate')
ax.plot(adjusted_episodes, ma_collision, color=colors[2], label='Collision Rate')

ax.set_title('Success and Collision Rates')
ax.set_xlabel('Episode')
ax.set_ylabel('Rate (%)')
ax.set_ylim([0, 100])
ax.set_xlim(left=0)
ax.legend()
apply_academic_style(ax)
plt.tight_layout()
plt.savefig('success_collision_rates.png', dpi=300)
plt.close()

# ----------------------------
# 6. Plot: Steps per Episode
# ----------------------------
fig, ax = plt.subplots()

ax.plot(adjusted_episodes, data['steps'], alpha=0.3, color=colors[3], label='Raw Steps')
ma_steps = moving_average_partial(data['steps'], window)
ax.plot(adjusted_episodes, ma_steps, color=colors[3], linewidth=2,
        label=f'{window}-Episode Moving Avg')

ax.set_title('Steps per Episode')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps')
ax.set_xlim(left=0)
ax.legend()
apply_academic_style(ax)
plt.tight_layout()
plt.savefig('steps_per_episode.png', dpi=300)
plt.close()

# ----------------------------
# 7. Plot: Learning Efficiency (Reward per Step)
# ----------------------------
fig, ax = plt.subplots()

# Avoid division by zero by ensuring a minimum step count of 1
reward_per_step = data['rewards'] / np.maximum(data['steps'], 1)
ma_efficiency = moving_average_partial(reward_per_step, window)

ax.plot(adjusted_episodes, ma_efficiency, color=colors[4], label='Reward per Step')

ax.set_title('Learning Efficiency')
ax.set_xlabel('Episode')
ax.set_ylabel('Average Reward per Step')
ax.set_xlim(left=0)
ax.legend()
apply_academic_style(ax)
plt.tight_layout()
plt.savefig('learning_efficiency.png', dpi=300)
plt.close()

# ----------------------------
# 8. Plot: Performance Metrics Summary (Text Plot)
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Final statistics from the computed moving averages (which now use partial windows)
final_success_rate = ma_success[-1] if len(ma_success) > 0 else 0
avg_steps = np.mean(data['steps'])
best_reward = np.max(data['rewards'])

start_success_rate = ma_success[0] if len(ma_success) > 0 else 0
success_improvement = final_success_rate - start_success_rate

metrics_text = [
    "Training Summary:",
    f"Total Episodes: {len(data['episodes'])}",
    f"Best Reward: {best_reward:.1f}",
    f"Final Success Rate: {final_success_rate:.1f}%",
    f"Final Collision Rate: {ma_collision[-1]:.1f}%",
    f"Average Steps/Episode: {avg_steps:.1f}",
    f"Total Successful Episodes: {int(np.sum(data['successes']))}",
    f"Total Collisions: {int(np.sum(data['collisions']))}",
    f"Number of Checkpoints: {data['checkpoints']}",
    "",
    "Progress:",
    f"Starting Success Rate: {start_success_rate:.1f}%",
    f"Ending Success Rate: {final_success_rate:.1f}%",
    f"Success Rate Improvement: {success_improvement:.1f}%"
]

ax.text(0.1, 0.5, '\n'.join(metrics_text), fontsize=10, verticalalignment='center',
        family='serif', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
ax.set_title('Training Performance Summary')
plt.tight_layout()
plt.savefig('performance_summary.png', dpi=300)
plt.close()

# ----------------------------
# 9. Print Additional Statistics
# ----------------------------
print("Plots saved with academic styling:")
print(" - episode_rewards.png")
print(" - success_collision_rates.png")
print(" - steps_per_episode.png")
print(" - learning_efficiency.png")
print(" - performance_summary.png")
print("")
print("Training Statistics:")
print(f"Total Episodes: {len(data['episodes'])}")
print(f"Best Reward: {best_reward:.2f}")
print(f"Final Success Rate: {final_success_rate:.2f}%")
print(f"Success Rate Improvement: {success_improvement:.2f}%")
print(f"Average Steps per Episode: {avg_steps:.2f}")
