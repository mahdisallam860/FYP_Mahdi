#!/usr/bin/env python3

import rospy
import numpy as np
import os
import rospkg
import pickle
import torch
import signal
import json
from collections import deque
from random import randrange
from geometry_msgs.msg import Twist
from turtlebot3_env import TurtleBot3Env
from nodes.dqn_agent2 import DQNAgent
from pathlib import Path
import tempfile
import shutil

class TrainingManager:
    def __init__(self):
        # Setup paths
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('dqn3')
        self.model_dir = os.path.join(self.pkg_path, "models")
        self.checkpoint_dir = os.path.join(self.pkg_path, "checkpoints")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training parameters
        self.BATCH_SIZE = 128  # Keep as is
        self.MEMORY_SIZE = 100000  # Keep as is
        self.INITIAL_EPSILON = 0.5  # Reduced from 0.7 for more exploitation of Stage 1 knowledge
        self.EPSILON_DECAY = 0.9997  # Slower decay (from 0.9995)
        self.EPSILON_MIN = 0.05  # Lower minimum (from 0.1)
        self.GAMMA = 0.98  # Increased from 0.95 for better long-term planning
        self.LEARNING_RATE = 0.00015  # Slightly increased from 0.0001
        self.UPDATE_TARGET_EVERY = 5  # More frequent updates (from 10)
        self.WARMUP_EPISODES = 5  # Reduced from 10
        self.CHECKPOINT_FREQUENCY = 25  # Save every 50 episodes
        
        # Initialize training state
        self.episode = 0
        self.best_average_reward = float('-inf')
        self.no_improvement_count = 0
        self.reward_window = deque(maxlen=50)
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'avg_rewards': [],
            'steps': [],
            'losses': []
        }
        
        # Setup interruption handling
        self.interrupted = False
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        rospy.loginfo("Interrupt received, saving checkpoint and cleaning up...")
        self.interrupted = True
        
    def safe_save(self, data, filepath):
        """Safely save data using a temporary file."""
        temp_path = filepath + '.tmp'
        try:
            # Save to temporary file first
            if isinstance(data, dict) and 'model_state_dict' in data:
                torch.save(data, temp_path)
            elif isinstance(data, dict):
                with open(temp_path, 'w') as f:
                    json.dump(data, f)
            
            # Atomic rename
            shutil.move(temp_path, filepath)
            return True
        except Exception as e:
            rospy.logwarn(f"Error during save operation: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
            
    def save_checkpoint(self, agent, forced=False):
        """Save training state and model checkpoint."""
        if self.episode % self.CHECKPOINT_FREQUENCY == 0 or forced:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{self.episode}.pth")
            metrics_path = os.path.join(self.checkpoint_dir, f"metrics_episode_{self.episode}.json")
            
            # Save model checkpoint
            checkpoint_data = {
                'episode': self.episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_average_reward': self.best_average_reward,
                'reward_window': list(self.reward_window)
            }
            self.safe_save(checkpoint_data, checkpoint_path)
            
            # Save training metrics
            self.safe_save(self.training_metrics, metrics_path)
            
            # Clean up old checkpoints (keep last 3)
            self.cleanup_old_checkpoints()
            
    def cleanup_old_checkpoints(self):
        """Keep only the last 3 checkpoints to save space."""
        checkpoints = sorted(Path(self.checkpoint_dir).glob("checkpoint_episode_*.pth"))
        if len(checkpoints) > 3:
            for checkpoint in checkpoints[:-3]:
                try:
                    checkpoint.unlink()
                    metrics_file = checkpoint.with_name(checkpoint.stem.replace('checkpoint', 'metrics')).with_suffix('.json')
                    if metrics_file.exists():
                        metrics_file.unlink()
                except Exception as e:
                    rospy.logwarn(f"Error cleaning up old checkpoint: {e}")

    def train_stage2(self):
        rospy.init_node('train_stage2', anonymous=True)
        rospy.sleep(1)
        
        # Initialize environment and agent
        env = TurtleBot3Env(
            is_training=True,
            goal_position=(3.0, -2.0),
            timeout=700,
            stage=2
        )
        
        state_size = 26
        action_size = 3
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=self.LEARNING_RATE,
            gamma=self.GAMMA,
            epsilon=self.INITIAL_EPSILON,
            epsilon_decay=self.EPSILON_DECAY,
            epsilon_min=self.EPSILON_MIN,
            batch_size=self.BATCH_SIZE,
            memory_size=self.MEMORY_SIZE
        )
        
        # Load previous checkpoint if exists
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            self.load_checkpoint(agent, latest_checkpoint)
        else:
            # Load stage1 weights if no checkpoint exists
            self.load_stage1_weights(agent)
        
        rospy.sleep(1)
        
        # Main training loop
        episodes = 700
        while self.episode < episodes and not self.interrupted and not rospy.is_shutdown():
            episode_metrics = self.run_episode(env, agent)
            self.update_metrics(episode_metrics)
            self.save_checkpoint(agent)
            
            if self.interrupted:
                break
                
        # Final save
        self.save_checkpoint(agent, forced=True)
        rospy.loginfo("Training completed or interrupted. Final checkpoint saved.")
        
    def run_episode(self, env, agent):
        state = env.reset()
        rospy.sleep(0.1)
        
        done = False
        total_reward = 0
        steps = 0
        losses = []
        
        while not done and not self.interrupted and not rospy.is_shutdown():
            # Action selection with noise
            if np.random.random() < 0.1:
                action = randrange(agent.action_size)
            else:
                action = agent.select_action(state)
            
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > self.BATCH_SIZE:
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Update target network if needed
        if self.episode % self.UPDATE_TARGET_EVERY == 0:
            agent.update_target_network()
            rospy.loginfo("Target network updated")
        
        return {
            'steps': steps,
            'total_reward': total_reward,
            'losses': losses
        }
        
    def update_metrics(self, episode_metrics):
        self.reward_window.append(episode_metrics['total_reward'])
        current_avg_reward = np.mean(list(self.reward_window))
        
        # Update training metrics
        self.training_metrics['episodes'].append(self.episode)
        self.training_metrics['rewards'].append(episode_metrics['total_reward'])
        self.training_metrics['avg_rewards'].append(current_avg_reward)
        self.training_metrics['steps'].append(episode_metrics['steps'])
        self.training_metrics['losses'].append(
            np.mean(episode_metrics['losses']) if episode_metrics['losses'] else 0
        )
        
        # Log progress
        rospy.loginfo(
            f"Episode {self.episode + 1}: Steps={episode_metrics['steps']}, "
            f"Reward={episode_metrics['total_reward']:.2f}, "
            f"Avg Reward={current_avg_reward:.2f}"
        )
        
        self.episode += 1
        
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint file."""
        checkpoints = sorted(Path(self.checkpoint_dir).glob("checkpoint_episode_*.pth"))
        return checkpoints[-1] if checkpoints else None
        
    def load_checkpoint(self, agent, checkpoint_path):
        """Load training state from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=agent.device)
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = checkpoint['epsilon']
            self.episode = checkpoint['episode']
            self.best_average_reward = checkpoint['best_average_reward']
            self.reward_window = deque(checkpoint['reward_window'], maxlen=50)
            rospy.loginfo(f"Resumed training from checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            rospy.logwarn(f"Error loading checkpoint: {e}")
            return False
            
    def load_stage1_weights(self, agent):
        """Load pre-trained weights from stage 1."""
        stage1_model_path = os.path.join(self.model_dir, "stage1_best_model.pth")
        if os.path.exists(stage1_model_path):
            try:
                state_dict = torch.load(stage1_model_path, map_location=agent.device)
                agent.model.load_state_dict(state_dict['model_state_dict'])
                rospy.loginfo("Successfully loaded Stage 1 weights")
            except Exception as e:
                rospy.logwarn(f"Could not load Stage 1 weights: {e}")

def main():
    trainer = TrainingManager()
    try:
        trainer.train_stage2()
    except rospy.ROSInterruptException:
        trainer.save_checkpoint(trainer.agent, forced=True)
        rospy.loginfo("Training interrupted by ROS. Final checkpoint saved.")

if __name__ == '__main__':
    main()