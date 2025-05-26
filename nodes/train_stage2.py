#!/usr/bin/env python3

import rospy
import numpy as np
import os
import rospkg
import json
import torch
import signal
import time
import psutil
import GPUtil
from datetime import datetime, timedelta
from collections import deque
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from dqn3.turtlebot3_env import TurtleBot3Env
from dqn3.dqn_agent import DQNAgent

class ResourceMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.start_time = time.time()
        self.gpu_available = torch.cuda.is_available()
        self.metrics = {
            'training_time': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': []
        }
        
    def update(self):
        current_time = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.Process().memory_percent()
        
        self.metrics['training_time'].append(current_time)
        self.metrics['cpu_usage'].append(cpu_percent)
        self.metrics['memory_usage'].append(memory_percent)
        
        if self.gpu_available:
            try:
                gpu = GPUtil.getGPUs()[0]
                self.metrics['gpu_usage'].append(gpu.load * 100)
                self.metrics['gpu_memory'].append(gpu.memoryUsed)
            except Exception as e:
                rospy.logwarn(f"Failed to get GPU metrics: {e}")
                self.metrics['gpu_usage'].append(0)
                self.metrics['gpu_memory'].append(0)
    
    def save_metrics(self, episode):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(self.log_dir, f"resource_metrics_ep{episode}_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        rospy.loginfo(f"Resource metrics saved to {metrics_file}")
    
    def get_summary(self):
        if len(self.metrics['training_time']) == 0:
            return "No metrics collected"
            
        elapsed_time = timedelta(seconds=int(self.metrics['training_time'][-1]))
        avg_cpu = np.mean(self.metrics['cpu_usage'])
        avg_memory = np.mean(self.metrics['memory_usage'])
        
        summary = (f"Training Time: {elapsed_time}\n"
                  f"Avg CPU Usage: {avg_cpu:.1f}%\n"
                  f"Avg Memory Usage: {avg_memory:.1f}%")
        
        if self.gpu_available:
            avg_gpu = np.mean(self.metrics['gpu_usage'])
            avg_gpu_memory = np.mean(self.metrics['gpu_memory'])
            summary += (f"\nAvg GPU Usage: {avg_gpu:.1f}%\n"
                       f"Avg GPU Memory: {avg_gpu_memory:.1f}MB")
        
        return summary

class Stage2Trainer:
    def __init__(self):
        rospy.init_node('turtlebot3_stage2_trainer', anonymous=True)

        # Setup paths
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('dqn3')
        self.model_dir = os.path.join(self.pkg_path, "models", "stage2_enhanced")
        self.log_dir = os.path.join(self.pkg_path, "logs", "stage2_enhanced")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(self.log_dir)

        # Training parameters
        self.params = {
            'max_episodes': 1500,
            'max_steps': 600,
            'batch_size': 64,
            'learning_rate': 0.00002,
            'gamma': 0.99,
            'initial_epsilon': 0.1,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01,
            'memory_size': 150000,
            'target_update_freq': 8,
            'checkpoint_freq': 100,
            'resource_monitor_freq': 10
        }

        # Define obstacles
        self.obstacles = [
            # Inner square
            {'pos': (-1.0, 1.0), 'radius': 0.4},
            {'pos': (1.0, 1.0), 'radius': 0.4},
            {'pos': (-1.0, -1.0), 'radius': 0.4},
            {'pos': (1.0, -1.0), 'radius': 0.4},
            # Outer square
            {'pos': (-2.0, 2.0), 'radius': 0.4},
            {'pos': (2.0, 2.0), 'radius': 0.4},
            {'pos': (-2.0, -2.0), 'radius': 0.4},
            {'pos': (2.0, -2.0), 'radius': 0.4},
            # Cross pattern
            {'pos': (0.0, 2.0), 'radius': 0.4},
            {'pos': (0.0, -2.0), 'radius': 0.4},
            {'pos': (2.0, 0.0), 'radius': 0.4},
            {'pos': (-2.0, 0.0), 'radius': 0.4}
        ]

        # Safe zones for goal generation
        self.safe_zones = [
            {'center': (0.5, 0.5), 'radius': 0.8},
            {'center': (-1.5, 1.5), 'radius': 0.8},
            {'center': (1.5, -1.5), 'radius': 0.8},
            {'center': (-1.5, -1.5), 'radius': 0.8},
            {'center': (1.5, 1.5), 'radius': 0.8}
        ]

        # Initialize tracking variables
        self.best_reward = float('-inf')
        self.best_success_rate = 0.0
        self.episode_rewards = deque(maxlen=100)
        self.success_history = deque(maxlen=100)
        self.collision_history = deque(maxlen=100)
        self.step_history = deque(maxlen=100)
        self.episode_result = None

        # Publishers
        self.result_pub = rospy.Publisher('training_results', Float32MultiArray, queue_size=5)
        self.goal_marker_pub = rospy.Publisher('/goal_visualization', Marker, queue_size=10)
        
        # Initialize environment
        initial_goal = self.generate_random_goal()
        self.env = TurtleBot3Env(
            is_training=True,
            goal_position=initial_goal,
            timeout=self.params['max_steps'],
            stage=2
        )

        # Initialize agent
        self.agent = DQNAgent(
            state_size=28,
            action_size=5,
            learning_rate=self.params['learning_rate'],
            gamma=self.params['gamma'],
            epsilon=self.params['initial_epsilon'],
            epsilon_decay=self.params['epsilon_decay'],
            epsilon_min=self.params['min_epsilon'],
            batch_size=self.params['batch_size'],
            memory_size=self.params['memory_size']
        )

        # Handle interruptions
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        rospy.loginfo("Training interrupted. Cleaning up...")
        self.interrupted = True
        self.resource_monitor.save_metrics("interrupted")
        rospy.loginfo("\nResource Usage Summary:\n" + self.resource_monitor.get_summary())

    def is_position_safe(self, x, y):
        """Check if a position is safe from obstacles."""
        for obstacle in self.obstacles:
            dist = np.sqrt((x - obstacle['pos'][0])**2 + (y - obstacle['pos'][1])**2)
            if dist < obstacle['radius'] + 0.35:  # Safety margin
                return False
        return True

    def is_in_safe_zone(self, x, y):
        """Check if position is in one of the predefined safe zones."""
        for zone in self.safe_zones:
            dist = np.sqrt((x - zone['center'][0])**2 + (y - zone['center'][1])**2)
            if dist < zone['radius']:
                return True
        return False

    def generate_random_goal(self):
        """Generate random goal position using safe zones."""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            safe_zone = np.random.choice(self.safe_zones)
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, safe_zone['radius'])
            x = safe_zone['center'][0] + radius * np.cos(angle)
            y = safe_zone['center'][1] + radius * np.sin(angle)
            
            if self.is_position_safe(x, y) and self.is_in_safe_zone(x, y):
                self.update_goal_marker(x, y)
                return (x, y)
            
            attempts += 1
        
        # Fallback to center safe zone
        fallback_pos = (0.5, 0.5)
        self.update_goal_marker(fallback_pos[0], fallback_pos[1])
        return fallback_pos

    def update_goal_marker(self, x, y):
        """Update the goal visualization marker."""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal_markers"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.25
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.5
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.lifetime = rospy.Duration()
        self.goal_marker_pub.publish(marker)

    def save_checkpoint(self, episode, forced=False):
        """Save training checkpoint with metrics."""
        if episode % self.params['checkpoint_freq'] == 0 or forced:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"model_ep{episode}_{timestamp}.pth")
            
            metrics = {
                'episode': episode,
                'best_reward': self.best_reward,
                'best_success_rate': self.best_success_rate,
                'recent_rewards': list(self.episode_rewards),
                'recent_successes': list(self.success_history),
                'recent_collisions': list(self.collision_history),
                'recent_steps': list(self.step_history),
                'params': self.params,
                'resource_summary': self.resource_monitor.get_summary()
            }
            
            self.agent.save_model(model_path, episode, metrics)
            
            metrics_path = os.path.join(self.log_dir, f"metrics_ep{episode}_{timestamp}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.resource_monitor.save_metrics(episode)
            rospy.loginfo(f"Checkpoint saved: {model_path}")

    def _update_metrics(self, episode_reward, steps, done):
        """Update training metrics."""
        self.episode_rewards.append(episode_reward)
        self.step_history.append(steps)
        
        if self.env.collision:
            self.success_history.append(0)
            self.collision_history.append(1)
            self.episode_result = "collision"
        elif done and episode_reward > 0:
            self.success_history.append(1)
            self.collision_history.append(0)
            self.episode_result = "success"
        else:
            self.success_history.append(0)
            self.collision_history.append(0)
            self.episode_result = "timeout"
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        if len(self.success_history) >= 20:
            recent_success_rate = sum(list(self.success_history)[-20:]) / 20
            if recent_success_rate > self.best_success_rate:
                self.best_success_rate = recent_success_rate

    def _publish_training_info(self, episode, step, loss, reward):
        """Publish training information."""
        msg = Float32MultiArray()
        msg.data = [float(episode), float(step), loss, reward]
        self.result_pub.publish(msg)

    def _log_progress(self, episode):
        """Log training progress with enhanced status reporting."""
        try:
            avg_reward = np.mean(list(self.episode_rewards))
            success_rate = np.mean(list(self.success_history))
            collision_rate = np.mean(list(self.collision_history))
            
            status = "TIMEOUT"
            if self.episode_result == "success":
                status = "SUCCESS"
            elif self.episode_result == "collision":
                status = "COLLISION"
            
            log_msg = (
                f"\nEpisode: {episode} - Status: {status}\n"
                f"Avg Reward: {avg_reward:.2f}\n"
                f"Success Rate: {success_rate:.2f}\n"
                f"Collision Rate: {collision_rate:.2f}\n"
                f"Epsilon: {self.agent.epsilon:.3f}\n"
                f"\nResource Usage:\n{self.resource_monitor.get_summary()}"
            )
            
            rospy.loginfo(log_msg)
            
        except Exception as e:
            rospy.logerr(f"Error logging progress: {str(e)}")

    def load_stage1_weights(self, model_path):
        """Load pretrained weights from stage 1."""
        if self.agent.load_model(model_path):
            rospy.loginfo(f"Successfully loaded stage 1 weights from {model_path}")
            # Adjust epsilon for transfer learning
            self.agent.epsilon = self.params['initial_epsilon']
            return True
        return False

    def train(self):
        """Enhanced training loop with proper collision detection."""
        try:
            episode = 0
            start_time = time.time()
            
            while episode < self.params['max_episodes'] and not self.interrupted:
                # Update resource monitoring
                if episode % self.params['resource_monitor_freq'] == 0:
                    self.resource_monitor.update()
                
                # Generate new random goal for each episode
                goal_pos = self.generate_random_goal()
                self.env.goal_position = Point(goal_pos[0], goal_pos[1], 0.0)
                self.update_goal_marker(goal_pos[0], goal_pos[1])
                
                state = self.env.reset()
                episode_reward = 0
                steps = 0
                self.env.collision = False  # Reset collision flag
                
                # Episode loop
                for step in range(self.params['max_steps']):
                    action = self.agent.get_action(state)
                    next_state, reward, done = self.env.step(action)
                    
                    # Scale rewards for transfer learning
                    reward *= 1.2
                    
                    # Update episode metrics
                    episode_reward += reward
                    steps += 1
                    
                    # Store transition with higher priority for significant events
                    is_significant = self.env.collision or (done and reward > 0)
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    if len(self.agent.memory) > self.params['batch_size']:
                        loss = self.agent.replay()
                        if loss is not None:
                            self._publish_training_info(episode, step, loss, reward)
                    
                    state = next_state
                    
                    if done or self.env.collision:
                        break
                
                # Update metrics with proper collision detection
                self._update_metrics(episode_reward, steps, done)
                
                # Regular updates
                if episode % self.params['target_update_freq'] == 0:
                    self.agent.update_target_network()
                
                if episode % 10 == 0:
                    self._log_progress(episode)
                
                # Save checkpoints
                self.save_checkpoint(episode)
                
                episode += 1
            
            # Final cleanup
            self.save_checkpoint(episode, forced=True)
            
            # Print final resource usage summary
            rospy.loginfo("\nFinal Resource Usage Summary:\n" + self.resource_monitor.get_summary())
            
        except Exception as e:
            rospy.logerr(f"Training error: {str(e)}")
            raise
        finally:
            # Ensure resource metrics are saved
            self.resource_monitor.save_metrics("final")

def main():
    try:
        trainer = Stage2Trainer()
        
        # Load Stage 1 model weights
        stage1_model = "/home/alma/catkin_ws/src/dqn3/models/stage1_enhanced_dqn/model_ep1700_20250105_123219.pth"
        
        if trainer.load_stage1_weights(stage1_model):
            rospy.loginfo("Starting stage 2 training with pretrained weights...")
            trainer.train()
        else:
            rospy.logerr(f"Failed to load stage 1 model from {stage1_model}")
            
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()