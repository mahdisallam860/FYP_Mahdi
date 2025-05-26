#!/usr/bin/env python3

import math
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
from dqn3.turtlebot3_env3_copy import TurtleBot3Env3
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

class Stage3Trainer:
    def __init__(self):
        rospy.init_node('turtlebot3_stage3_trainer', anonymous=True)

        # Setup paths
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('dqn3')
        self.model_dir = os.path.join(self.pkg_path, "models", "stage311")
        self.log_dir = os.path.join(self.pkg_path, "logs", "stage311")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize publishers first
        self.result_pub = rospy.Publisher('training_results', Float32MultiArray, queue_size=5)
        self.goal_marker_pub = rospy.Publisher('/goal_visualization', Marker, queue_size=10)

        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(self.log_dir)

        # Training parameters
        self.params = {
            'max_episodes': 2000,
            'max_steps': 500,
            'batch_size': 64,
            'learning_rate': 0.00002,
            'gamma': 0.99,
            'initial_epsilon': 0.2,
            'epsilon_decay': 0.997,
            'min_epsilon': 0.05,
            'memory_size': 200000,
            'target_update_freq': 10,
            'checkpoint_freq': 50,
            'resource_monitor_freq': 10
        }

        # Initialize dynamic obstacle tracking
        self.dynamic_obstacle_positions = {}
        for i in range(1, 4):  # 3 dynamic obstacles
            topic = f'/dynamic_obstacle{i}/odom'
            rospy.Subscriber(topic, Odometry, self.dynamic_obstacle_callback, callback_args=i)

        # Safe zones setup
        self.safe_zones = [
            {'center': (0.0, 0.0), 'radius': 1.0},
            {'center': (-2.0, 2.0), 'radius': 0.6},
            {'center': (2.0, 2.0), 'radius': 0.6},
            {'center': (-2.0, -2.0), 'radius': 0.6},
            {'center': (2.0, -2.0), 'radius': 0.6},
            {'center': (0.0, 2.0), 'radius': 0.7},
            {'center': (0.0, -2.0), 'radius': 0.7},
            {'center': (2.0, 0.0), 'radius': 0.7},
            {'center': (-2.0, 0.0), 'radius': 0.7}
        ]

        # Static wall positions
        self.static_walls = [
            {'start': (-1.8, 1.0), 'end': (0.2, 1.0)},
            {'start': (1.8, -1.0), 'end': (-0.2, -1.0)},
            {'start': (-1.0, -2.0), 'end': (-1.0, -0.5)},
            {'start': (1.0, 2.0), 'end': (1.0, 0.5)}
        ]

        # Dynamic obstacle parameters
        self.dynamic_obstacle_radius = 0.2
        self.dynamic_safety_margin = 0.3

        # Performance tracking
        self.best_reward = float('-inf')
        self.best_success_rate = 0.0
        self.episode_rewards = deque(maxlen=100)
        self.success_history = deque(maxlen=100)
        self.collision_history = deque(maxlen=100)
        self.step_history = deque(maxlen=100)
        self.episode_result = None

        # Generate initial goal and initialize environment
        initial_goal = self.generate_random_goal()
        self.env = TurtleBot3Env3(
            is_training=True,
            goal_position=initial_goal,
            timeout=self.params['max_steps']
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

        # Wait a bit for publishers and subscribers to initialize
        rospy.sleep(1.0)

    def dynamic_obstacle_callback(self, msg, obstacle_id):
        """Track dynamic obstacle positions"""
        pos = msg.pose.pose.position
        self.dynamic_obstacle_positions[f'obstacle{obstacle_id}'] = (pos.x, pos.y)

    def is_position_safe(self, x, y):
        """Enhanced safety check for goal placement"""
        # Check world boundaries
        margin = 0.3
        if not (-3.0 + margin < x < 3.0 - margin and -3.0 + margin < y < 3.0 - margin):
            return False

        # Check static walls
        for wall in self.static_walls:
            if self._point_to_line_segment_distance(x, y, 
                                                  wall['start'][0], wall['start'][1],
                                                  wall['end'][0], wall['end'][1]) < 0.4:
                return False

        # Check dynamic obstacles with larger margin
        for obs_pos in self.dynamic_obstacle_positions.values():
            if math.sqrt((x - obs_pos[0])**2 + (y - obs_pos[1])**2) < (self.dynamic_obstacle_radius + self.dynamic_safety_margin):
                return False

        return True

    def _point_to_line_segment_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            return math.sqrt(A * A + B * B)

        param = dot / len_sq

        if param < 0:
            return math.sqrt(A * A + B * B)
        elif param > 1:
            return math.sqrt((px - x2) * (px - x2) + (py - y2) * (py - y2))
        else:
            return abs(A * D - C * B) / math.sqrt(len_sq)

    def generate_random_goal(self):
        """Generate random goal position considering dynamic environment"""
        max_attempts = 150
        
        for _ in range(max_attempts):
            # Select random safe zone with weighted probability
            weights = [1.5 if zone['radius'] > 0.8 else 1.0 for zone in self.safe_zones]
            zone = np.random.choice(self.safe_zones, p=np.array(weights)/sum(weights))
            
            # Generate random point within zone
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.triangular(0, zone['radius']*0.5, zone['radius'])
            
            x = zone['center'][0] + radius * np.cos(angle)
            y = zone['center'][1] + radius * np.sin(angle)
            
            if self.is_position_safe(x, y):
                self.update_goal_marker(x, y)
                return (x, y)
        
        # Fallback to center if no safe position found
        rospy.logwarn("Failed to find safe goal position, using fallback position")
        return (0.0, 0.0)

    def update_goal_marker(self, x, y):
        """Update goal visualization in RViz"""
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
        
        self.goal_marker_pub.publish(marker)

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal"""
        rospy.loginfo("Training interrupted. Cleaning up...")
        self.interrupted = True
        self.resource_monitor.save_metrics("interrupted")
        rospy.loginfo("\nResource Usage Summary:\n" + self.resource_monitor.get_summary())

    def load_stage1_weights(self, weights_path):
        """Load pretrained Stage 1 weights"""
        if self.agent.load_model(weights_path):
            rospy.loginfo(f"Successfully loaded Stage 1 weights from {weights_path}")
            self.agent.epsilon = self.params['initial_epsilon']
            return True
        return False

    def save_checkpoint(self, episode, forced=False):
        """Save training checkpoint with metrics"""
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
            """Update training metrics and detect success/collision/timeout"""
            self.episode_rewards.append(episode_reward)
            self.step_history.append(steps)
            
            if self.env.collision:
                self.success_history.append(0)
                self.collision_history.append(1)
                self.episode_result = "collision"
            elif done and episode_reward > 0:  # Successful completion
                self.success_history.append(1)
                self.collision_history.append(0)
                self.episode_result = "success"
            else:  # Timeout
                self.success_history.append(0)
                self.collision_history.append(0)
                self.episode_result = "timeout"
            
            # Update best metrics
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
            
            if len(self.success_history) >= 20:
                recent_success_rate = sum(list(self.success_history)[-20:]) / 20
                if recent_success_rate > self.best_success_rate:
                    self.best_success_rate = recent_success_rate

    def _publish_training_info(self, episode, step, loss, reward):
            """Publish training information for visualization"""
            msg = Float32MultiArray()
            msg.data = [float(episode), float(step), float(loss), float(reward)]
            self.result_pub.publish(msg)

    def _log_progress(self, episode):
            """Log training progress with enhanced status reporting"""
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
                    f"Avg Reward (last 100): {avg_reward:.2f}\n"
                    f"Success Rate (last 100): {success_rate:.2f}\n"
                    f"Collision Rate (last 100): {collision_rate:.2f}\n"
                    f"Epsilon: {self.agent.epsilon:.3f}\n"
                    f"\nResource Usage:\n{self.resource_monitor.get_summary()}"
                )
                
                rospy.loginfo(log_msg)
                
            except Exception as e:
                rospy.logerr(f"Error logging progress: {str(e)}")

    def train(self):
            """Main training loop with enhanced dynamic obstacle handling"""
            try:
                episode = 0
                start_time = time.time()
                
                while episode < self.params['max_episodes'] and not self.interrupted:
                    # Update resource monitoring
                    if episode % self.params['resource_monitor_freq'] == 0:
                        self.resource_monitor.update()
                    
                    # Generate new goal position
                    goal_pos = self.generate_random_goal()
                    self.env.goal_position = Point(goal_pos[0], goal_pos[1], 0.0)
                    
                    # Reset environment
                    state = self.env.reset()
                    episode_reward = 0
                    steps = 0
                    self.env.collision = False
                    
                    # Episode loop
                    for step in range(self.params['max_steps']):
                        # Get action
                        action = self.agent.get_action(state)
                        
                        # Execute action
                        next_state, reward, done = self.env.step(action)
                        
                        # Update episode metrics
                        episode_reward += reward
                        steps += 1
                        
                        # Store experience
                        priority = 2.0 if (done and reward > 0) or self.env.collision else 1.0
                        self.agent.remember(state, action, reward, next_state, done)
                        
                        # Train agent
                        if len(self.agent.memory) > self.params['batch_size']:
                            loss = self.agent.replay()
                            if loss is not None:
                                self._publish_training_info(episode, step, loss, reward)
                        
                        state = next_state
                        
                        if done or self.env.collision:
                            break
                    
                    # Update metrics
                    self._update_metrics(episode_reward, steps, done)
                    
                    # Update target network
                    if episode % self.params['target_update_freq'] == 0:
                        self.agent.update_target_network()
                    
                    # Log progress
                    if episode % 10 == 0:
                        self._log_progress(episode)
                    
                    # Save checkpoint
                    self.save_checkpoint(episode)
                    
                    episode += 1
                
                # Final cleanup
                self.save_checkpoint(episode, forced=True)
                rospy.loginfo("\nFinal Resource Usage Summary:\n" + self.resource_monitor.get_summary())
                
            except Exception as e:
                rospy.logerr(f"Training error: {str(e)}")
                raise
            finally:
                # Ensure resource metrics are saved
                self.resource_monitor.save_metrics("final")


def main():
    try:
        trainer = Stage3Trainer()
        
        # Load Stage 1 weights if available
        stage1_weights = "/home/alma/catkin_ws/src/dqn3/models/stage2_enhanced/model_ep1100_20250105_161824.pth"  # Update path
        if os.path.exists(stage1_weights):
            if trainer.load_stage1_weights(stage1_weights):
                rospy.loginfo("Starting Stage 3 training with pretrained weights...")
            else:
                rospy.logwarn("Could not load Stage 1 weights. Starting fresh training.")
        else:
            rospy.logwarn("Stage 1 weights not found. Starting fresh training.")
        
        # Begin training
        trainer.train()
        
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()