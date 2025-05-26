#!/usr/bin/env python3

from collections import deque
import rospy
import numpy as np
import os
import rospkg
import json
import torch
import signal
import time
import random
import matplotlib.pyplot as plt
from datetime import datetime
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from dqn3.turtlebot3_env import TurtleBot3Env
from dqn3.ddqn_agent import DQNAgent
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion

class Stage2Trainer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot3_stage2_trainer', anonymous=True)
        
        # Setup paths (Change stage1 to stage2)
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('dqn3')
        self.model_dir = os.path.join(self.pkg_path, "models", "stage2")  # Changed from stage1
        self.log_dir = os.path.join(self.pkg_path, "logs", "stage2")      # Changed from stage1
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Training parameters (Modified for stage2)
        self.params = {
            'max_episodes': 2500,
            'max_steps': 500,
            'batch_size': 64,
            'learning_rate': 0.00005,  # Reduced for fine-tuning
            'gamma': 0.99,
            'initial_epsilon': 0.2,    # Lower initial epsilon
            'epsilon_decay': 0.998,
            'min_epsilon': 0.02,
            'memory_size': 100000,
            'target_update_freq': 5
        }
        
        # Define obstacles and safety parameters (New for stage2)
        self.obstacles = [
            {'pos': (-1.0, 1.0), 'radius': 0.4},   # obstacle1
            {'pos': (1.0, 1.0), 'radius': 0.4},    # obstacle2
            {'pos': (-1.0, -1.0), 'radius': 0.4},  # obstacle3
            {'pos': (1.0, -1.0), 'radius': 0.4},   # obstacle4
            {'pos': (0.0, 2.0), 'radius': 0.4},    # obstacle5
            {'pos': (2.0, 0.0), 'radius': 0.4},    # obstacle6
            {'pos': (-2.0, -2.0), 'radius': 0.4},  # obstacle7
            {'pos': (2.0, 2.0), 'radius': 0.4},    # obstacle8
        ]

        self.bounds = {
            'x_min': -3.2,
            'x_max': 3.2,
            'y_min': -3.2,
            'y_max': 3.2,
            'safety_margin': 0.3
        }
        
        # Initialize performance tracking
        self.best_reward = float('-inf')
        self.best_success_rate = 0.0
        self.episode_rewards = deque(maxlen=100)
        self.success_history = deque(maxlen=100)
        self.collision_history = deque(maxlen=100)
        self.step_history = deque(maxlen=100)
        
        # Initialize ROS publishers
        self.result_pub = rospy.Publisher('training_results', Float32MultiArray, queue_size=5)
        
        # Setup interruption handling
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        # Initialize environment with random goal
        initial_goal = self.generate_random_goal()
        self.env = TurtleBot3Env(
            is_training=True,
            goal_position=initial_goal,
            timeout=self.params['max_steps'],
            stage=2  # Changed from stage1
        )
        
        # Initialize agent
        self.agent = DQNAgent(
            state_size=26,
            action_size=3,
            learning_rate=self.params['learning_rate'],
            gamma=self.params['gamma'],
            epsilon=self.params['initial_epsilon'],
            epsilon_decay=self.params['epsilon_decay'],
            epsilon_min=self.params['min_epsilon'],
            batch_size=self.params['batch_size'],
            memory_size=self.params['memory_size']
        )

        self.goal_marker_pub = rospy.Publisher('/goal_visualization', Marker, queue_size=10)
        rospy.sleep(1)

    
    def handle_interrupt(self, signum, frame):
        """Handle interruption gracefully."""
        rospy.loginfo("Received interrupt signal. Cleaning up...")
        self.interrupted = True



    def is_position_safe(self, x, y):
        """Check if a position is safe (away from obstacles and walls)."""
        # Check boundary safety margins
        if (x < self.bounds['x_min'] + self.bounds['safety_margin'] or 
            x > self.bounds['x_max'] - self.bounds['safety_margin'] or
            y < self.bounds['y_min'] + self.bounds['safety_margin'] or
            y > self.bounds['y_max'] - self.bounds['safety_margin']):
            return False

        # Check distance from all obstacles
        for obstacle in self.obstacles:
            dist = np.sqrt((x - obstacle['pos'][0])**2 + (y - obstacle['pos'][1])**2)
            if dist < obstacle['radius'] + self.bounds['safety_margin']:
                return False

        return True
    
    def generate_random_goal(self):
        """Generate random goal position using rejection sampling."""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            x = np.random.uniform(
                self.bounds['x_min'] + self.bounds['safety_margin'],
                self.bounds['x_max'] - self.bounds['safety_margin']
            )
            y = np.random.uniform(
                self.bounds['y_min'] + self.bounds['safety_margin'],
                self.bounds['y_max'] - self.bounds['safety_margin']
            )

            if self.is_position_safe(x, y):
                # Add this line
                self.update_goal_marker(x, y)
                #rospy.loginfo(f"Generated safe goal position at ({x:.2f}, {y:.2f})")
                return (x, y)

            attempts += 1

        # Fallback to predefined safe positions
        pos = random.choice(self.safe_positions)
        # Add this line
        self.update_goal_marker(pos[0], pos[1])
        rospy.logwarn(f"Using predefined safe position at {pos}")
        return pos

    def load_stage1_weights(self, model_path):
        """Load pretrained weights from stage 1."""
        if self.agent.load_model(model_path):
            rospy.loginfo(f"Successfully loaded stage 1 weights from {model_path}")
            return True
        return False
    
    def update_goal_marker(self, x, y):
        """Publish a marker to visualize the current goal."""
        try:
            marker = Marker()
            marker.header.frame_id = "odom"  # Try "map" frame instead of "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "goal_markers"
            marker.id = 0
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Set the pose
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.25  # Height from ground
            marker.pose.orientation.w = 1.0
            
            # Set the scale
            marker.scale.x = 0.3  # Diameter
            marker.scale.y = 0.3
            marker.scale.z = 0.5  # Height
            
            # Set the color (bright green with transparency)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime = rospy.Duration()  # Marker persists indefinitely
            
            # Publish the marker
            self.goal_marker_pub.publish(marker)
            #rospy.loginfo(f"Published goal marker at ({x}, {y})")
            
        except Exception as e:
            rospy.logerr(f"Error publishing goal marker: {str(e)}")
    



    def save_checkpoint(self, episode, forced=False):
        """Save training checkpoint with metrics."""
        if episode % 100 == 0 or forced:
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
                'params': self.params
            }
            
            self.agent.save_model(model_path, episode, metrics)
            
            # Save readable metrics
            metrics_path = os.path.join(self.log_dir, f"metrics_ep{episode}_{timestamp}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            rospy.loginfo(f"Checkpoint saved: {model_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        if self.agent.load_model(checkpoint_path):
            metrics = torch.load(checkpoint_path, map_location='cpu')
            if 'metrics' in metrics:
                self.curriculum_level = metrics['metrics']['curriculum_level']
                self.env.goal_position = Point(*self.curriculum_goals[self.curriculum_level], 0.0)
                rospy.loginfo(f"Resumed training from curriculum level {self.curriculum_level}")
            return True
        return False


    def train(self):
        """Modified training loop for stage 2."""
        start_time = time.time()
        episode = 0
        
        
        while episode < self.params['max_episodes'] and not self.interrupted:
            try:
                # Generate new random goal for each episode
                goal_pos = self.generate_random_goal()
                self.env.goal_position = Point(goal_pos[0], goal_pos[1], 0.0)
                
                state = self.env.reset()
                episode_reward = 0
                steps = 0
                goal_pos = self.generate_random_goal()
                self.env.goal_position = Point(goal_pos[0], goal_pos[1], 0.0)
            
            # Add this line to update the marker
                self.update_goal_marker(goal_pos[0], goal_pos[1])
                
                # Episode loop
                for step in range(self.params['max_steps']):
                    action = self.agent.get_action(state)
                    next_state, reward, done = self.env.step(action)
                    
                    self.agent.remember(state, action, reward, next_state, done,
                                        priority=(abs(reward) > 50))
                    
                    if len(self.agent.memory) > self.params['batch_size']:
                        loss = self.agent.replay()
                        if loss is not None:
                            self._publish_training_info(episode, step, loss, reward)
                    
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                # Update metrics
                self._update_metrics(episode_reward, steps, done)
                
                # Regular updates
                if episode % self.params['target_update_freq'] == 0:
                    self.agent.update_target_network()
                    rospy.loginfo("Updated target network")
                
                if episode % 10 == 0:
                    self._log_progress(episode, start_time)
                
                if episode % 100 == 0:
                    self.save_checkpoint(episode)
                
                episode += 1
                
            except Exception as e:
                rospy.logerr(f"Error in episode {episode}: {str(e)}")
                continue
        
        self.save_checkpoint(episode, forced=True)
        self.env.stop()
        rospy.loginfo("Stage 2 training completed!")

    def _update_metrics(self, episode_reward, steps, done):
        """Update training metrics with better error handling."""
        try:
            # Update episode tracking
            self.episode_rewards.append(episode_reward)
            self.step_history.append(steps)
            
            # Update success/collision tracking
            success = episode_reward > 50
            collision = episode_reward < -50
            self.success_history.append(1 if success else 0)
            self.collision_history.append(1 if collision else 0)
            
            # Update best metrics
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
            
            if len(self.success_history) >= 20:
                # Fix: Convert deque to list for proper slicing
                recent_history = list(self.success_history)
                success_rate = sum(recent_history) / len(recent_history)
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    
            return True
        except Exception as e:
            rospy.logerr(f"Error updating metrics: {str(e)}")
            return False


    def _publish_training_info(self, episode, step, loss, reward):
        """Publish training information for visualization."""
        msg = Float32MultiArray()
        msg.data = [float(episode), float(step), loss, reward]
        self.result_pub.publish(msg)


    def _compute_distance(self, pos1, pos2):
        """Compute Euclidean distance between two points."""
        try:
            return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
        except Exception as e:
            rospy.logwarn(f"Error computing distance: {str(e)}")
            return None

    def _log_progress(self, episode, start_time):
        """Log training progress."""
        try:
            elapsed_time = time.time() - start_time
            
            rewards_list = list(self.episode_rewards)
            success_list = list(self.success_history)
            collision_list = list(self.collision_history)
            
            avg_reward = np.mean(rewards_list) if rewards_list else 0
            success_rate = np.mean(success_list) if success_list else 0
            collision_rate = np.mean(collision_list) if collision_list else 0
            
            current_distance = self._compute_distance(
                self.env.position,
                self.env.goal_position
            ) if hasattr(self.env, '_compute_distance') else None
            
            log_msg = (
                f"Episode: {episode}, "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Success Rate: {success_rate:.2f}, "
                f"Collision Rate: {collision_rate:.2f}, "
                f"Epsilon: {self.agent.epsilon:.3f}, "
                f"Elapsed Time: {elapsed_time:.0f}s"
            )
            
            if current_distance is not None:
                log_msg += f", Distance to Goal: {current_distance:.2f}m"
                
            rospy.loginfo(log_msg)
            
        except Exception as e:
            rospy.logerr(f"Error logging progress: {str(e)}")

def main():
    try:
        trainer = Stage2Trainer()
        
        # Load Stage 1 model weights
        stage1_model = "/home/alma/catkin_ws/src/dqn3/models/stage1/model_ep1500_20241125_042943.pth"
        
        if trainer.load_stage1_weights(stage1_model):
            rospy.loginfo("Starting stage 2 training with pretrained weights...")
            trainer.train()
        else:
            rospy.logerr(f"Failed to load stage 1 model from {stage1_model}")
            
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()