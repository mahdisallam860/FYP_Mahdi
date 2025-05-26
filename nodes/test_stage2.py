#!/usr/bin/env python3

import rospy
import numpy as np
import os
import rospkg
import json
from collections import deque
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from dqn3.turtlebot3_env import TurtleBot3Env
from dqn3.dqn_agent import DQNAgent
from std_msgs.msg import Float32MultiArray

class Stage2Tester:
    def __init__(self):
        # Initialize ROS node first
        rospy.init_node('turtlebot3_stage2_tester', anonymous=True)
        
        # Initialize publishers right after node initialization
        self.goal_marker_pub = rospy.Publisher('/goal_visualization', Marker, queue_size=10)
        self.result_pub = rospy.Publisher('test_results', Float32MultiArray, queue_size=5)
        
        # Allow time for publishers to initialize
        rospy.sleep(1)
        
        # Initialize paths
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('dqn3')
        self.model_path = "/home/alma/catkin_ws/src/dqn3/models/stage2/model_ep1500_20241112_011059.pth"
        
        # Test parameters
        self.params = {
            'num_test_episodes': 100,
            'max_steps': 500,
            'success_threshold': 0.2
        }
        
        # Define obstacles and bounds
        self.obstacles = [
            {'pos': (-1.0, 1.0), 'radius': 0.4},
            {'pos': (1.0, 1.0), 'radius': 0.4},
            {'pos': (-1.0, -1.0), 'radius': 0.4},
            {'pos': (1.0, -1.0), 'radius': 0.4},
            {'pos': (0.0, 2.0), 'radius': 0.4},
            {'pos': (2.0, 0.0), 'radius': 0.4},
            {'pos': (-2.0, -2.0), 'radius': 0.4},
            {'pos': (2.0, 2.0), 'radius': 0.4},
        ]
        
        self.bounds = {
            'x_min': -3.2,
            'x_max': 3.2,
            'y_min': -3.2,
            'y_max': 3.2,
            'safety_margin': 0.3
        }
        
        # Initialize metrics
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.episode_steps = []
        self.episode_rewards = []
        self.final_distances = []
        
        # Define some safe fallback positions for goal generation
        self.safe_positions = [
            (1.0, 1.0),
            (-1.0, -1.0),
            (2.0, 2.0),
            (-2.0, 2.0)
        ]
        
        # Initialize environment with an initial goal
        initial_goal = self.generate_random_goal()
        self.env = TurtleBot3Env(
            is_training=False,
            goal_position=initial_goal,
            timeout=self.params['max_steps'],
            stage=2
        )
        
        # Initialize agent
        self.agent = DQNAgent(
            state_size=26,
            action_size=3,
            learning_rate=0.00005,
            gamma=0.99,
            epsilon=0.0,  # No exploration during testing
            epsilon_min=0.0,
            batch_size=64,
            memory_size=100000
        )

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
                self.update_goal_marker(x, y)
                return (x, y)
            
            attempts += 1
        
        # Fallback to a safe position
        return (1.0, 1.0)  # You might want to define more fallback positions

    def update_goal_marker(self, x, y):
        """Publish a marker to visualize the current goal."""
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

    def load_model(self):
        """Load the trained model."""
        if self.agent.load_model(self.model_path):
            rospy.loginfo(f"Successfully loaded model from {self.model_path}")
            return True
        return False

    def run_tests(self):
        """Run test episodes and collect metrics."""
        if not self.load_model():
            rospy.logerr("Failed to load model. Aborting tests.")
            return

        rospy.loginfo("Starting test episodes...")
        
        for episode in range(self.params['num_test_episodes']):
            # Generate new random goal
            goal_pos = self.generate_random_goal()
            self.env.goal_position = Point(goal_pos[0], goal_pos[1], 0.0)
            
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            # Episode loop
            for step in range(self.params['max_steps']):
                action = self.agent.get_action(state)
                next_state, reward, done = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Calculate final distance to goal
            final_dist = np.sqrt(
                (self.env.position.x - self.env.goal_position.x)**2 +
                (self.env.position.y - self.env.goal_position.y)**2
            )
            
            # Update metrics
            self.episode_steps.append(steps)
            self.episode_rewards.append(episode_reward)
            self.final_distances.append(final_dist)
            
            if final_dist <= self.params['success_threshold']:
                self.success_count += 1
            elif episode_reward < -50:  # Collision
                self.collision_count += 1
            elif steps >= self.params['max_steps']:
                self.timeout_count += 1
            
            if episode % 10 == 0:
                self._log_progress(episode)
        
        self._log_final_results()

    def _log_progress(self, episode):
        """Log intermediate test results."""
        success_rate = self.success_count / (episode + 1)
        collision_rate = self.collision_count / (episode + 1)
        timeout_rate = self.timeout_count / (episode + 1)
        avg_steps = np.mean(self.episode_steps)
        avg_reward = np.mean(self.episode_rewards)
        avg_dist = np.mean(self.final_distances)
        
        rospy.loginfo(
            f"Episode {episode}/{self.params['num_test_episodes']}\n"
            f"Success Rate: {success_rate:.2f}, Collision Rate: {collision_rate:.2f}, "
            f"Timeout Rate: {timeout_rate:.2f}\n"
            f"Avg Steps: {avg_steps:.1f}, Avg Reward: {avg_reward:.1f}, "
            f"Avg Final Distance: {avg_dist:.2f}m"
        )

    def _log_final_results(self):
        """Log final test results and save to file."""
        success_rate = self.success_count / self.params['num_test_episodes']
        collision_rate = self.collision_count / self.params['num_test_episodes']
        timeout_rate = self.timeout_count / self.params['num_test_episodes']
        
        results = {
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_steps': np.mean(self.episode_steps),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_final_distance': np.mean(self.final_distances),
            'std_final_distance': np.std(self.final_distances),
            'min_final_distance': np.min(self.final_distances),
            'max_final_distance': np.max(self.final_distances)
        }
        
        # Save results
        results_path = os.path.join(
            self.pkg_path,
            'test_results',
            f'stage2_results_{rospy.Time.now().to_sec():.0f}.json'
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        rospy.loginfo("\nFinal Test Results:")
        rospy.loginfo(f"Number of episodes: {self.params['num_test_episodes']}")
        rospy.loginfo(f"Success rate: {success_rate:.2%}")
        rospy.loginfo(f"Collision rate: {collision_rate:.2%}")
        rospy.loginfo(f"Timeout rate: {timeout_rate:.2%}")
        rospy.loginfo(f"Average steps per episode: {results['avg_steps']:.1f}")
        rospy.loginfo(f"Average reward per episode: {results['avg_reward']:.1f}")
        rospy.loginfo(f"Average final distance to goal: {results['avg_final_distance']:.2f}m")
        rospy.loginfo(f"Results saved to: {results_path}")

def main():
    try:
        tester = Stage2Tester()
        tester.run_tests()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()





