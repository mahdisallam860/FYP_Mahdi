#!/usr/bin/env python3

import rospy
import numpy as np
import os
import rospkg
import torch
import time
import json
from geometry_msgs.msg import Point
from dqn3.turtlebot3_env import TurtleBot3Env
from dqn3.dqn_agent import DQNAgent
import glob

class Stage1ModelEvaluator:
    def __init__(self):
        rospy.init_node('stage1_model_evaluator', anonymous=True)
        
        # Setup paths
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('dqn3')
        self.model_dir = os.path.join(self.pkg_path, "models", "stage1")
        self.results_dir = os.path.join(self.pkg_path, "evaluation_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define curriculum goals
        self.curriculum_goals = [
            (0.8, 0.0),    # Level 0: Square pattern
            (1.5, 0.0),    # Level 1: Cardinal direction
            (2.0, 2.0),   # Level 2: Diagonal pattern
            (2.5, 2.5)     # Level 3: Complex path
        ]
        
        # Evaluation parameters
        self.eval_episodes = 10  # Number of episodes per goal position
        
        # Initialize environment and agent
        self.env = None
        self.agent = None
        
        # Dictionary to store model performance
        self.model_performances = {}

    def list_available_models(self):
        """List all available model checkpoints."""
        models = glob.glob(os.path.join(self.model_dir, "model_ep*.pth"))
        models.sort(key=lambda x: int(x.split('_ep')[1].split('_')[0]))
        return models

    def initialize_environment(self, goal_position):
        """Initialize or update environment."""
        if self.env is None:
            self.env = TurtleBot3Env(
                is_training=False,
                goal_position=goal_position,
                timeout=500,
                stage=1
            )
        else:
            self.env.goal_position = Point(*goal_position, 0.0)

    def initialize_agent(self, model_path):
        """Initialize agent with specific model."""
        self.agent = DQNAgent(
            state_size=26,
            action_size=3,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon=0.01,
            epsilon_decay=1.0,
            epsilon_min=0.01,
            batch_size=64,
            memory_size=100000
        )
        
        # Load model
        if self.agent.load_model(model_path, training=False):
            episode_num = int(model_path.split('_ep')[1].split('_')[0])
            rospy.loginfo(f"Successfully loaded model from episode {episode_num}")
            return True
        return False

    def evaluate_goal(self, goal_position, level):
        """Evaluate performance for a specific goal position."""
        self.initialize_environment(goal_position)
        
        metrics = {
            'success_count': 0,
            'collision_count': 0,
            'timeout_count': 0,
            'episode_rewards': [],
            'episode_steps': [],
            'path_lengths': [],
            'completion_times': [],
            'min_distances': []
        }
        
        for episode in range(self.eval_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            start_time = time.time()
            path_length = 0
            previous_position = None
            min_distance_to_goal = float('inf')
            
            while not done and not rospy.is_shutdown():
                action = self.agent.get_action(state, testing=True)
                next_state, reward, done = self.env.step(action)
                
                # Track metrics
                current_position = [self.env.position.x, self.env.position.y]
                distance_to_goal = np.sqrt(
                    (current_position[0] - goal_position[0])**2 +
                    (current_position[1] - goal_position[1])**2
                )
                min_distance_to_goal = min(min_distance_to_goal, distance_to_goal)
                
                if previous_position is not None:
                    path_length += np.sqrt(
                        (current_position[0] - previous_position[0])**2 +
                        (current_position[1] - previous_position[1])**2
                    )
                previous_position = current_position
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Update metrics
            completion_time = time.time() - start_time
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_steps'].append(steps)
            metrics['path_lengths'].append(path_length)
            metrics['completion_times'].append(completion_time)
            metrics['min_distances'].append(min_distance_to_goal)
            
            if total_reward > 50:
                metrics['success_count'] += 1
            elif total_reward < -50:
                metrics['collision_count'] += 1
            else:
                metrics['timeout_count'] += 1
            
            rospy.loginfo(f"Level {level} - Episode {episode + 1}: " +
                         f"Reward: {total_reward:.2f}, Steps: {steps}, " +
                         f"Path Length: {path_length:.2f}m, " +
                         f"Min Distance: {min_distance_to_goal:.2f}m")
        
        return metrics

    def evaluate_model(self, model_path):
        """Evaluate a specific model checkpoint."""
        if not self.initialize_agent(model_path):
            return None
        
        results = {
            'model_path': model_path,
            'episode_number': int(model_path.split('_ep')[1].split('_')[0]),
            'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results_by_level': {}
        }
        
        overall_success = 0
        overall_path_efficiency = 0
        
        for level, goal in enumerate(self.curriculum_goals):
            rospy.loginfo(f"\nEvaluating Level {level} - Goal: ({goal[0]}, {goal[1]})")
            metrics = self.evaluate_goal(goal, level)
            
            # Calculate level summary
            success_rate = (metrics['success_count'] / self.eval_episodes) * 100
            avg_path_length = np.mean(metrics['path_lengths'])
            optimal_path = np.sqrt(goal[0]**2 + goal[1]**2)  # Straight line distance
            path_efficiency = optimal_path / avg_path_length if avg_path_length > 0 else 0
            
            summary = {
                'success_rate': success_rate,
                'collision_rate': (metrics['collision_count'] / self.eval_episodes) * 100,
                'timeout_rate': (metrics['timeout_count'] / self.eval_episodes) * 100,
                'avg_reward': np.mean(metrics['episode_rewards']),
                'avg_steps': np.mean(metrics['episode_steps']),
                'avg_path_length': avg_path_length,
                'optimal_path_length': optimal_path,
                'path_efficiency': path_efficiency,
                'avg_completion_time': np.mean(metrics['completion_times']),
                'min_distance_to_goal': min(metrics['min_distances'])
            }
            
            results['results_by_level'][f'level_{level}'] = {
                'goal': goal,
                'metrics': metrics,
                'summary': summary
            }
            
            overall_success += success_rate
            overall_path_efficiency += path_efficiency
            
            rospy.loginfo(f"\nLevel {level} Summary:")
            rospy.loginfo(f"Success Rate: {success_rate:.1f}%")
            rospy.loginfo(f"Average Reward: {summary['avg_reward']:.2f}")
            rospy.loginfo(f"Path Efficiency: {path_efficiency:.2f}")
            rospy.loginfo(f"Min Distance to Goal: {summary['min_distance_to_goal']:.2f}m")
        
        # Calculate overall performance
        results['overall_performance'] = {
            'average_success_rate': overall_success / len(self.curriculum_goals),
            'average_path_efficiency': overall_path_efficiency / len(self.curriculum_goals)
        }
        
        return results

    def run_evaluation(self, specific_model=None):
        """Run evaluation on specific model or all models."""
        if specific_model:
            models_to_evaluate = [specific_model]
        else:
            models_to_evaluate = self.list_available_models()
            rospy.loginfo(f"Found {len(models_to_evaluate)} models to evaluate")
        
        for model_path in models_to_evaluate:
            rospy.loginfo(f"\nEvaluating model: {os.path.basename(model_path)}")
            results = self.evaluate_model(model_path)
            
            if results:
                # Save individual model results
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_name = os.path.basename(model_path).replace('.pth', '')
                results_path = os.path.join(
                    self.results_dir, 
                    f'evaluation_{model_name}_{timestamp}.json'
                )
                
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
                
                rospy.loginfo(f"\nOverall Performance for {model_name}:")
                rospy.loginfo(f"Average Success Rate: {results['overall_performance']['average_success_rate']:.1f}%")
                rospy.loginfo(f"Average Path Efficiency: {results['overall_performance']['average_path_efficiency']:.2f}")
                rospy.loginfo(f"Results saved to: {results_path}")

def main():
    try:
        evaluator = Stage1ModelEvaluator()
        
        # List available models
        models = evaluator.list_available_models()
        rospy.loginfo("\nAvailable models:")
        for i, model in enumerate(models):
            rospy.loginfo(f"{i+1}. {os.path.basename(model)}")
        
        # Let user choose model
        while True:
            try:
                choice = input("\nEnter model number to evaluate (or 'all' for all models): ")
                if choice.lower() == 'all':
                    evaluator.run_evaluation()
                    break
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        evaluator.run_evaluation(models[idx])
                        break
                    else:
                        rospy.logwarn("Invalid model number")
            except ValueError:
                rospy.logwarn("Invalid input")
            except KeyboardInterrupt:
                rospy.loginfo("\nEvaluation cancelled")
                break
        
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()