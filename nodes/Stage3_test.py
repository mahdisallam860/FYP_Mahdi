#!/usr/bin/env python3

import rospy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point
from dqn3.turtlebot3_env3_copy import TurtleBot3Env3
from dqn3.ddqn_agent import DDQNAgent
from dqn3.dqn_agent import DQNAgent

class Stage3Tester:
    def __init__(self):
        rospy.init_node('stage3_tester', anonymous=True)

        # Directories for results
        self.results_dir = "/home/alma/catkin_ws/src/dqn3/results/stage3"
        os.makedirs(self.results_dir, exist_ok=True)

        # Define test goals
        self.test_goals = [      
            (-3.0, 3.0, 0.0),  # Top-left
            (3.0, 3.0, 0.0),   # Top-right
            (-3.0, -3.0, 0.0), # Bottom-left
            (3.0, -3.0, 0.0) 
            # Top-left, near wall
               # Bottom-center, near dynamic obstacle 2
        ]


        # Evaluation parameters
        self.eval_episodes = 1 # Increased for better statistics
        self.collision_threshold = 0.3  # Distance threshold for dynamic collision detection

        # Initialize the environment
        self.env = TurtleBot3Env3(
            is_training=False,
            goal_position=(0.0, 0.0),
            timeout=500
        )

        # Initialize agents
        self.ddqn_agent = DDQNAgent(
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

        self.dqn_agent = DQNAgent(
            state_size=28,
            action_size=5,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon=0.01,
            epsilon_decay=1.0,
            epsilon_min=0.01,
            batch_size=64,
            memory_size=100000
        )

    def evaluate_model(self, agent, model_path, agent_type):
        """Evaluate the agent for all goals."""
        if not agent.load_model(model_path):
            rospy.logerr(f"Failed to load {agent_type} model: {model_path}")
            return None

        results = {
            "agent_type": agent_type,
            "results_by_goal": {},
            "overall_metrics": {
                "total_success": 0,
                "total_collisions": 0,
                "total_dynamic_collisions": 0,
                "total_timeouts": 0,
                "average_steps": 0,
                "average_reward": 0
            }
        }

        total_episodes = 0
        for goal in self.test_goals:
            rospy.loginfo(f"\nEvaluating {agent_type} for goal {goal}")
            self.env.goal_position = Point(goal[0], goal[1], 0.0)
            metrics = self.evaluate_goal(agent)
            results["results_by_goal"][str(goal)] = metrics
            
            # Update overall metrics
            results["overall_metrics"]["total_success"] += metrics["success"]
            results["overall_metrics"]["total_collisions"] += metrics["collisions"]
            results["overall_metrics"]["total_dynamic_collisions"] += metrics["dynamic_collisions"]
            results["overall_metrics"]["total_timeouts"] += metrics["timeouts"]
            results["overall_metrics"]["average_steps"] += sum(metrics["steps"])
            results["overall_metrics"]["average_reward"] += sum(metrics["rewards"])
            total_episodes += self.eval_episodes

        # Finalize averages
        results["overall_metrics"]["average_steps"] /= total_episodes
        results["overall_metrics"]["average_reward"] /= total_episodes
        results["overall_metrics"]["success_rate"] = results["overall_metrics"]["total_success"] / total_episodes * 100
        results["overall_metrics"]["collision_rate"] = (results["overall_metrics"]["total_collisions"] + 
                                                      results["overall_metrics"]["total_dynamic_collisions"]) / total_episodes * 100

        return results

    def evaluate_goal(self, agent):
        """Evaluate a single goal with enhanced metrics."""
        metrics = {
            "success": 0,
            "collisions": 0,
            "dynamic_collisions": 0,
            "timeouts": 0,
            "rewards": [],
            "steps": [],
            "min_obstacle_distances": [],
            "path_efficiency": [],
            "average_speed": []
        }

        for episode in range(self.eval_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            min_obstacle_dist = float('inf')
            start_distance = self.env._compute_distance(self.env.position, self.env.goal_position)
            total_distance = 0
            previous_pos = self.env.position

            while not done and not rospy.is_shutdown():
                action = agent.get_action(state, testing=True)
                next_state, reward, done = self.env.step(action)

                # Track minimum distance to obstacles
                laser_readings = state[:24]
                min_obstacle_dist = min(min_obstacle_dist, min(laser_readings))

                # Calculate distance traveled
                current_pos = self.env.position
                step_distance = self.env._compute_distance(previous_pos, current_pos)
                total_distance += step_distance
                previous_pos = current_pos

                state = next_state
                total_reward += reward
                steps += 1

            # Calculate path efficiency
            final_distance = self.env._compute_distance(self.env.position, self.env.goal_position)
            if total_distance > 0:
                path_efficiency = (start_distance - final_distance) / total_distance
            else:
                path_efficiency = 0

            metrics["rewards"].append(total_reward)
            metrics["steps"].append(steps)
            metrics["min_obstacle_distances"].append(min_obstacle_dist)
            metrics["path_efficiency"].append(path_efficiency)
            metrics["average_speed"].append(total_distance / steps if steps > 0 else 0)

            # Classify episode outcome
            if total_reward > 50:
                metrics["success"] += 1
            elif total_reward < -50:
                if min_obstacle_dist < self.collision_threshold:
                    metrics["dynamic_collisions"] += 1
                else:
                    metrics["collisions"] += 1
            else:
                metrics["timeouts"] += 1

            rospy.loginfo(f"Episode {episode + 1}/{self.eval_episodes}: " +
                         f"Steps: {steps}, Reward: {total_reward:.2f}, " +
                         f"Min Obstacle Dist: {min_obstacle_dist:.2f}")

        return metrics

    def plot_results(self, ddqn_results, dqn_results):
        """Generate comprehensive performance visualizations."""
        goals = list(ddqn_results["results_by_goal"].keys())
        x = np.arange(len(goals))
        width = 0.35

        # Helper function for plotting
        def create_comparison_plot(ddqn_data, dqn_data, title, ylabel, filename):
            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, ddqn_data, width, label="DDQN")
            plt.bar(x + width/2, dqn_data, width, label="DQN")
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel("Goals")
            plt.xticks(x, goals, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()

        # Plot various metrics
        metrics_to_plot = [
            (lambda g: np.mean(ddqn_results["results_by_goal"][g]["rewards"]),
             lambda g: np.mean(dqn_results["results_by_goal"][g]["rewards"]),
             "Average Rewards per Goal", "Rewards", "rewards_comparison.png"),
            
            (lambda g: np.mean(ddqn_results["results_by_goal"][g]["steps"]),
             lambda g: np.mean(dqn_results["results_by_goal"][g]["steps"]),
             "Average Steps per Goal", "Steps", "steps_comparison.png"),
            
            (lambda g: ddqn_results["results_by_goal"][g]["success"],
             lambda g: dqn_results["results_by_goal"][g]["success"],
             "Success Count per Goal", "Success Count", "success_comparison.png"),
            
            (lambda g: ddqn_results["results_by_goal"][g]["dynamic_collisions"],
             lambda g: dqn_results["results_by_goal"][g]["dynamic_collisions"],
             "Dynamic Collision Count per Goal", "Collision Count", "dynamic_collisions_comparison.png"),
            
            (lambda g: np.mean(ddqn_results["results_by_goal"][g]["path_efficiency"]),
             lambda g: np.mean(dqn_results["results_by_goal"][g]["path_efficiency"]),
             "Average Path Efficiency per Goal", "Efficiency", "efficiency_comparison.png"),
            
            (lambda g: np.mean(ddqn_results["results_by_goal"][g]["average_speed"]),
             lambda g: np.mean(dqn_results["results_by_goal"][g]["average_speed"]),
             "Average Speed per Goal", "Speed (m/s)", "speed_comparison.png")
        ]

        for ddqn_metric, dqn_metric, title, ylabel, filename in metrics_to_plot:
            create_comparison_plot(
                [ddqn_metric(g) for g in goals],
                [dqn_metric(g) for g in goals],
                title, ylabel, filename
            )

    def print_summary(self, ddqn_results, dqn_results):
        """Print comprehensive performance summary."""
        for agent_type, results in [("DDQN", ddqn_results), ("DQN", dqn_results)]:
            metrics = results["overall_metrics"]
            total_episodes = len(self.test_goals) * self.eval_episodes
            
            summary = f"\n{'-'*50}\n"
            summary += f"{agent_type} Performance Summary:\n"
            summary += f"{'-'*50}\n"
            summary += f"Success Rate: {metrics['success_rate']:.2f}%\n"
            summary += f"Collision Rate: {metrics['collision_rate']:.2f}%\n"
            summary += f"Average Steps per Episode: {metrics['average_steps']:.2f}\n"
            summary += f"Average Reward per Episode: {metrics['average_reward']:.2f}\n"
            summary += f"Total Episodes: {total_episodes}\n"
            summary += f"Total Successes: {metrics['total_success']}\n"
            summary += f"Total Dynamic Collisions: {metrics['total_dynamic_collisions']}\n"
            summary += f"Total Static Collisions: {metrics['total_collisions']}\n"
            summary += f"Total Timeouts: {metrics['total_timeouts']}\n"
            
            rospy.loginfo(summary)

    def run_evaluation(self, ddqn_model_path, dqn_model_path):
        """Run comprehensive evaluation of both models."""
        try:
            # Evaluate both models
            rospy.loginfo("Starting DDQN evaluation...")
            ddqn_results = self.evaluate_model(self.ddqn_agent, ddqn_model_path, "DDQN")
            
            rospy.loginfo("Starting DQN evaluation...")
            dqn_results = self.evaluate_model(self.dqn_agent, dqn_model_path, "DQN")

            # Save detailed results
            results_file = os.path.join(self.results_dir, "comparison_results.json")
            with open(results_file, 'w') as f:
                json.dump({"DDQN": ddqn_results, "DQN": dqn_results}, f, indent=4)
            rospy.loginfo(f"Detailed results saved to {results_file}")

            # Generate plots
            self.plot_results(ddqn_results, dqn_results)
            rospy.loginfo(f"Performance plots saved in {self.results_dir}")

            # Print summary
            self.print_summary(ddqn_results, dqn_results)

        except Exception as e:
            rospy.logerr(f"Error during evaluation: {str(e)}")
            raise

def main():
    try:
        tester = Stage3Tester()
        ddqn_model_path = "/home/alma/catkin_ws/src/dqn3/models/stage3ddqn/cc.pth"
        dqn_model_path = "/home/alma/catkin_ws/src/dqn3/models/stage311/model_ep550_20250107_080215.pth"
        tester.run_evaluation(ddqn_model_path, dqn_model_path)
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()


