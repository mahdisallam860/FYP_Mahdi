#!/usr/bin/env python3

import rospy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point
from dqn3.turtlebot3_env import TurtleBot3Env
from dqn3.ddqn_agent import DDQNAgent as DDQNAgent
from dqn3.dqn_agent import DQNAgent as DQNAgent

class Stage1Tester:
    def __init__(self):
        rospy.init_node('stage1_tester', anonymous=True)

        # Directories for results
        self.results_dir = "/home/alma/catkin_ws/src/dqn3results/stage1"
        os.makedirs(self.results_dir, exist_ok=True)

        # Define test goals
        self.test_goals = [
        (1.2, 1.5),    # Very close to obstacle, requires precise navigation
        (-2.5, -2.5),  # Far corner, requires navigating around multiple obstacles
        (3.0, -3.0),   # Opposite corner, maximum distance
        (-0.5, 2.8)    # Near wall, tight space navigation
    ]

        # Evaluation parameters
        self.eval_episodes = 1

        # Initialize the TurtleBot3 environment
        self.env = TurtleBot3Env(
            is_training=False,
            goal_position=(0.0, 0.0),
            timeout=500,
            stage=1
        )

        # Initialize agents
        self.ddqn_agent = DDQNAgent(
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
        if not agent.load_model(model_path, training=False):
            rospy.logerr(f"Failed to load {agent_type} model: {model_path}")
            return None

        results = {"agent_type": agent_type, "results_by_goal": {}}
        for goal in self.test_goals:
            rospy.loginfo(f"Evaluating goal {goal} with {agent_type}")
            self.env.goal_position = Point(goal[0], goal[1], 0.0)
            metrics = self.evaluate_goal(agent)
            results["results_by_goal"][str(goal)] = metrics
        return results

    def evaluate_goal(self, agent):
        """Evaluate a single goal."""
        metrics = {"success": 0, "collisions": 0, "timeouts": 0, "rewards": [], "steps": []}
        for _ in range(self.eval_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done and not rospy.is_shutdown():
                action = agent.get_action(state, testing=True)
                next_state, reward, done = self.env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
            metrics["rewards"].append(total_reward)
            metrics["steps"].append(steps)
            if total_reward > 50:
                metrics["success"] += 1
            elif total_reward < -50:
                metrics["collisions"] += 1
            else:
                metrics["timeouts"] += 1
        return metrics

    def plot_results(self, ddqn_results, dqn_results):
        """Plot detailed comparisons."""
        goals = list(ddqn_results["results_by_goal"].keys())
        ddqn_rewards = [np.mean(ddqn_results["results_by_goal"][goal]["rewards"]) for goal in goals]
        dqn_rewards = [np.mean(dqn_results["results_by_goal"][goal]["rewards"]) for goal in goals]

        ddqn_steps = [np.mean(ddqn_results["results_by_goal"][goal]["steps"]) for goal in goals]
        dqn_steps = [np.mean(dqn_results["results_by_goal"][goal]["steps"]) for goal in goals]

        ddqn_success = [ddqn_results["results_by_goal"][goal]["success"] for goal in goals]
        dqn_success = [dqn_results["results_by_goal"][goal]["success"] for goal in goals]

        ddqn_collisions = [ddqn_results["results_by_goal"][goal]["collisions"] for goal in goals]
        dqn_collisions = [dqn_results["results_by_goal"][goal]["collisions"] for goal in goals]

        x = np.arange(len(goals))
        width = 0.35

        # Plot Rewards
        plt.figure()
        plt.bar(x - width/2, ddqn_rewards, width, label="DDQN")
        plt.bar(x + width/2, dqn_rewards, width, label="DQN")
        plt.title("Average Rewards per Goal")
        plt.ylabel("Rewards")
        plt.xticks(x, goals, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "rewards_comparison.png"))

        # Plot Steps
        plt.figure()
        plt.bar(x - width/2, ddqn_steps, width, label="DDQN")
        plt.bar(x + width/2, dqn_steps, width, label="DQN")
        plt.title("Average Steps per Goal")
        plt.ylabel("Steps")
        plt.xticks(x, goals, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "steps_comparison.png"))

        # Plot Success
        plt.figure()
        plt.bar(x - width/2, ddqn_success, width, label="DDQN")
        plt.bar(x + width/2, dqn_success, width, label="DQN")
        plt.title("Success Count per Goal")
        plt.ylabel("Success Count")
        plt.xticks(x, goals, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "success_comparison.png"))

        # Plot Collisions
        plt.figure()
        plt.bar(x - width/2, ddqn_collisions, width, label="DDQN")
        plt.bar(x + width/2, dqn_collisions, width, label="DQN")
        plt.title("Collision Count per Goal")
        plt.ylabel("Collision Count")
        plt.xticks(x, goals, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "collisions_comparison.png"))

    def run_evaluation(self, ddqn_model_path, dqn_model_path):
        """Run evaluation for both DDQN and DQN models."""
        ddqn_results = self.evaluate_model(self.ddqn_agent, ddqn_model_path, "DDQN")
        dqn_results = self.evaluate_model(self.dqn_agent, dqn_model_path, "DQN")

        # Save results to JSON
        results_file = os.path.join(self.results_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump({"DDQN": ddqn_results, "DQN": dqn_results}, f, indent=4)

        # Plot results
        self.plot_results(ddqn_results, dqn_results)


if __name__ == "__main__":
    try:
        tester = Stage1Tester()
        ddqn_model_path = "/home/alma/catkin_ws/src/dqn3/models/stage3/model_ep1300_20250102_105530.pth"  # Update with actual path
        dqn_model_path = "/home/alma/catkin_ws/src/dqn3/models/stage1_enhanced_dqn/model_ep1700_20250105_123219.pth"   # Replace with your DQN model path
        tester.run_evaluation(ddqn_model_path, dqn_model_path)
    except rospy.ROSInterruptException:
        pass
