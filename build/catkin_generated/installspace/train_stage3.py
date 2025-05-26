#!/usr/bin/env python3
import rospy
from nodes.turtlebot3_env1 import TurtleBot3Env
from nodes.dqn_agent import DQNAgent
import numpy as np
import os

def train_stage3():
    rospy.init_node('train_stage3', anonymous=True)

    # Initialize environment and agent
    env = TurtleBot3Env(goal_position=(2.0, 2.0), timeout=800, obstacle_enabled=True, complex_obstacles=True)
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    # Load weights from Stage 2
    agent.load("models/stage2_episode_300.pth")

    # Training parameters
    episodes = 300
    save_frequency = 100
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward
            steps += 1

            if done or steps >= env.timeout:
                print(f"Episode: {episode+1}, Steps: {steps}, Reward: {total_reward:.2f}")
                break

        # Save model at intervals
        if (episode + 1) % save_frequency == 0:
            agent.save(os.path.join(model_dir, f"stage3_episode_{episode+1}.pth"))

if __name__ == '__main__':
    try:
        train_stage3()
    except rospy.ROSInterruptException:
        pass
