#!/usr/bin/env python3

import rospy
import numpy as np
import os
import rospkg
from turtlebot3_env import TurtleBot3Env
from nodes.dqn_agent2 import DQNAgent
import torch
from torch.serialization import add_safe_globals

class ModelTester:
    def __init__(self):
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('dqn3')
        self.checkpoint_dir = "/home/alma/catkin_ws/src/dqn3/checkpoints"  # Direct path
        
        # Test parameters
        self.NUM_TEST_EPISODES = 100
        self.success_threshold = 0.2
        
        # Metrics
        self.successes = 0
        self.total_steps = 0
        self.total_distance = 0
        self.collision_count = 0
        
    def find_best_checkpoint(self):
        """Find checkpoint with highest episode number"""
        try:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
            if not checkpoints:
                rospy.logwarn("No checkpoint files found in directory")
                return None
                
            rospy.loginfo(f"Found {len(checkpoints)} checkpoint files")
            
            # Get the checkpoint with highest episode number
            latest_checkpoint = max(checkpoints, 
                                 key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            rospy.loginfo(f"Selected checkpoint: {latest_checkpoint}")
            
            return checkpoint_path
            
        except Exception as e:
            rospy.logerr(f"Error searching for checkpoints: {e}")
            return None
        
    def test_model(self):
        rospy.init_node('test_stage2', anonymous=True)
        
        # Initialize environment and agent
        env = TurtleBot3Env(
            is_training=False,
            goal_position=(3.0, -2.0),
            timeout=500,
            stage=2
        )
        
        state_size = 26
        action_size = 3
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            batch_size=64,
            memory_size=1000000
        )
        
        # Load best checkpoint
        best_checkpoint_path = self.find_best_checkpoint()
        if best_checkpoint_path:
            try:
                # Load checkpoint without weights_only flag
                checkpoint = torch.load(best_checkpoint_path, 
                                     map_location=torch.device('cpu'))
                
                # Load model state dict
                agent.model.load_state_dict(checkpoint['model_state_dict'])
                agent.target_model.load_state_dict(checkpoint['model_state_dict'])
                rospy.loginfo(f"Successfully loaded weights from: {best_checkpoint_path}")
            except Exception as e:
                rospy.logerr(f"Error loading checkpoint: {e}")
                return
        else:
            rospy.logerr("No valid checkpoint found!")
            return
            
        agent.epsilon = 0.0  # No exploration during testing
        
        try:
            for episode in range(self.NUM_TEST_EPISODES):
                state = env.reset()
                rospy.sleep(0.1)  # Short delay for stability
                
                done = False
                episode_steps = 0
                initial_distance = env._compute_distance(env.position, env.goal_position)
                current_distance = initial_distance
                
                while not done and not rospy.is_shutdown():
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    state = next_state
                    episode_steps += 1
                    
                    # Get current distance to goal
                    current_distance = env._compute_distance(env.position, env.goal_position)
                    
                    # Check conditions
                    if episode_steps >= env.timeout:
                        rospy.loginfo("Episode timeout")
                        break
                        
                    if min(state[:-2]) < env.SAFE_DISTANCE:
                        self.collision_count += 1
                        rospy.loginfo("Collision detected")
                        break
                        
                    if current_distance < self.success_threshold:
                        self.successes += 1
                        self.total_distance += env._total_distance
                        rospy.loginfo("Goal reached!")
                        break
                    
                    rospy.sleep(0.1)  # Control rate
                        
                self.total_steps += episode_steps
                
                # Log episode results
                rospy.loginfo(f"\nEpisode {episode + 1}/{self.NUM_TEST_EPISODES}")
                rospy.loginfo(f"Steps: {episode_steps}")
                rospy.loginfo(f"Distance traveled: {env._total_distance:.2f}")
                rospy.loginfo(f"Final distance to goal: {current_distance:.2f}")
                
                # Short delay between episodes
                rospy.sleep(1.0)
                
            # Print final statistics
            if self.NUM_TEST_EPISODES > 0:
                success_rate = (self.successes / self.NUM_TEST_EPISODES) * 100
                avg_steps = self.total_steps / self.NUM_TEST_EPISODES
                avg_distance = self.total_distance / max(1, self.successes)
                collision_rate = (self.collision_count / self.NUM_TEST_EPISODES) * 100
                
                rospy.loginfo("\nFinal Test Results:")
                rospy.loginfo(f"Success Rate: {success_rate:.2f}%")
                rospy.loginfo(f"Average Steps: {avg_steps:.2f}")
                rospy.loginfo(f"Average Path Length: {avg_distance:.2f}m")
                rospy.loginfo(f"Collision Rate: {collision_rate:.2f}%")
            
        except Exception as e:
            rospy.logerr(f"Error during testing: {e}")
        finally:
            try:
                env.stop()
            except:
                pass

def main():
    tester = ModelTester()
    try:
        tester.test_model()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()