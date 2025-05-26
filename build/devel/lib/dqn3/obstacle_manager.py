#!/usr/bin/env python3

import rospy
import rospkg
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
import random
import math
import numpy as np
from std_msgs.msg import String

class ObstacleManager:
    def __init__(self):
        """Initialize the obstacle manager"""
        self.obstacles = []
        
        # Create a basic cylinder SDF template (changed from URDF to SDF for better Gazebo compatibility)
        self.obstacle_sdf = """
        <?xml version='1.0'?>
        <sdf version='1.6'>
            <model name='obstacle'>
                <static>true</static>
                <link name='link'>
                    <collision name='collision'>
                        <geometry>
                            <cylinder>
                                <radius>0.15</radius>
                                <length>0.5</length>
                            </cylinder>
                        </geometry>
                    </collision>
                    <visual name='visual'>
                        <geometry>
                            <cylinder>
                                <radius>0.15</radius>
                                <length>0.5</length>
                            </cylinder>
                        </geometry>
                        <material>
                            <ambient>0 0 0.8 1</ambient>
                            <diffuse>0 0 0.8 1</diffuse>
                            <specular>0.1 0.1 0.1 1</specular>
                            <emissive>0 0 0 0</emissive>
                        </material>
                    </visual>
                </link>
            </model>
        </sdf>
        """
        
        # Wait for gazebo services
        rospy.loginfo("Waiting for Gazebo spawn and delete services...")
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        rospy.loginfo("Gazebo services ready")

    def spawn_obstacles(self, num_obstacles, stage):
        """Spawn obstacles based on stage configuration"""
        rospy.loginfo(f"Spawning {num_obstacles} obstacles for stage {stage}")
        self.clear_obstacles()
        rospy.sleep(0.5)  # Wait for obstacles to clear
        
        if stage == 1:
            # Fixed positions for stage 1
            positions = [(1.0, 0.0)]
            rospy.loginfo("Stage 1: Using fixed obstacle position")
        else:
            # Random positions for stage 2
            positions = self._generate_random_positions(num_obstacles)
            rospy.loginfo(f"Stage 2: Generated {len(positions)} random positions")
            
        for i, (x, y) in enumerate(positions):
            self._spawn_single_obstacle(f'obstacle_{i}', x, y)
            rospy.sleep(0.1)  # Add small delay between spawns

    def _spawn_single_obstacle(self, name, x, y):
        """Spawn a single obstacle"""
        obstacle_pose = Pose()
        obstacle_pose.position.x = x
        obstacle_pose.position.y = y
        obstacle_pose.position.z = 0.25  # Raise it slightly above ground
        obstacle_pose.orientation.x = 0.0
        obstacle_pose.orientation.y = 0.0
        obstacle_pose.orientation.z = 0.0
        obstacle_pose.orientation.w = 1.0

        try:
            rospy.loginfo(f"Attempting to spawn obstacle {name} at ({x}, {y})")
            result = self.spawn_model(
                model_name=name,
                model_xml=self.obstacle_sdf,
                robot_namespace='/obstacle',
                initial_pose=obstacle_pose,
                reference_frame='world'
            )
            self.obstacles.append(name)
            rospy.loginfo(f"Successfully spawned obstacle {name}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn obstacle {name}: {e}")

    def clear_obstacles(self):
        """Remove all obstacles"""
        rospy.loginfo(f"Clearing {len(self.obstacles)} obstacles")
        for name in self.obstacles:
            try:
                self.delete_model(name)
                rospy.loginfo(f"Removed obstacle {name}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to remove obstacle {name}: {e}")
        self.obstacles = []
        rospy.sleep(0.5)