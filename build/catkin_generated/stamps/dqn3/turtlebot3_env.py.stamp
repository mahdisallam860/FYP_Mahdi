#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion
import time
import random

class TurtleBot3Env:
    def __init__(self, is_training=True, timeout=500):
        """Initialize the TurtleBot3 environment"""
        
        # Robot Physical Parameters
        self.MAX_LINEAR_SPEED = 0.26  # m/s (from TurtleBot3 specs)
        self.MAX_ANGULAR_SPEED = 1.82  # rad/s
        self.ROBOT_RADIUS = 0.15  # Simplified collision radius
        self.SAFE_DISTANCE = self.ROBOT_RADIUS + 0.10  # Added safety margin
        
        # Environment Boundaries (for goal generation)
        self.MIN_X = -2.5
        self.MAX_X = 2.5
        self.MIN_Y = -2.5
        self.MAX_Y = 2.5
        
        # Laser Scanner Parameters
        self.MIN_SCAN_RANGE = 0.12  # meters
        self.MAX_SCAN_RANGE = 3.5   # meters
        self.N_SCAN_SAMPLES = 24    # Number of samples to use
        
        # State and Action Space
        self.state_size = self.N_SCAN_SAMPLES + 2  # laser scans + [distance_to_goal, angle_to_goal]
        self.action_size = 5  # [stop, forward, strong_left, left, right, strong_right]
        
        # Training Parameters
        self.is_training = is_training
        self.timeout = timeout
        
        # Performance Tracking
        self.episode_step = 0
        self.previous_distance = None
        self.collision_count = 0
        self.goal_count = 0
        self.total_reward = 0
        self.min_distance_to_goal = float('inf')
        
        # Initialize ROS node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('turtlebot3_env_node')
            
        # Initialize ROS publishers/subscribers
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.goal_marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)
        
        # Initialize position and scan data
        self.position = Point()
        self.heading = 0.0
        self.scan_ranges = []
        
        # Initialize goal
        self.goal_position = None
        
        # Wait for subscribers to connect
        time.sleep(1)
        
        # Initialize reset service
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def _generate_random_goal(self):
        """Generate a random goal position avoiding obstacles"""
        for _ in range(100):  # Maximum attempts to find valid goal
            x = random.uniform(self.MIN_X, self.MAX_X)
            y = random.uniform(self.MIN_Y, self.MAX_Y)
            
            if self._is_valid_goal_position(x, y):
                goal = Point(x, y, 0)
                self._visualize_goal(goal)
                return goal
                
        # Fallback to a safe default if no valid position found
        return Point(2.0, 2.0, 0)

    def _is_valid_goal_position(self, x, y):
        """Check if a goal position is valid"""
        # Stage 1 obstacle positions (cylinders)
        obstacles = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        
        # Minimum distances
        min_dist_to_obstacle = 0.6  # Increased for better paths
        min_dist_to_start = 1.0     # Ensure meaningful episodes
        
        # Check distance from each obstacle
        for obs_x, obs_y in obstacles:
            dist = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dist < min_dist_to_obstacle:
                return False
        
        # Check distance from start position
        if math.sqrt(x**2 + y**2) < min_dist_to_start:
            return False
            
        return True

    def _visualize_goal(self, goal):
        """Visualize the goal position in Gazebo"""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal_marker"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set the pose
        marker.pose.position.x = goal.x
        marker.pose.position.y = goal.y
        marker.pose.position.z = 0.15
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set the scale
        marker.scale.x = 0.2  # Diameter
        marker.scale.y = 0.2
        marker.scale.z = 0.3  # Height
        
        # Set the color (green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Set marker lifetime
        marker.lifetime = rospy.Duration(0)  # 0 = forever
        
        self.goal_marker_pub.publish(marker)

    def _get_state(self):
        """Get the current state with proper normalization"""
        # Process laser scans
        scan_range = self._process_scan_ranges()
        
        # Compute distance and angle to goal
        current_distance = self._compute_distance(self.position, self.goal_position)
        angle_to_goal = self._compute_angle_to_goal()
        
        # Update minimum distance for reward shaping
        self.min_distance_to_goal = min(self.min_distance_to_goal, current_distance)
        
        # Normalize values
        normalized_scans = [min(x / self.MAX_SCAN_RANGE, 1.0) for x in scan_range]
        normalized_distance = current_distance / math.sqrt((self.MAX_X - self.MIN_X)**2 + 
                                                         (self.MAX_Y - self.MIN_Y)**2)
        normalized_angle = angle_to_goal / math.pi
        
        return np.array(normalized_scans + [normalized_distance, normalized_angle], 
                       dtype=np.float32)

    def _compute_reward(self):
        """Compute reward and check termination conditions"""
        current_distance = self._compute_distance(self.position, self.goal_position)
        
        # Initialize reward and done flag
        reward = 0
        done = False
        
        # Check collision
        min_scan = min(self._process_scan_ranges())
        if min_scan < self.SAFE_DISTANCE:
            reward = -200  # Severe penalty for collision
            done = True
            self.collision_count += 1
            return reward, done
            
        # Check goal reached
        if current_distance < self.SAFE_DISTANCE:
            # Reward includes bonus for reaching goal quickly
            reward = 200 + (self.timeout - self.episode_step) * 2
            done = True
            self.goal_count += 1
            return reward, done
            
        # Distance-based reward
        if self.previous_distance is not None:
            # Reward for moving towards goal, penalty for moving away
            distance_reward = (self.previous_distance - current_distance) * 20
            reward += distance_reward
            
            # Bonus for achieving new minimum distance to goal
            if current_distance < self.min_distance_to_goal:
                reward += 5
        
        # Heading alignment reward
        angle = abs(self._compute_angle_to_goal())
        if angle < math.pi/6:  # Aligned within 30 degrees
            reward += 0.5
        elif angle > 2*math.pi/3:  # Facing away (more than 120 degrees)
            reward -= 1.0
            
        # Penalty for being too close to obstacles
        if min_scan < self.SAFE_DISTANCE * 1.5:  # 1.5 times safety distance
            reward -= ((self.SAFE_DISTANCE * 1.5 - min_scan) * 10)
        
        # Small time penalty to encourage efficiency
        reward -= 0.1
        
        # Check timeout
        if self.episode_step >= self.timeout:
            reward -= 50  # Penalty for timeout
            done = True
            
        self.previous_distance = current_distance
        self.total_reward += reward
        
        return reward, done

    def _process_scan_ranges(self):
        """Process and downsample laser scan data"""
        scan_range = []
        
        if len(self.scan_ranges) > 0:
            # Calculate step size for downsampling
            step = len(self.scan_ranges) // self.N_SCAN_SAMPLES
            
            for i in range(0, len(self.scan_ranges), step):
                range_value = self.scan_ranges[i]
                
                # Handle invalid measurements
                if np.isnan(range_value) or range_value < self.MIN_SCAN_RANGE:
                    scan_range.append(self.MIN_SCAN_RANGE)
                elif range_value > self.MAX_SCAN_RANGE:
                    scan_range.append(self.MAX_SCAN_RANGE)
                else:
                    scan_range.append(range_value)
        
        # Ensure we have exactly N_SCAN_SAMPLES
        while len(scan_range) < self.N_SCAN_SAMPLES:
            scan_range.append(self.MAX_SCAN_RANGE)
        
        return scan_range[:self.N_SCAN_SAMPLES]

    def _execute_action(self, action):
        """Execute the selected action"""
        twist = Twist()
        
        if action == 0:  # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif action == 1:  # Forward
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.8
            twist.angular.z = 0.0
        elif action == 2:  # Strong left
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.2
            twist.angular.z = self.MAX_ANGULAR_SPEED * 0.8
        elif action == 3:  # Left
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.4
            twist.angular.z = self.MAX_ANGULAR_SPEED * 0.4
        elif action == 4:  # Right
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.4
            twist.angular.z = -self.MAX_ANGULAR_SPEED * 0.4
            
        self.cmd_vel_pub.publish(twist)

    def _compute_distance(self, pos1, pos2):
        """Compute Euclidean distance between two points"""
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

    def _compute_angle_to_goal(self):
        """Compute angle between robot heading and goal direction"""
        goal_angle = math.atan2(self.goal_position.y - self.position.y,
                               self.goal_position.x - self.position.x)
        angle = goal_angle - self.heading
        
        # Normalize to [-pi, pi]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
            
        return angle

    def reset(self):
        """Reset the environment and robot state"""
        # Reset simulation
        self.reset_sim()
        rospy.sleep(0.5)  # Wait for reset to complete
        
        # Reset internal variables
        self.episode_step = 0
        self.previous_distance = None
        self.min_distance_to_goal = float('inf')
        self.total_reward = 0
        
        # Generate new goal position
        self.goal_position = self._generate_random_goal()
        
        # Get initial state
        state = self._get_state()
        
        return state

    def step(self, action):
        """Execute action and return new state, reward, done"""
        self.episode_step += 1
        
        # Execute action
        self._execute_action(action)
        rospy.sleep(0.1)  # Allow time for action to have effect
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward and check if episode is done
        reward, done = self._compute_reward()
        
        return state, reward, done

    def odom_callback(self, data):
        """Update robot position and heading from odometry"""
        self.position = data.pose.pose.position
        orientation = data.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, 
                                         orientation.z, orientation.w])
        self.heading = yaw

    def scan_callback(self, data):
        """Update laser scan data"""
        self.scan_ranges = data.ranges

    def stop(self):
        """Stop the robot safely"""
        twist = Twist()
        for _ in range(5):  # Send multiple stop commands
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)