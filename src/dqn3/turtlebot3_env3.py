#!/usr/bin/env python3

from collections import deque
import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
import time

class TurtleBot3Env3:
    def __init__(self, is_training=True, goal_position=(2.0, 2.0), timeout=600):
        # Robot Parameters
        self.MAX_LINEAR_SPEED = 0.26  # m/s
        self.MAX_ANGULAR_SPEED = 1.82  # rad/s
        self.ROBOT_RADIUS = 0.220  # meters
        self.SAFE_DISTANCE = self.ROBOT_RADIUS + 0.050  # Increased safety margin for dynamic obstacles
        
        # Environment Properties
        self.position = Point()
        self.heading = 0.0
        self.scan_ranges = []
        self.goal_position = Point(*goal_position, 0.0)
        self.action_size = 3
        self.timeout = timeout
        self.is_training = is_training
        self.start_time = rospy.get_time()
        
        # LiDAR Parameters
        self.MIN_SCAN_RANGE = 0.12
        self.MAX_SCAN_RANGE = 3.5
        self.SCAN_SECTORS = 24  # Number of sectors for laser scan
        
        # Enhanced Reward Tracking
        self._previous_distance = None
        self._min_distance_to_goal = float('inf')
        self._max_distance_to_goal = 0.0
        self._previous_heading = None
        self._total_distance = 0
        self._previous_position = None
        self._step_count = 0
        self._cumulative_reward = 0
        self._distance_history = deque(maxlen=10)
        self._heading_history = deque(maxlen=10)
        self._last_actions = deque(maxlen=5)
        self._progress_threshold = 0.05  # 5cm progress threshold
        
        # Initialize ROS components
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)
        
        # Initialize reset service
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def _compute_distance(self, pos1, pos2):
        """Compute Euclidean distance between points."""
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

    def _get_goal_angle(self):
        """Calculate angle to goal with normalization."""
        goal_angle = math.atan2(self.goal_position.y - self.position.y,
                              self.goal_position.x - self.position.x)
        heading_diff = goal_angle - self.heading
        
        # Normalize to [-pi, pi]
        if heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        elif heading_diff < -math.pi:
            heading_diff += 2 * math.pi
            
        return heading_diff

    def _segment_scans(self, scan_data, num_segments):
        """Segment scan data into specified number of sectors."""
        if len(scan_data) == 0:
            return [self.MAX_SCAN_RANGE] * num_segments
            
        segment_size = len(scan_data) // num_segments
        segments = []
        
        for i in range(0, len(scan_data) - segment_size + 1, segment_size):
            segment = scan_data[i:i + segment_size]
            segments.append(min(segment))  # Use minimum value in segment for safety
            
        # Ensure we have exactly num_segments
        while len(segments) < num_segments:
            segments.append(self.MAX_SCAN_RANGE)
            
        return segments[:num_segments]

    def reset(self):
        """Reset environment with enhanced initialization."""
        try:
            self.reset_sim()
            time.sleep(0.1)
            
            # Reset state variables
            self.scan_ranges = []
            self.position = Point()
            self.heading = 0.0
            self.start_time = rospy.get_time()
            
            # Reset tracking variables
            self._previous_distance = None
            self._min_distance_to_goal = float('inf')
            self._max_distance_to_goal = 0.0
            self._previous_heading = None
            self._total_distance = 0
            self._previous_position = None
            self._step_count = 0
            self._cumulative_reward = 0
            self._distance_history.clear()
            self._heading_history.clear()
            self._last_actions.clear()
            
            rospy.sleep(0.1)
            
            return self.get_state()[0]
            
        except Exception as e:
            rospy.logerr(f"Reset error: {str(e)}")
            raise

    def get_state(self):
        """Get current state with dynamic obstacle information."""
        scan_range = []
        done = False
        
        if self.scan_ranges:
            # Process laser scans
            for reading in self.scan_ranges:
                if np.isinf(reading) or np.isnan(reading):
                    scan_range.append(self.MAX_SCAN_RANGE)
                else:
                    scan_range.append(min(reading, self.MAX_SCAN_RANGE))
        
        # Ensure we have scan data
        while len(scan_range) < 360:  # Assuming 360 degree scan
            scan_range.append(self.MAX_SCAN_RANGE)
        
        # Segment into sectors
        scan_range = self._segment_scans(scan_range, self.SCAN_SECTORS)
        
        # Calculate distances and angles
        distance_to_goal = self._compute_distance(self.position, self.goal_position)
        angle_to_goal = self._get_goal_angle()
        
        # Check termination conditions
        if distance_to_goal < (self.ROBOT_RADIUS + 0.05):
            done = True
        elif min(scan_range) < self.SAFE_DISTANCE:
            done = True
        elif rospy.get_time() - self.start_time > self.timeout:
            done = True
            
        # Update tracking
        self._update_tracking(distance_to_goal)
        
        # Construct state vector [24 laser readings + goal angle + goal distance]
        state = np.concatenate([
            scan_range,
            [angle_to_goal, distance_to_goal]
        ])
        
        return state, done

    def _update_tracking(self, current_distance):
        """Update tracking variables for reward calculation."""
        self._min_distance_to_goal = min(self._min_distance_to_goal, current_distance)
        self._max_distance_to_goal = max(self._max_distance_to_goal, current_distance)
        
        if self._previous_position:
            step_distance = self._compute_distance(self.position, self._previous_position)
            self._total_distance += step_distance
            
        self._previous_position = Point(self.position.x, self.position.y, self.position.z)
        self._distance_history.append(current_distance)
        self._step_count += 1

    def _compute_reward(self, state, done):
        """Enhanced reward function with immediate feedback."""
        current_distance = self._compute_distance(self.position, self.goal_position)
        reward = 0
        
        # Terminal rewards
        if done:
            if current_distance < (self.ROBOT_RADIUS + 0.05):
                # Goal reached reward with efficiency bonus
                path_efficiency = self._min_distance_to_goal / max(self._total_distance, self._min_distance_to_goal)
                time_efficiency = max(0, 1 - (self._step_count / self.timeout))
                return 200 + (100 * path_efficiency) + (50 * time_efficiency)
            elif min(state[:self.SCAN_SECTORS]) < self.SAFE_DISTANCE:
                # Collision penalty with progress consideration
                progress_ratio = (self._max_distance_to_goal - current_distance) / self._max_distance_to_goal
                return -150 * (1 - progress_ratio)
            else:
                # Timeout penalty with progress consideration
                progress_ratio = (self._max_distance_to_goal - current_distance) / self._max_distance_to_goal
                return -50 * (1 - progress_ratio)
        
        # Progressive rewards
        if self._previous_distance is not None:
            # Distance progress reward
            progress = self._previous_distance - current_distance
            if abs(progress) > self._progress_threshold:
                # Significant progress reward
                reward += progress * 30
            else:
                # Small progress reward
                reward += progress * 10
        
        # Heading alignment reward
        goal_angle = self._get_goal_angle()
        if abs(goal_angle) < 0.1:  # Aligned within ~6 degrees
            reward += 5
        elif abs(goal_angle) < 0.3:  # Aligned within ~17 degrees
            reward += 2
        
        # Obstacle avoidance rewards
        min_scan = min(state[:self.SCAN_SECTORS])
        if min_scan < self.SAFE_DISTANCE * 2.0:
            # Progressive penalty based on proximity
            obstacle_penalty = ((self.SAFE_DISTANCE * 2.0 - min_scan) * 15)
            reward -= obstacle_penalty
        
        # Update tracking
        self._previous_distance = current_distance
        self._cumulative_reward += reward
        
        return reward

    def step(self, action):
        """Execute action and return next state, reward, and done flag."""
        try:
            # Execute action
            twist = Twist()
            if action == 0:  # Forward
                twist.linear.x = self.MAX_LINEAR_SPEED * 0.8
                twist.angular.z = 0.0
            elif action == 1:  # Left turn
                twist.linear.x = self.MAX_LINEAR_SPEED * 0.4
                twist.angular.z = self.MAX_ANGULAR_SPEED * 0.6
            else:  # Right turn
                twist.linear.x = self.MAX_LINEAR_SPEED * 0.4
                twist.angular.z = -self.MAX_ANGULAR_SPEED * 0.6
                
            self.cmd_vel_pub.publish(twist)
            self._last_actions.append(action)
            
            # Wait for execution
            time.sleep(0.1)
            
            # Get new state and calculate reward
            state, done = self.get_state()
            reward = self._compute_reward(state, done)
            
            return state, reward, done
            
        except Exception as e:
            rospy.logerr(f"Step error: {str(e)}")
            return self.get_state()[0], -100, True

    def odom_callback(self, msg):
        """Update robot odometry data."""
        self.position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.heading = yaw

    def scan_callback(self, msg):
        """Update laser scan data."""
        self.scan_ranges = msg.ranges

    def stop(self):
        """Stop the robot."""
        twist = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)