#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from collections import deque
import time
from std_msgs.msg import Bool

class TurtleBot3Env3:
    def __init__(self, is_training=True, goal_position=(2.0, 2.0), timeout=600):
        # Robot Parameters - matched with Stage 1
        self.MAX_LINEAR_SPEED = 0.26
        self.MAX_ANGULAR_SPEED = 1.82
        self.ROBOT_RADIUS = 0.220
        self.SAFE_DISTANCE = self.ROBOT_RADIUS + 0.1  # Same as Stage 1
        self.CRITICAL_DISTANCE = 0.17  # Fixed value like Stage 1
        
        # Environment Properties
        self.position = Point()
        self.heading = 0.0
        self.scan_ranges = np.array([])  # Initialize as empty numpy array
        self.goal_position = Point(*goal_position, 0.0)
        self.action_size = 5
        self.timeout = timeout
        self.is_training = is_training
        
        # State tracking
        self.collision = False
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
        # Laser Scanner Parameters
        self.MIN_SCAN_RANGE = 0.12
        self.MAX_SCAN_RANGE = 3.5
        self.SCAN_SECTORS = 24
        
        # Enhanced Reward Tracking
        self._previous_distance = None
        self._initial_distance = None
        self._min_distance_to_goal = float('inf')
        self._max_distance_to_goal = 0.0
        self._total_distance = 0
        self._step_count = 0
        self._previous_heading = None
        self._distance_history = deque(maxlen=10)
        self._heading_history = deque(maxlen=10)
        self._last_actions = deque(maxlen=5)
        self._progress_window = deque(maxlen=10)
        self._last_min_obstacle_dist = float('inf')
        
        # Initialize ROS components
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.collision_pub = rospy.Publisher('robot_collision', Bool, queue_size=1)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)
        
        # Initialize reset service
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        self.start_time = rospy.get_time()

    def reset(self):
        try:
            self.reset_sim()
            time.sleep(0.1)
            
            # Reset state variables
            self.scan_ranges = np.array([])
            self.position = Point()
            self.heading = 0.0
            self.collision = False
            self.current_linear_vel = 0.0
            self.current_angular_vel = 0.0
            
            # Reset tracking variables
            self._step_count = 0
            self._total_distance = 0
            self._previous_distance = None
            self._distance_history.clear()
            self._heading_history.clear()
            self._last_actions.clear()
            self._progress_window.clear()
            
            # Initialize distance measurements
            self._initial_distance = self._compute_distance(self.position, self.goal_position)
            self._min_distance_to_goal = self._initial_distance
            self._max_distance_to_goal = self._initial_distance
            self.start_time = rospy.get_time()
            
            rospy.sleep(0.1)  # Allow sensors to stabilize
            
            return self._get_state()
            
        except Exception as e:
            rospy.logerr(f"Reset error: {str(e)}")
            raise

    def step(self, action):
        """Execute action and return new state, reward, done"""
        self._step_count += 1
        
        # Execute action with Stage 1 compatible velocities
        twist = Twist()
        if action == 0:  # Forward
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.8
            twist.angular.z = 0.0
        elif action == 1:  # Left turn
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.4
            twist.angular.z = self.MAX_ANGULAR_SPEED * 0.5
        elif action == 2:  # Right turn
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.4
            twist.angular.z = -self.MAX_ANGULAR_SPEED * 0.5
        elif action == 3:  # Gentle left
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.6
            twist.angular.z = self.MAX_ANGULAR_SPEED * 0.3
        elif action == 4:  # Gentle right
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.6
            twist.angular.z = -self.MAX_ANGULAR_SPEED * 0.3
            
        self.cmd_vel_pub.publish(twist)
        self._last_actions.append(action)
        
        time.sleep(0.1)  # Allow action to take effect
        
        # Get new state and check termination
        state = self._get_state()
        done = self._is_done()
        reward = self._compute_reward(state, done)
        
        # Update tracking
        current_distance = self._compute_distance(self.position, self.goal_position)
        if self._previous_distance is not None:
            step_distance = abs(self._previous_distance - current_distance)
            self._total_distance += step_distance
        
        self._previous_distance = current_distance
        
        return state, reward, done

    def _get_state(self):
        """Get state representation matching Stage 1"""
        scan_range = []
        
        if len(self.scan_ranges) > 0:
            for reading in self.scan_ranges:
                if np.isinf(reading) or np.isnan(reading):
                    scan_range.append(self.MAX_SCAN_RANGE)
                else:
                    scan_range.append(min(reading, self.MAX_SCAN_RANGE))
        
        # Ensure we have scan data
        while len(scan_range) < 360:
            scan_range.append(self.MAX_SCAN_RANGE)
        
        # Segment into sectors (24)
        scan_range = self._segment_scans(scan_range, self.SCAN_SECTORS)
        
        # Calculate goal info
        distance_to_goal = self._compute_distance(self.position, self.goal_position)
        angle_to_goal = self._get_goal_angle()
        
        # Match Stage 1's state vector format
        state = np.concatenate([
            scan_range,  # 24 laser readings
            [angle_to_goal, distance_to_goal],  # 2 goal info
            [self.current_linear_vel / self.MAX_LINEAR_SPEED, 
             self.current_angular_vel / self.MAX_ANGULAR_SPEED]  # 2 velocity info
        ])
        
        return state

    def _compute_reward(self, state, done):
        """Enhanced reward function for dynamic environment"""
        current_distance = self._compute_distance(self.position, self.goal_position)
        min_scan = min(state[:self.SCAN_SECTORS])
        reward = 0
        
        # Terminal rewards
        if done:
            if current_distance < (self.ROBOT_RADIUS + 0.05):
                # Goal reached reward with efficiency bonuses
                path_efficiency = self._min_distance_to_goal / max(self._total_distance, self._min_distance_to_goal)
                time_efficiency = max(0, 1 - (self._step_count / self.timeout))
                obstacle_avoidance = min(1.0, min_scan / (self.SAFE_DISTANCE * 2))
                return 200 + (100 * path_efficiency) + (50 * time_efficiency) + (50 * obstacle_avoidance)
                
            elif self.collision:
                progress_ratio = (self._max_distance_to_goal - current_distance) / self._max_distance_to_goal
                return -200 * (1 - 0.5 * progress_ratio)
            
            else:  # Timeout
                progress_ratio = (self._max_distance_to_goal - current_distance) / self._max_distance_to_goal
                return -100 * (1 - progress_ratio)
        
        # Progressive rewards
        if self._previous_distance is not None:
            # Distance progress reward
            progress = self._previous_distance - current_distance
            self._progress_window.append(progress)
            avg_progress = sum(self._progress_window) / len(self._progress_window)
            
            if avg_progress > 0.02:  # Significant progress
                reward += 20
            elif avg_progress > 0:  # Small progress
                reward += 10
            else:  # Moving away
                reward -= 15
        
        # Heading alignment reward
        goal_angle = self._get_goal_angle()
        angle_reward = math.cos(goal_angle) * 15
        reward += angle_reward
        
        # Obstacle avoidance with Stage 1 thresholds
        if min_scan < (self.SAFE_DISTANCE * 1.5):  # Reduced detection distance
            safety_factor = ((self.SAFE_DISTANCE * 1.5) - min_scan) / (self.SAFE_DISTANCE * 1.5)
            safety_penalty = -20 * safety_factor
            reward += safety_penalty
        
        return reward

    def _is_done(self):
        """Terminal conditions using Stage 1's thresholds"""
        if len(self.scan_ranges) == 0:
            return False
            
        if self.collision:
            return True
            
        current_distance = self._compute_distance(self.position, self.goal_position)
        if current_distance < (self.ROBOT_RADIUS + 0.05):
            return True
            
        if rospy.get_time() - self.start_time > self.timeout:
            return True
            
        return False

    def _segment_scans(self, scan_data, num_segments):
        """Convert laser scan data into fixed number of segments"""
        if len(scan_data) == 0:
            return [self.MAX_SCAN_RANGE] * num_segments
            
        segment_size = len(scan_data) // num_segments
        segments = []
        
        for i in range(0, len(scan_data), segment_size):
            segment = scan_data[i:i + segment_size]
            segments.append(min(segment))
            
        return segments[:num_segments]

    def _compute_distance(self, pos1, pos2):
        """Compute Euclidean distance between points"""
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

    def _get_goal_angle(self):
        """Calculate angle to goal relative to robot heading"""
        goal_angle = math.atan2(self.goal_position.y - self.position.y,
                               self.goal_position.x - self.position.x)
        heading_diff = goal_angle - self.heading
        
        # Normalize to [-pi, pi]
        if heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        elif heading_diff < -math.pi:
            heading_diff += 2 * math.pi
            
        return heading_diff

    def odom_callback(self, msg):
        """Update robot pose and velocities"""
        self.position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y,
                                         orientation.z, orientation.w])
        self.heading = yaw
        
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z

    def scan_callback(self, msg):
        """Handle laser scan updates with Stage 1 thresholds"""
        self.scan_ranges = np.array(msg.ranges)
        
        if len(self.scan_ranges) > 0:
            min_distance = min([x for x in self.scan_ranges if not np.isinf(x) and not np.isnan(x)], default=float('inf'))
            
            # Check collision using Stage 1's critical distance
            previous_collision_state = self.collision
            self.collision = min_distance < self.CRITICAL_DISTANCE
            
            # Publish collision event only when collision first occurs
            if self.collision and not previous_collision_state:
                self.collision_pub.publish(Bool(True))
                rospy.logwarn(f"Collision detected! Min distance: {min_distance:.2f}m")