#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
import time

class TurtleBot3Env:
    def __init__(self, is_training=True, goal_position=(3.0, -2.0), timeout=700, stage=2):
        # First initialize goal position and basic properties
        self.goal_position = Point(*goal_position, 0.0)
        self.is_training = is_training
        self.stage = stage
        self.timeout = timeout
        self.action_size = 3  # [forward, left, right]
        self._step_count = 0
        # Robot Physical Parameters - Rectangular Footprint
        self.MAX_LINEAR_SPEED = 0.26  # m/s
        self.MAX_ANGULAR_SPEED = 1.82  # rad/s
        
        # Precise rectangular dimensions from specs
        self.ROBOT_LENGTH = 0.281  # meters (281mm)
        self.ROBOT_WIDTH = 0.306   # meters (306mm)
        self.WHEEL_RADIUS = 0.033  # meters (66mm diameter)

        # Half dimensions for collision checking
        self.HALF_LENGTH = self.ROBOT_LENGTH / 2
        self.HALF_WIDTH = self.ROBOT_WIDTH / 2
        
        # Calculate effective radius (used for goal reaching)
        self.ROBOT_RADIUS = math.sqrt((self.HALF_LENGTH)**2 + (self.HALF_WIDTH)**2)

        # Safety margins
        self.FRONT_SAFETY_MARGIN = 0.05  # 5cm front safety margin
        self.SIDE_SAFETY_MARGIN = 0.03   # 3cm side safety margin
        self.SAFE_DISTANCE = self.ROBOT_RADIUS + 0.05  # General safety distance
        
        # LDS-01 Sensor Parameters
        self.MIN_SCAN_RANGE = 0.12  # meters
        self.MAX_SCAN_RANGE = 3.5   # meters
        self.N_SCAN_SAMPLES = 24    # Downsampled from 360 to 24 points
        
        # Initialize state variables
        self.position = Point()
        self.heading = 0.0
        self.scan_ranges = []
        self.start_time = time.time()  # Changed to system time
        
        # Initialize tracking variables
        self._previous_distance = None
        self._previous_pos = None
        self._total_distance = 0
        self._step_count = 0
        self._initial_distance = self._compute_distance(Point(0, 0, 0), self.goal_position)
        self._previous_min_scan = None
        
        # Initialize publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)

        # ROS Service Proxies
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def stop(self):
        """Stop the robot's movement."""
        try:
            # Create stop command
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            
            # Publish stop command multiple times to ensure robot stops
            for _ in range(5):
                self.cmd_vel_pub.publish(twist)
                rospy.sleep(0.1)
            
            rospy.loginfo("Robot stopped successfully")
            
        except Exception as e:
            rospy.logerr(f"Error stopping robot: {str(e)}")
            
        finally:
            # Try to close subscribers and publishers cleanly
            try:
                self.cmd_vel_pub.unregister()
                self.odom_sub.unregister()
                self.scan_sub.unregister()
            except:
                pass

    def __del__(self):
        """Destructor to ensure robot stops when object is deleted."""
        self.stop()

    def reset(self):
        """Reset environment with additional Stage 2 initializations"""
        try:
            # Reset simulation
            self.reset_sim()
            time.sleep(0.5)  # Changed to system time
            self._step_count = 0
            # Reset state variables
            self.scan_ranges = []
            self.position = Point()
            self.heading = 0.0
            self._previous_distance = None
            self._previous_min_scan = None
            self._total_distance = 0
            self.start_time = time.time()  # Changed to system time
            
            # Get initial state
            state, _ = self.get_state()
            return state
            
        except Exception as e:
            rospy.logerr("Reset error: {}".format(str(e)))
            raise

    def step(self, action):
        """Execute an action and return the new state, reward, and done status."""
        self._execute_action(action)
        time.sleep(0.1)  # Changed to system time

        state, done = self.get_state()
        reward = self._compute_reward(state, done)

        # Add safety timeout based on total distance
        if self._total_distance > self._initial_distance * 3:
            done = True
            reward = -20

        return state, reward, done



    def get_state(self):
        """Get the current observation state with consistent dimensions."""
        scan_range = []
        done = False
        reward = 0  # Initialize reward
        
        if self.scan_ranges:
            # Always ensure we get exactly 24 scan points
            step = max(1, len(self.scan_ranges) // 24)
            for i in range(0, min(len(self.scan_ranges), 24 * step), step):
                range_value = self.scan_ranges[i]
                
                if range_value == float('Inf') or range_value > self.MAX_SCAN_RANGE:
                    scan_range.append(self.MAX_SCAN_RANGE)
                elif np.isnan(range_value) or range_value < self.MIN_SCAN_RANGE:
                    scan_range.append(self.MIN_SCAN_RANGE)
                else:
                    scan_range.append(range_value)
        
        # Pad or truncate to exactly 24 points
        if len(scan_range) < 24:
            scan_range.extend([self.MAX_SCAN_RANGE] * (24 - len(scan_range)))
        elif len(scan_range) > 24:
            scan_range = scan_range[:24]
        
        distance_to_goal = self._compute_distance(self.position, self.goal_position)
        
        # Create state with exact dimensions
        state = np.zeros(26, dtype=np.float32)  # 24 scans + heading + distance
        state[:24] = scan_range
        state[24] = self.heading
        state[25] = distance_to_goal
        
        # Check termination conditions using HALF_LENGTH and HALF_WIDTH
        if distance_to_goal < (min(self.HALF_LENGTH, self.HALF_WIDTH) + 0.05):
            done = True
            reward = 200  # Goal reached
        elif min(scan_range) < self._get_safe_distance(self.heading):
            done = True
            reward = -200  # Collision
        elif rospy.get_time() - self.start_time > self.timeout:
            done = True
            reward = -50  # Timeout
        
        return state, done

    

    def _get_safe_distance(self, angle):
        """
        Calculate safe distance based on angle relative to robot's heading.
        Args:
            angle: Angle in radians relative to robot's heading
        Returns:
            Safe distance in meters
        """
        # Normalize angle to [-pi, pi]
        angle = (angle + self.heading) % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
            
        # Calculate the distance from center to edge of robot at this angle
        abs_angle = abs(angle)
        if abs_angle < 0.1 or abs_angle > np.pi - 0.1:  # Front/back
            base_distance = self.HALF_LENGTH
            safety_margin = self.FRONT_SAFETY_MARGIN
        elif abs_angle > np.pi/2 - 0.1 and abs_angle < np.pi/2 + 0.1:  # Sides
            base_distance = self.HALF_WIDTH
            safety_margin = self.SIDE_SAFETY_MARGIN
        else:
            # For diagonal angles, use the actual distance to the corner
            dx = self.HALF_LENGTH * np.cos(angle)
            dy = self.HALF_WIDTH * np.sin(angle)
            base_distance = np.sqrt(dx*dx + dy*dy)
            # Interpolate safety margin
            front_weight = abs(np.cos(angle))
            safety_margin = (self.FRONT_SAFETY_MARGIN * front_weight + 
                           self.SIDE_SAFETY_MARGIN * (1 - front_weight))
        
        return base_distance + safety_margin

    def _compute_reward(self, state, done):
        """Enhanced reward function for Stage 2"""
        distance_to_goal = self._compute_distance(self.position, self.goal_position)
        
        # Terminal rewards
        if done:
            if distance_to_goal < (min(self.HALF_LENGTH, self.HALF_WIDTH) + 0.05):
                # Enhanced goal reward with efficiency bonus
                time_efficiency = max(0, 1 - (self._step_count / 300))
                path_efficiency = self._initial_distance / max(self._total_distance, self._initial_distance)
                
                return 200 + (100 * path_efficiency) + (50 * time_efficiency)
                
            # Enhanced collision penalty
            if min(state[:-2]) < self._get_safe_distance(self.heading):
                # Higher penalty near goal
                if distance_to_goal < self._initial_distance * 0.3:
                    return -150
                return -100
                
            return -50  # Timeout penalty
        
        # Progressive rewards
        reward = 0
        
        # Enhanced distance-based reward
        if self._previous_distance is not None:
            progress = self._previous_distance - distance_to_goal
            # Increased scaling near goal
            distance_scale = 2.0 / max(distance_to_goal, 0.1)
            progress_reward = progress * 25 * distance_scale
            reward += np.clip(progress_reward, -15, 15)
        
        # Enhanced heading reward with dynamic scaling
        goal_angle = math.atan2(self.goal_position.y - self.position.y,
                            self.goal_position.x - self.position.x)
        angle_diff = abs(self.heading - goal_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
            
        # Scale heading reward based on progress
        heading_scale = 1.5 / max(distance_to_goal, 0.1)
        heading_reward = (1 - angle_diff/math.pi) * 8 * heading_scale
        reward += heading_reward
        
        # Enhanced obstacle avoidance
        scan_ranges = state[:-2]
        min_scan = min(scan_ranges)
        safe_dist = self._get_safe_distance(self.heading)
        
        # Progressive obstacle penalties
        if min_scan < safe_dist * 2.0:
            # Increased penalty close to obstacles
            obstacle_penalty = ((safe_dist * 2.0 - min_scan) * 15)
            reward -= obstacle_penalty
            
            # Extra penalty for dangerous proximity
            if min_scan < safe_dist * 1.2:
                reward -= 20
                
            # Additional penalty when close to goal
            if distance_to_goal < self._initial_distance * 0.3:
                reward -= obstacle_penalty * 0.5
        
        # Smaller step penalty
        reward -= 0.05  # Reduced from 0.1
        
        # Update tracking
        self._previous_distance = distance_to_goal
        if self._previous_pos is not None:
            self._total_distance += self._compute_distance(self._previous_pos, self.position)
        self._previous_pos = Point(self.position.x, self.position.y, self.position.z)
        
        return np.clip(reward, -25, 25)

    def _execute_action(self, action):
        """Refined action execution for Stage 2"""
        twist = Twist()
        if action == 0:  # Move forward
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.85  # Increased from 0.8
            twist.angular.z = 0.0
        elif action == 1:  # Turn left
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.35  # Increased from 0.3
            twist.angular.z = self.MAX_ANGULAR_SPEED * 0.55  # Reduced from 0.6
        elif action == 2:  # Turn right
            twist.linear.x = self.MAX_LINEAR_SPEED * 0.35
            twist.angular.z = -self.MAX_ANGULAR_SPEED * 0.55
        self.cmd_vel_pub.publish(twist)


    def _compute_distance(self, pos1, pos2):
        """Helper function to compute distance between two points."""
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

    def odom_callback(self, data):
        """Callback to update robot position and heading from odometry data."""
        self.position = data.pose.pose.position
        orientation = data.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.heading = yaw

    def scan_callback(self, data):
        """Callback to update laser scan data."""
        self.scan_ranges = data.ranges

def stop(self):
        """Stop the robot's movement."""
        twist = Twist()
        try:
            # Publish stop command multiple times
            for _ in range(5):
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.1)  # Changed to system time
        except Exception as e:
            rospy.logerr("Error stopping robot: {}".format(str(e)))
            
