import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
import time
from std_msgs.msg import Bool

class TurtleBot3Env:
    def __init__(self, is_training=True, goal_position=(2.0, 2.0), timeout=400, stage=1):
        # Robot Physical Parameters
        self.MAX_LINEAR_SPEED = 0.26  # m/s
        self.MAX_ANGULAR_SPEED = 1.82  # rad/s
        self.ROBOT_RADIUS = 0.220  # meters
        
        # Adjust these distance parameters
        self.SAFE_DISTANCE = self.ROBOT_RADIUS + 0.1  # Reduced from 0.2 to 0.1
        self.CRITICAL_DISTANCE = 0.17  # Fixed value instead of relative to SAFE_DISTANCE
        
        # Laser Scanner Parameters
        self.MIN_SCAN_RANGE = 0.12  # meters
        self.MAX_SCAN_RANGE = 3.5   # meters
            
        # Environment Properties
        self.position = Point()
        self.heading = 0.0
        self.scan_ranges = []
        self.goal_position = Point(*goal_position, 0.0)
        self.action_size = 5  # [forward, left, right, gentle_left, gentle_right]
        self.timeout = timeout
        self.is_training = is_training
        self.stage = stage
        
        # State tracking
        self.collision = False
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.previous_action = None
        self._initial_distance = None
        self._previous_distance = None
        self._step_count = 0
        self._min_distance_to_goal = float('inf')
        self._total_distance = 0
        self._path_points = []
        
        # Performance tracking
        self.episode_rewards = []
        self.collision_count = 0
        self.success_count = 0
        self.total_steps = 0
        
        # Initialize ROS components
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.collision_pub = rospy.Publisher('robot_collision', Bool, queue_size=1)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)  # Fixed the truncated callback name
    
        # Reset service
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            
        # Environment bounds and obstacles
        self.bounds = {
            'x_min': -3.0, 'x_max': 3.0,
            'y_min': -3.0, 'y_max': 3.0,
            'safety_margin': 0.3
        }
        self.obstacles = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        self.obstacle_radius = 0.3
        
        # Training parameters
        self.start_time = rospy.get_time()

    def reset(self):
        """Reset the environment to initial state."""
        try:
            self.reset_sim()
            time.sleep(0.1)
            
            # Reset state variables
            self.scan_ranges = []
            self.position = Point()
            self.heading = 0.0
            self.collision = False
            self.current_linear_vel = 0.0
            self.current_angular_vel = 0.0
            self.previous_action = None
            self._step_count = 0
            self._path_points = []
            
            # Set initial measurements
            self._initial_distance = self._compute_distance(self.position, self.goal_position)
            self._previous_distance = self._initial_distance
            self._min_distance_to_goal = self._initial_distance
            self.start_time = rospy.get_time()
            
            # Get initial state
            rospy.sleep(0.1)  # Wait for sensors to stabilize
            state = self._get_state()
            return state
            
        except Exception as e:
            rospy.logerr(f"Reset error: {str(e)}")
            raise

    def step(self, action):
        """Execute one environment step."""
        self._step_count += 1
        self.previous_action = action
        
        # Execute action
        self._execute_action(action)
        time.sleep(0.1)  # Allow action to take effect
        
        # Get new state and check terminal conditions
        state = self._get_state()
        done = self._is_terminal()
        reward = self._compute_reward(state, done)
        
        # Update tracking
        self._path_points.append((self.position.x, self.position.y))
        current_distance = self._compute_distance(self.position, self.goal_position)
        self._min_distance_to_goal = min(self._min_distance_to_goal, current_distance)
        
        # Update performance metrics
        if done:
            if self.collision:
                self.collision_count += 1
            elif current_distance < (self.ROBOT_RADIUS + 0.05):
                self.success_count += 1
            self.total_steps += self._step_count
            
        return state, reward, done

    def _get_state(self):
        """Get the current state representation."""
        # Process laser scans
        scan_ranges = self._process_laser_scans()
        
        # Calculate goal distance and angle
        distance_to_goal = self._compute_distance(self.position, self.goal_position)
        angle_to_goal = self._get_goal_angle()
        
        # Get velocity information
        normalized_linear_vel = self.current_linear_vel / self.MAX_LINEAR_SPEED
        normalized_angular_vel = self.current_angular_vel / self.MAX_ANGULAR_SPEED
        
        # Combine all state components
        state = np.concatenate([
            scan_ranges,
            [angle_to_goal, distance_to_goal],
            [normalized_linear_vel, normalized_angular_vel]
        ])
        
        return state

    def _compute_reward(self, state, done):
        """Calculate the reward for the current state."""
        current_distance = self._compute_distance(self.position, self.goal_position)
        min_scan = min(state[:-4])  # Laser readings excluding goal and velocity info
        
        # Terminal rewards
        if done:
            if current_distance < (self.ROBOT_RADIUS + 0.05):
                # Goal reached - reward includes enhanced efficiency bonus
                efficiency_bonus = self._calculate_efficiency_bonus()
                path_ratio = self._initial_distance / max(self._total_distance, 0.1)
                direct_path_bonus = 200 * path_ratio  # Extra bonus for direct paths
                return 500 + efficiency_bonus + direct_path_bonus
                
            elif min_scan < self.SAFE_DISTANCE:
                # Collision penalty scales with proximity to goal
                progress_factor = 1 - (current_distance / self._initial_distance)
                collision_penalty = -300 * (1 + progress_factor)
                return max(collision_penalty, -500)
                
            # Increased timeout penalty based on path length
            path_inefficiency = self._total_distance / (self._initial_distance * 1.2)
            return -100 * path_inefficiency
        
        # Initialize running reward
        reward = 0
        
        # Enhanced distance progress reward
        if self._previous_distance is not None:
            progress = self._previous_distance - current_distance
            # Higher reward for progress when moving straight
            straight_motion_factor = 1 - abs(self.current_angular_vel) / self.MAX_ANGULAR_SPEED
            progress_reward = progress * 30 * (1 + straight_motion_factor)
            reward += np.clip(progress_reward, -10, 25)
        
        # Forward motion reward with path optimization
        if self.current_linear_vel > 0:
            angle_to_goal = abs(self._get_goal_angle())
            alignment_factor = 1 - angle_to_goal / math.pi
            # Higher reward for straight motion toward goal
            if angle_to_goal < math.pi/6:  # Within 30 degrees
                forward_reward = self.current_linear_vel * alignment_factor * 15  # Increased from 10
                reward += forward_reward
        
        # Enhanced heading alignment reward
        angle_to_goal = abs(self._get_goal_angle())
        distance_factor = min(1.0, current_distance)
        alignment_reward = (1 - angle_to_goal/math.pi) * 20 * distance_factor  # Increased from 15
        
        # Extra reward for maintaining good alignment
        if angle_to_goal < math.pi/6:  # Within 30 degrees
            alignment_reward *= 1.5
        reward += alignment_reward
        
        # Safety reward (unchanged)
        safety_margin = self.SAFE_DISTANCE * 1.5
        if min_scan < safety_margin:
            safety_factor = (safety_margin - min_scan) / safety_margin
            safety_penalty = -20 * np.exp(safety_factor * 2)
            reward += safety_penalty
        
        # Progressive time penalty (increases with path length)
        path_length_factor = self._total_distance / (self._initial_distance * 1.2)
        reward -= 0.1 * (1 + path_length_factor)
        
        self._previous_distance = current_distance
        return reward

    def _calculate_efficiency_bonus(self):
        """Enhanced efficiency bonus calculation."""
        # Calculate straight line efficiency
        path_efficiency = self._initial_distance / max(self._total_distance, 0.1)
        
        # Calculate time efficiency
        time_efficiency = 1.0 - (self._step_count / self.timeout)
        
        # Higher weight to path efficiency
        bonus = 200 * (0.7 * path_efficiency + 0.3 * time_efficiency)
        
        # Extra bonus for very direct paths
        if path_efficiency > 0.8:
            bonus *= 1.5
            
        return bonus

    def _execute_action(self, action):
        """Execute the selected action."""
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
        self.current_linear_vel = twist.linear.x
        self.current_angular_vel = twist.angular.z

    def _calculate_efficiency_bonus(self):
        """Calculate bonus reward based on path efficiency."""
        optimal_steps = self._initial_distance / (self.MAX_LINEAR_SPEED * 0.8)
        efficiency = optimal_steps / max(self._step_count, 1)
        return 200 * min(efficiency, 1.0)

    def _process_laser_scans(self):
        """Process raw laser scans into state representation."""
        if len(self.scan_ranges) == 0:
            return np.array([self.MAX_SCAN_RANGE] * 24)
            
        processed_scans = []
        for scan in self.scan_ranges:
            if np.isinf(scan) or np.isnan(scan):
                processed_scans.append(3.5)  # Max range
            else:
                processed_scans.append(min(scan, 3.5))
                
        # Downsample to 24 points
        return self._downsample_scans(processed_scans, 24)

    def _is_terminal(self):
        """Check if the episode should terminate."""
        if self.collision:
            return True
            
        current_distance = self._compute_distance(self.position, self.goal_position)
        if current_distance < (self.ROBOT_RADIUS + 0.05):
            return True
            
        if rospy.get_time() - self.start_time > self.timeout:
            return True
            
        return False

    def _compute_distance(self, pos1, pos2):
        """Compute Euclidean distance between two points."""
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

    def _get_goal_angle(self):
        """Calculate angle between robot heading and goal."""
        goal_angle = math.atan2(self.goal_position.y - self.position.y,
                               self.goal_position.x - self.position.x)
        heading_diff = goal_angle - self.heading
        return ((heading_diff + math.pi) % (2 * math.pi)) - math.pi

    def _downsample_scans(self, scan_data, target_size):
        """Downsample scan data with averaging."""
        if len(scan_data) == 0:
            return np.array([3.5] * target_size)
        
        step = max(len(scan_data) // target_size, 1)
        downsampled = []
        
        for i in range(0, len(scan_data), step):
            chunk = scan_data[i:i + step]
            downsampled.append(np.mean(chunk))
        
        # Ensure exactly target_size elements
        if len(downsampled) > target_size:
            downsampled = downsampled[:target_size]
        while len(downsampled) < target_size:
            downsampled.append(3.5)
            
        return np.array(downsampled)

    def generate_random_goal(self):
        """Generate a random goal position."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(self.bounds['x_min'] + self.bounds['safety_margin'],
                                self.bounds['x_max'] - self.bounds['safety_margin'])
            y = np.random.uniform(self.bounds['y_min'] + self.bounds['safety_margin'],
                                self.bounds['y_max'] - self.bounds['safety_margin'])
                                
            if self.is_position_safe(x, y):
                rospy.loginfo(f"Generated safe goal: ({x:.2f}, {y:.2f})")
                return (x, y)
                
        rospy.logwarn("Failed to generate safe goal; using fallback position")
        return (0.0, 0.0)

    def is_position_safe(self, x, y):
        """Check if a position is safe (away from obstacles and bounds)."""
        # Check bounds
        if (x < self.bounds['x_min'] + self.bounds['safety_margin'] or
            x > self.bounds['x_max'] - self.bounds['safety_margin'] or
            y < self.bounds['y_min'] + self.bounds['safety_margin'] or
            y > self.bounds['y_max'] - self.bounds['safety_margin']):
            return False
            
        # Check obstacles
        for ox, oy in self.obstacles:
            if math.sqrt((x - ox)**2 + (y - oy)**2) < (self.obstacle_radius + self.bounds['safety_margin']):
                return False
                
        return True

    def odom_callback(self, data):
        """Handle odometry updates."""
        self.position = data.pose.pose.position
        
        # Extract orientation
        orientation = data.pose.pose.orientation
        euler = euler_from_quaternion([orientation.x, orientation.y, 
                                     orientation.z, orientation.w])
        self.heading = euler[2]
        
        # Update velocities
        self.current_linear_vel = data.twist.twist.linear.x
        self.current_angular_vel = data.twist.twist.angular.z

    def scan_callback(self, data):
        """Handle laser scan updates."""
        self.scan_ranges = np.array(data.ranges)
        
        # Update collision status
        if len(self.scan_ranges) > 0:
            min_distance = min([x for x in self.scan_ranges if not np.isinf(x) and not np.isnan(x)], default=float('inf'))
            
            # Check if we're in collision state with lower threshold
            previous_collision_state = self.collision
            self.collision = min_distance < self.CRITICAL_DISTANCE  # Using critical distance for actual collisions
            
            # Publish collision event only when collision first occurs
            if self.collision and not previous_collision_state:
                self.collision_pub.publish(Bool(True))
                rospy.logwarn(f"Collision detected! Min distance: {min_distance:.2f}m")