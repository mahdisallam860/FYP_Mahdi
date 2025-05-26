#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist

class AdjustedObstacleManager:
    def __init__(self):
        rospy.init_node('adjusted_obstacle_manager', anonymous=True)
        
        # Publishers for each dynamic obstacle
        self.pubs = {
            1: rospy.Publisher('/dynamic_obstacle1/cmd_vel', Twist, queue_size=1),
            2: rospy.Publisher('/dynamic_obstacle2/cmd_vel', Twist, queue_size=1),
            3: rospy.Publisher('/dynamic_obstacle3/cmd_vel', Twist, queue_size=1)
        }
        
        # Movement parameters
        self.params = {
            # Obstacle 1 (Red) - circle, clockwise
            'circle1': {
                'center_x': 1.5,
                'center_y': 1.0,
                'radius': 0.3,
                'linear_speed': 0.12,
                'angular_speed': 0.4
            },
            # Obstacle 2 (Green) - circle, counterclockwise
            'circle2': {
                'center_x': -1.5,
                'center_y': -1.0,
                'radius': 0.3,
                'linear_speed': 0.12,
                'angular_speed': 0.4
            },
            # Obstacle 3 (Blue) - patrol
            'patrol': {
                'y_pos': 1.5,
                'min_x': -1.2,
                'max_x': 1.2,
                'speed': 0.1,
                'turn_buffer': 0.3
            }
        }
        
        self.start_time = rospy.Time.now().to_sec()
        
        # Patrol logic
        self.patrol_direction = 1  # 1 = moving right, -1 = moving left
        self.last_turn_time = self.start_time
        
        self.rate = rospy.Rate(10)  # 10 Hz

    def move_in_circle(self, params, t, clockwise=True):
        """
        Calculate a twist that results in roughly circular motion.
        If desired, you can do a boundary check. For demonstration,
        we remove the strict boundary check that sets velocity to 0.
        """
        twist = Twist()
        twist.linear.x = params['linear_speed']
        
        # Angular speed negative => clockwise; positive => CCW
        twist.angular.z = -params['angular_speed'] if clockwise else params['angular_speed']
        return twist

    def patrol_movement(self, current_time):
        """
        Moves the obstacle back and forth along X at a fixed Y.
        """
        p = self.params['patrol']
        twist = Twist()
        
        time_in_direction = current_time - self.last_turn_time
        # Estimate where we'd be in X
        estimated_x = (
            (p['min_x'] if self.patrol_direction == 1 else p['max_x'])
            + self.patrol_direction * p['speed'] * time_in_direction
        )
        
        # If we are close to a boundary, flip direction
        if (self.patrol_direction == 1 and estimated_x >= p['max_x'] - p['turn_buffer']) or \
           (self.patrol_direction == -1 and estimated_x <= p['min_x'] + p['turn_buffer']):
            self.patrol_direction *= -1
            self.last_turn_time = current_time
            return Twist()  # brief pause at the turn
        
        # Otherwise, continue in the same direction
        twist.linear.x = p['speed'] * self.patrol_direction
        return twist

    def run(self):
        """
        Main loop: publish velocity commands to each obstacle.
        """
        try:
            while not rospy.is_shutdown():
                current_time = rospy.Time.now().to_sec()
                time_delta = current_time - self.start_time
                
                # Obstacle 1 (Red) - circle, clockwise
                twist1 = self.move_in_circle(self.params['circle1'], time_delta, clockwise=True)
                self.pubs[1].publish(twist1)
                
                # Obstacle 2 (Green) - circle, counterclockwise
                twist2 = self.move_in_circle(self.params['circle2'], time_delta, clockwise=False)
                self.pubs[2].publish(twist2)
                
                # Obstacle 3 (Blue) - patrol
                twist3 = self.patrol_movement(current_time)
                self.pubs[3].publish(twist3)
                
                self.rate.sleep()
        
        except rospy.ROSInterruptException:
            pass
        except Exception as e:
            rospy.logerr(f"Error in obstacle manager: {str(e)}")
            # Emergency stop
            zero_twist = Twist()
            for pub in self.pubs.values():
                pub.publish(zero_twist)

def main():
    try:
        manager = AdjustedObstacleManager()
        rospy.sleep(1)  # wait for everything to initialize
        manager.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
