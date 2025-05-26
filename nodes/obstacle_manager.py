#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist

def main():
    rospy.init_node('obstacle_manager', anonymous=True)
    
    # Publishers for the obstacles
    pub_obst1 = rospy.Publisher('/dynamic_obstacle1/cmd_vel', Twist, queue_size=1)
    pub_obst2 = rospy.Publisher('/dynamic_obstacle2/cmd_vel', Twist, queue_size=1)
    pub_obst3 = rospy.Publisher('/dynamic_obstacle3/cmd_vel', Twist, queue_size=1)
    
    # Fetch parameters or set defaults
    rate_hz = rospy.get_param('~rate', 10)  # Increased rate
    phase_increment = rospy.get_param('~phase_increment', 0.03)  # Slower phase change
    
    # Obstacle speeds (adjusted for circular motion)
    obst1_linear_speed = rospy.get_param('/dynamic_obstacle1/linear_speed', 0.2)  # Slower
    obst1_angular_speed = rospy.get_param('/dynamic_obstacle1/angular_speed', 0.2)
    
    obst2_linear_speed = rospy.get_param('/dynamic_obstacle2/linear_speed', 0.2)   # Medium
    obst2_angular_speed = rospy.get_param('/dynamic_obstacle2/angular_speed', 0.25)
    
    obst3_linear_speed = rospy.get_param('/dynamic_obstacle3/linear_speed', 0.25)  # Faster
    obst3_angular_speed = rospy.get_param('/dynamic_obstacle3/angular_speed', 0.4)
    
    rate = rospy.Rate(rate_hz)
    phase = 0.0
    
    rospy.loginfo("Obstacle manager node is running with circular motion patterns.")
    
    while not rospy.is_shutdown():
        try:
            # All obstacles now move in circles with different speeds
            # OBSTACLE 1: Large slow circle
            twist1 = Twist()
            twist1.linear.x = obst1_linear_speed
            twist1.angular.z = obst1_angular_speed
            pub_obst1.publish(twist1)
            
            # OBSTACLE 2: Medium circle
            twist2 = Twist()
            twist2.linear.x = obst2_linear_speed
            twist2.angular.z = obst2_angular_speed
            pub_obst2.publish(twist2)
            
            # OBSTACLE 3: Small fast circle
            twist3 = Twist()
            twist3.linear.x = obst3_linear_speed
            twist3.angular.z = obst3_angular_speed
            pub_obst3.publish(twist3)
            
            # Log obstacle states periodically
            rospy.loginfo_throttle(10, "Publishing circular motion commands to obstacles")
            
            phase += phase_increment
            rate.sleep()
            
        except Exception as e:
            rospy.logerr(f"Error in obstacle manager loop: {str(e)}")
            continue

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Obstacle manager node terminated.")