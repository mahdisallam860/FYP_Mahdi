#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
import random

class DynamicGoalPublisher:
    def __init__(self):
        """
        Initialize the dynamic goal publisher.
        """
        rospy.init_node("dynamic_goal_publisher", anonymous=True)

        # Define the topic for publishing dynamic goals
        self.goal_pub = rospy.Publisher("/dynamic_goal", Point, queue_size=10)

        # Goal generation bounds
        self.bounds = {
            "x_min": -3.0,
            "x_max": 3.0,
            "y_min": -3.0,
            "y_max": 3.0,
            "safety_margin": 0.5  # Avoid placing goals too close to obstacles
        }

        # Goal update rate
        self.rate = rospy.Rate(1)  # 1 goal per second

    def is_goal_safe(self, x, y):
        """
        Check if a generated goal is within safe bounds.

        Args:
            x (float): X-coordinate of the goal.
            y (float): Y-coordinate of the goal.

        Returns:
            bool: True if the goal is safe, False otherwise.
        """
        return (self.bounds["x_min"] + self.bounds["safety_margin"] <= x <= self.bounds["x_max"] - self.bounds["safety_margin"] and
                self.bounds["y_min"] + self.bounds["safety_margin"] <= y <= self.bounds["y_max"] - self.bounds["safety_margin"])

    def generate_random_goal(self):
        """
        Generate a random goal position within safe bounds.

        Returns:
            Point: Randomly generated goal position.
        """
        max_attempts = 100  # Max attempts to find a valid goal
        for _ in range(max_attempts):
            x = random.uniform(self.bounds["x_min"], self.bounds["x_max"])
            y = random.uniform(self.bounds["y_min"], self.bounds["y_max"])

            if self.is_goal_safe(x, y):
                return Point(x=x, y=y, z=0.0)

        # Fallback to the center if no valid position is found
        rospy.logwarn("Unable to generate a valid goal. Using fallback position (0, 0).")
        return Point(x=0.0, y=0.0, z=0.0)

    def publish_goals(self):
        """
        Continuously publish random goals at a set rate.
        """
        while not rospy.is_shutdown():
            goal = self.generate_random_goal()
            rospy.loginfo(f"Publishing new dynamic goal: x={goal.x:.2f}, y={goal.y:.2f}")
            self.goal_pub.publish(goal)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        dynamic_goal_publisher = DynamicGoalPublisher()
        dynamic_goal_publisher.publish_goals()
    except rospy.ROSInterruptException:
        rospy.loginfo("Dynamic Goal Publisher shutting down.")
