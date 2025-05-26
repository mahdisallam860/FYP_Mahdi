cmake_minimum_required(VERSION 3.0.2)
project(dqn3)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  nodes/train_stage1.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)