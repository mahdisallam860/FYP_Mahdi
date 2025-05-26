execute_process(COMMAND "/home/alma/catkin_ws/src/dqn3/build/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/alma/catkin_ws/src/dqn3/build/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
