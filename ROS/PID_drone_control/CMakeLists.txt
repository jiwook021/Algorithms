cmake_minimum_required(VERSION 3.0.2)
project(drone_control)

## Find catkin and required packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  gazebo_msgs
  rospy
  sensor_msgs
  laser_geometry
  tf2_ros
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs gazebo_msgs rospy sensor_msgs laser_geometry tf2_ros
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare and link the C++ executable
add_executable(drone_control_node src/drone_control_node.cpp)
target_link_libraries(drone_control_node ${catkin_LIBRARIES})

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/laser_to_pointcloud.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install other files (optional)
install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  PATTERN "*.py" EXCLUDE
)