cmake_minimum_required(VERSION 3.20)
project(imgpcd_project_real_time)
set(CMAKE_CXX_STANDARD 17)  # 设置为 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 强制要求该标准
set(CMAKE_CXX_EXTENSIONS OFF)  # 不使用编译器特定的扩展

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  rosbag
  cv_bridge
  image_transport
)

find_package(OpenCV 3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros  rosbag cv_bridge image_transport
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(imgpcd_project_real_time src/pcd_image_to_bag.cpp)

target_link_libraries(imgpcd_project_real_time
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
