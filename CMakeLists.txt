cmake_minimum_required(VERSION 3.8)
project(tf_tool)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
## addon
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)

# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

# Add executable
add_executable(frame_connector 
  src/frame_connector/frame_connector_node.cpp 
  src/frame_connector/frame_connector_component.cpp
)
ament_target_dependencies(frame_connector
  rclcpp
  rcl_interfaces
  tf2
  tf2_ros
  tf2_msgs
)

# target_link_libraries(frame_connector ${PCL_LIBRARIES})

# Install Cpp executables
install(TARGETS
  frame_connector 
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()