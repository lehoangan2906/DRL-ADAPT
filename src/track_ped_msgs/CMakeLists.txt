# Suppress CMake policy warnings related to Python modules
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Suppress all developer warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress developer warnings")

cmake_minimum_required(VERSION 3.8)
project(track_ped_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate ROS2 interfaces (messages, services, actions)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedPerson.msg"
  "msg/TrackedPersons.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(std_msgs geometry_msgs)

# Install ROS interfaces
install(DIRECTORY msg/
  DESTINATION share/${PROJECT_NAME}/msg
)

# Include directories
include_directories(include)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
