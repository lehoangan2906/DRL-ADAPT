cmake_minimum_required(VERSION 3.8)
project(drl-vo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Ensure Python Interpreter is available
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Install Python modules
install(
	DIRECTORY src/track_utils/                  # Copy the entire track_utils folder to the installation directory of the package so that it can be used by other scripts within drl-vo
  DESTINATION lib/${PROJECT_NAME}/track_utils
)

# Install model weights
install(DIRECTORY src/weights/
  DESTINATION lib/${PROJECT_NAME}/weights
)

# Install Python nodes
install(PROGRAMS
  src/track_func.py
  src/track_data_pub.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
# install (DIRECTORY launch/
#   DESTINATION share/${PROJECT_NAME}/launch
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
