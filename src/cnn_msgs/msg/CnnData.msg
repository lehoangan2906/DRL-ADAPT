# This file defines a custom message format for the cnn_msgs package.
# It aggregates several types of sensor data.

float32[] ped_pos_map         # pedestrian's position costmap in Cartesian coordinate: 2 channels, 20m * 20m, 80 * 80 grid
float32[] scan                # 240 range data from the laser scan
float32[] scan_all            # 450 range data from the laser scan
float32[] image_gray          # image data from the intel realsense camera
float32[] depth               # depth image data from the intel realsense camera
float32[] goal_cart           # current goal in robot frame
float32[] goal_final_polar    # final goal in robot frame
float32[] vel                 # current velocity in robot frame
