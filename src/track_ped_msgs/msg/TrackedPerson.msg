# TrackedPerson.msg

# Unique ID for the pedestrian
int32 id

# Distance from the camera to the pedestrian (Depth in meters)
float32 depth

# Angle to the camera perpendicular bisector (in radians)
float32 angle

# Bounding box coordinates (in pixel coordinates)
int32 bbox_upper_left_x
int32 bbox_upper_left_y
int32 bbox_bottom_right_x
int32 bbox_bottom_right_y

# Real life pose of the pedestrian relative to the camera
geometry_msgs/Pose pose

# Pedestrian velocity (linear and angular velocity)
geometry_msgs/Twist twist
