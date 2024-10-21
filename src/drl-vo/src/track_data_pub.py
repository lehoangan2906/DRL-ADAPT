#!/usr/bin/python3

import time
import rclpy
import argparse
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Header
from track_func import process_frame, initialize_processing
from geometry_msgs.msg import Twist, Pose 
from sensor_msgs.msg import Image, CameraInfo
from track_ped_msgs.msg import TrackedPerson, TrackedPersons

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # YOLOv8 strongsort root directory
WEIGHTS = ROOT /  'weights'


class TrackDataPublisher(Node):
    def __init__(self, camera_type):
        super().__init__('track_data_publisher')
        self.bridge = CvBridge()
        self.camera_type = camera_type

        self.color_received = False  # To track if color image has been received
        self.depth_received = False  # To track if depth image has been received
        
        self.points = None
        self.pipeline = None           # For Realsense pipeline if using real camera
        self.color_image = None        # Variable to store color image
        self.depth_image = None        # Variable to store depth image
        self.color_camera_info = None
        self.depth_camera_info = None

        # Initialize the processing components (e.g., model, tracker)
        self.processing_components = initialize_processing()

        # Publishers for tracking data
        self.tracked_person_pub = self.create_publisher(TrackedPerson, '/track_ped', 10)
        self.tracked_persons_pub = self.create_publisher(TrackedPersons, '/track_peds', 10)

        # Unified setup based on the camera type
        if self.camera_type == 'simulated':
            self.setup_simulated_camera_subscriptions()
        elif self.camera_type == 'real':
            self.setup_real_camera_subscription()

            # Set up a timer to periodically fetch frames from the RealSense camera
            self.timer = self.create_timer(1/30.0, self.timer_callback)  # Assuming 30 Hz refresh rate
        else:
            self.get_logger().error(f"Invalid camera type: {self.camera_type}")
            exit(1)


    def setup_simulated_camera_subscriptions(self):
        """Set up ROS2 subscriptions for the simulated camera."""
        print("\n ================================== \n Getting data from the Simulated Camera. \n ================================== \n")

        self.points_sub = self.create_subscription(Image, '/intel_realsense_d415_depth/points', self.points_callback, 10)
        self.color_sub = self.create_subscription(Image, '/intel_realsense_d415_depth/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/intel_realsense_d415_depth/depth/image_raw', self.depth_callback, 10)
        self.color_camera_info_sub = self.create_subscription(CameraInfo, '/intel_realsense_d415_depth/camera_info', self.color_camera_info_callback, 10)
        self.depth_camera_info_sub = self.create_subscription(CameraInfo, '/intel_realsense_d415_depth/depth/camera_info', self.depth_camera_info_callback, 10)

    
    def setup_real_camera_subscription(self):
        """Setup physical RealSense camera pipeline."""
        print("\n ================================== \n Getting data from the Real Camera. \n ================================== \n")

        # Helper function to initialize the real physical realsense camera
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get the device product line for camera information
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        camera_device = pipeline_profile.get_device()

        found_rgb = False
        for s in camera_device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("THe RGB camera is not found!")
            exit(0)

        # Enable RGB and depth streams
        COLOR_WIDTH = 1280
        COLOR_HEIGHT = 720
        DEPTH_WIDTH = 1280
        DEPTH_HEIGHT = 720
        FPS = 30

        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)

        # Enable emitter if supported
        depth_sensor = camera_device.first_depth_sensor()

        # Check if the emitter is supported by the device
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)    # Enable emitter
            print("\n ========================= \n Emitter enabled \n ========================= \n")

            if depth_sensor.supports(rs.option.laser_power):
                # Set the laser power between 0 and max laser power
                laser_power = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.opion.laser_power, 360)
                print(f"Laser power set to {150}.")
        
        self.pipeline.start(config)


    # Callback for simulated camera
    def points_callback(self, msg):
        self.points = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width, -1)


    # Callback for simulated camera
    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.color_received = True
        self.process_if_ready()


    # Callback for simulated camera
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_received = True
        self.process_if_ready()


    def depth_camera_info_callback(self, msg):
        self.depth_camera_info = msg


    def color_camera_info_callback(self, msg):
        self.color_camera_info = msg


    # Timer callback function for real camera
    def timer_callback(self):
        if self.camera_type == "real":
            # Get frames from the RealSense camera
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Skip if no valid frames are captured
            if not depth_frame or not color_frame:
                return
            
            # Convert depth and color frames to numpy arrays
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())
            
            self.depth_received = True
            self.color_received = True

            self.process_if_ready()

        
    def process_if_ready(self):
        """Process the data only if both color and depth images are available."""
        if self.color_received and self.depth_received:
            # Prepare parameters for processing
            params = {
                'color_image': self.color_image,
                'depth_image': self.depth_image,
                'processing_components': self.processing_components,
                'camera_type': self.camera_type,
                'show_vid': True 
            }

            # Call the process_frame function from track_func.py to process the frame
            rs_dicts, pre_velocity_cal = process_frame(params)

            # Publish the processed tracking data
            self.publish_tracking_data(rs_dicts, pre_velocity_cal)

            # Reset the images after processing
            self.color_image = None
            self.depth_image = None

            # Reset the flags after processing
            self.color_received = False
            self.depth_received = False


    def publish_tracking_data(self, rs_dicts, pre_velocity_cal):
        # Create TrackedPersons message (with a list of TrackedPerson)
        tracked_persons_msg = TrackedPersons()
        tracked_persons_msg.header = Header()
        tracked_persons_msg.header.stamp = self.get_clock().now().to_msg()
        tracked_persons_msg.header.frame_id = "camera_frame"

        twist = Twist()

        # Populate the tracked_person_list into the message
        for idx, rs_dict in enumerate(rs_dicts):
            # Create a new TrackedPerson message for each object
            tracked_person_msg = TrackedPerson()
            tracked_person_msg.id = int(rs_dict['id'])
            tracked_person_msg.depth = rs_dict['depth']
            tracked_person_msg.angle = rs_dict['angle']
            tracked_person_msg.bbox_upper_left_x = rs_dict['bbox'][0]
            tracked_person_msg.bbox_upper_left_y = rs_dict['bbox'][1]
            tracked_person_msg.bbox_bottom_right_x = rs_dict['bbox'][2]
            tracked_person_msg.bbox_bottom_right_y = rs_dict['bbox'][3]

            # Convert velocity from rs_dict (if available)
            if rs_dict['velocity'] is not None:
                twist.linear.x = rs_dict['velocity'][0] # velocity in x direction
                twist.linear.z = rs_dict['velocity'][1] # velocity in z direction
            else:
                twist.linear.x = 0.0
                twist.linear.z = 0.0

            tracked_person_msg.twist = twist

            # Calculate x and z coordinates of the pedestrian in the camera frame
            real_x = rs_dict["depth"] * np.tan(rs_dict["angle"])  # Calculate x using trigonometry
            real_z = rs_dict["depth"]                             #  z is the depth directly

            # Fill in the pose with real-world coordinates
            tracked_person_msg.pose = Pose()
            tracked_person_msg.pose.position.x = real_x
            tracked_person_msg.pose.position.y = 0.0
            tracked_person_msg.pose.position.z = real_z

            # Orientation can be set to default
            tracked_person_msg.pose.orientation.x = 0.0
            tracked_person_msg.pose.orientation.y = 0.0
            tracked_person_msg.pose.orientation.z = 0.0
            tracked_person_msg.pose.orientation.w = 1.0

            # Add the tracked person message to the TrackedPersons list
            tracked_persons_msg.tracks.append(tracked_person_msg)
            
            # Publish each tracked person individually
            self.tracked_person_pub.publish(tracked_person_msg)
        
        # Publish each tracked person indvidually
        self.tracked_persons_pub.publish(tracked_persons_msg)
    

    def destroy_node(self):
        if self.camera_type == "real" and self.pipeline:
            self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # Argument parsing to decide if the camera is real or simulated
    parser = argparse.ArgumentParser(description="Track Data Publisher")
    parser.add_argument("--camera_type", type=str, defaut="simulated", choices=["real", "simulated"], help="specify the camera type (real or simulated).")
    parsed_args = parser.parse_args()

    # Create the node
    track_data_publisher = TrackDataPublisher(camera_type=parsed_args.camera_type)

    # Set up periodic timer for RealSense camera if it is a real physical camera.
    if parsed_args.camera_type == "real":
        # Call process_real_camera every 0.1 seconds (10 Hz)
        timer_period = 0.1
        track_data_publisher.create_timer(timer_period, track_data_publisher.timer_callback)

    # Spin the node
    try:
        rclpy.spin(track_data_publisher)
    finally:
        track_data_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


"""
class TrackDataPublisher(Node):
    def __init__(self):
        super().__init__('track_data_publisher')

        # Create ROS2 publishers
        self.tracked_person_pub = self.create_publisher(TrackedPerson, '/track_ped', 10)
        self.tracked_persons_pub = self.create_publisher(TrackedPersons, '/track_peds', 10)

        # Call the tracking function and get the detected objects
        self.publish_tracking_data()

    def publish_tracking_data(self):
        twist = Twist()

        # Get the list of tracked persons, rs_dicts and pre_velocity_cal from track_func's run method
        rs_dicts, pre_velocity_cal = run(
            camera_type = 'simulated',  
            source = '0',
            yolo_weights = WEIGHTS / 'yolov8m-seg.engine',  # Path to the YOLO model weights
            reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt',  # Path to ReID model weights
            tracking_method = 'bytetrack',  # Tracking method
            tracking_config = ROOT / 'track_utils' / 'Tracking_Face' / 'yolov8_tracking' / 'trackers' / 'bytetrack' / 'configs' / ('bytetrack' + '.yaml'),
            # Optional tracking configuration
            imgsz = (640, 640),  # Image size
            conf_thres = 0.25,  # Confidence threshold
            iou_thres = 0.45,  # IOU threshold
            max_det = 1000,  # Maximum number of detections per image
            device = '0',  # CUDA device (GPU), can also be 'cpu'
            show_vid = True,  # Whether to show the video results in a window
            save_txt = False,  # Whether to save results in a text file
            save_conf = False,  # Whether to save confidence scores
            save_crop = False,  # Whether to save cropped bounding boxes
            save_trajectories = False,  # Whether to save the trajectories
            save_vid = False,  # Whether to save video output
            nosave = False,  # If set to True, will not save images/videos
            classes = None,  # Filter by class (0 = person)
            agnostic_nms = False,  # Class-agnostic non-max suppression
            augment = False,  # Whether to use augmented inference
            visualize = False,  # Whether to visualize features
            update = False,  # Update all models
            project = ROOT / 'runs' / 'track',  # Directory to save results
            name = 'exp',  # Experiment name
            exist_ok = False,  # Whether it's okay if the save directory exists
            line_thickness = 2,  # Line thickness for bounding boxes
            hide_labels = False,  # Whether to hide labels
            hide_conf = False,  # Whether to hide confidence scores
            hide_class = False,  # Whether to hide IDs
            half = False,  # Use FP16 half-precision inference
            dnn = False,  # Use OpenCV DNN for ONNX inference
            vid_stride = 1,  # Video frame-rate stride
            retina_masks = False,  # Whether to use retina masks
            )
        
        # Create TrackedPersons message (with a list of TrackedPerson)
        tracked_persons_msg = TrackedPersons()
        tracked_persons_msg.header = Header()
        tracked_persons_msg.header.stamp = self.get_clock().now().to_msg()
        tracked_persons_msg.header.frame_id = "camera_frame"

        # Populate the tracked_person_list into the message
        for idx, rs_dict in enumerate(rs_dicts):
            # Create a new TrackedPerson message for each object
            tracked_person_msg = TrackedPerson()
            tracked_person_msg.id = int(rs_dict['id'])
            tracked_person_msg.depth = rs_dict['depth']
            tracked_person_msg.angle = rs_dict['angle']
            tracked_person_msg.bbox_upper_left_x = rs_dict['bbox'][0]
            tracked_person_msg.bbox_upper_left_y = rs_dict['bbox'][1]
            tracked_person_msg.bbox_bottom_right_x = rs_dict['bbox'][2]
            tracked_person_msg.bbox_bottom_right_y = rs_dict['bbox'][3]
            
            # Convert velocity from rs_dict (if available)
            if rs_dict['velocity'] is not None:
                twist.linear.x = rs_dict['velocity'][0] # velocity in x direction
                twist.linear.z = rs_dict['velocity'][1] # velocity in z direction
            else:
                twist.linear.x = 0.0
                twist.linear.z = 0.0

            tracked_person_msg.twist = twist 

            # Calculate x and z coordinates of the pedestrian in the camera frame
            real_x = rs_dict["depth"] * np.tan(rs_dict["angle"])  # Calculate x using trigonometry
            real_z = rs_dict["depth"]                             # z is the depth directly

            # Fill in the pose with real-world coordinates
            tracked_person_msg.pose = Pose()
            tracked_person_msg.pose.position.x = real_x
            tracked_person_msg.pose.position.y = 0.0
            tracked_person_msg.pose.position.z = real_z

            # Orientation can be set to default
            tracked_person_msg.pose.orientation.x = 0.0
            tracked_person_msg.pose.orientation.y = 0.0
            tracked_person_msg.pose.orientation.z = 0.0
            tracked_person_msg.pose.orientation.w = 1.0

            # Add the tracked person message to the TrackedPersons list
            tracked_persons_msg.tracks.append(tracked_person_msg)

            # Publish each tracked person individually
            self.tracked_person_pub.publish(tracked_person_msg)
        
        # Publish the TrackedPersons message (all detected objects in each frame)
        self.tracked_persons_pub.publish(tracked_persons_msg)
        
def main(args=None):
    rclpy.init(args=args)
    track_data_publisher = TrackDataPublisher()
    rclpy.spin(track_data_publisher)
    track_data_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""