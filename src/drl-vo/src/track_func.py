#!/usr/bin/python3

import os 
import sys
import cv2
import time
import math
import rclpy
import torch 
import pickle
import logging
import argparse
import platform 
import numpy as np
import pandas as pd
from pathlib import Path
import pyrealsense2 as rs
import track_utils.insightface
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from track_utils.insightface.app import FaceAnalysis
from track_utils.Tracking_Face.find_distance import *
from track_utils.insightface.data import get_image as ins_get_image
from track_utils.Tracking_Face.yolov8_tracking.utils.milvus_tool import *
from track_utils.Tracking_Face.yolov8_tracking.utils.process_box import *
from track_utils.Tracking_Face.yolov8_tracking.trackers.multi_tracker_zoo import create_tracker
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.nn.autobackend import AutoBackend
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.data import load_inference_source
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.utils.files import increment_path
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.utils.torch_utils import select_device
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.data.augment import LetterBox, classify_transforms
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.utils.plotting import Annotator, colors, save_one_box
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from track_utils.Tracking_Face.yolov8_tracking.ultralytics.ultralytics.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # YOLOv8 strongsort root directory
WEIGHTS = ROOT /  'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'ultralytics') not in sys.path:
    sys.path.append(str(ROOT / 'ultralytics'))
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

sys.path.append(str(FILE.parents[2]))  # add strong_sort ROOT to PATH
sys.path.append(str(FILE.parents[1]))

app = FaceAnalysis(providers = [('CUDAExecutionProvider')])
app.prepare(ctx_id = 0, det_size = (640, 640))


# Initialize the CvBridge
bridge = CvBridge()


# Callback to handle the depth image from the simulated camera
depth_image = None
color_image = None
depth_camera_info = None
color_camera_info = None
points = None


def depth_callback(msg):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    print("Depth image received and processed:", depth_image.shape)

def color_callback(msg):
    global color_image
    color_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    print("Color image received and processed:", color_image.shape)

def depth_camera_info_callback(msg):
    global depth_camera_info
    depth_camera_info = msg

def color_camera_info_callback(msg):
    global color_camera_info
    color_camera_info = msg

def points_callback(msg):
    global points
    points = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width, -1)


def find_x_extremes(masks, bboxes, img1_shape, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    
    results = []
    
    for mask, bbox in zip(masks, bboxes):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate the y-coordinate at the center of the bounding box
        center_y = int((y2-y1)/2 + y1)
        # Get the row of the mask at center_y
        mask_row = mask[center_y, x1:x2]

        
        if len(mask_row) > 0:
            # Find the leftmost (smallest x) and rightmost (biggest x) non-zero pixels
            non_zero_indices = np.nonzero(mask_row)[:,0]
            if len(non_zero_indices) == 0:
                results.append(None)
                continue
            smallest_x = x1 + non_zero_indices[0]
            biggest_x = x1 + non_zero_indices[-1]

            # Scale back to original image coordinates
            smallest_x_original = int(smallest_x / gain)
            biggest_x_original = int(biggest_x / gain)
            center_x_original = int((smallest_x_original + biggest_x_original) / 2)
            center_y_original = int(center_y / gain)
            
            results.append([center_x_original, center_y_original])

        else:
            results.append(None)

    return results


def init_Milvus():
    client = Milvus(uri="tcp://192.168.44.27:19530")
    # client = Milvus(uri='tcp://172.21.100.254:19530')
    client.list_collections()

    client.drop_collection("data")

    # Create collection demo_collection if it dosen't exist.
    collection_name = "data"

    status, ok = client.has_collection(collection_name)
    if not ok:
        param = {
            "collection_name": collection_name,
            "dimension": 512,
            "metric_type": MetricType.IP,  # optional
        }
    client.create_collection(param)

    _, collection = client.get_collection_info(collection_name)
    # print("Collection: ", collection)
    status, result = client.count_entities(collection_name)
    # print("Result: ", result)

    pkl_path = "/home/anlh/DRL_Velocity_Obstacles/src/drl_vo/src/track_utils/sample_huy.pkl"
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    names = [sample["name"] for sample in samples]
    names = set(names)

    # embs = np.array([np.array(i['emb']) for i in samples])
    embs = np.array([sample["emb"] for sample in samples])
    # print(len(embs))
    # print(len(samples))
    # %% convert list to numpy array
    knownEmbedding = embs
    print(knownEmbedding.shape)

    # knownNamesId = [i['name'] for i in samples]
    knownNamesId = [sample["name"] for sample in samples]
    # print("len Ids: ", len(knownNamesId))
    # print("len Embs: ", knownEmbedding.shape[0])
    # insert true data into true_collection_
    status, ids = client.insert(
        collection_name=collection_name,
        records=knownEmbedding,
        ids=list(range(len(knownNamesId))),
    )
    if not status.OK():
        print("Insert failed: {}".format(status))

    # print(len(ids))
    # print("Status: ", status)
    # %%
    
    client.flush([collection_name])
    # Get demo_collection row count
    status, result = client.count_entities(collection_name)
    # print(result)
    # print(status)
    ivf_param = {"nlist": 1024}
    status = client.create_index(collection_name, IndexType.FLAT, ivf_param)

    # describe index, get information of index
    status, index = client.get_index_info(collection_name)

    return client, collection_name, samples


def normalize(emb):
    emb = np.squeeze(emb)
    norm = np.linalg.norm(emb)
    return emb/norm


def pre_transform(im, imgsz, model):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """

    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(imgsz, auto=same_shapes and model.pt, stride=model.stride)
    
    return [letterbox(image=x) for x in im]


def process_faces(img):
    face_frame = img 
    faces = app.get(face_frame)
    faces_data = pd.DataFrame([face.bbox for face in faces], columns = ['x1', 'y1', 'x2', 'y2'])
    faces_data['embedding'] = [normalize(face.embedding) for face in faces]

    return faces_data


def process_detection(dt, im0s, paths, model, device, imgsz, save_dir, is_seg, conf_thres, iou_thres, max_det, augment, visualize, classes, agnostic_nms):
    with dt[0]:
        not_tensor = not isinstance(im0s, torch.Tensor)
        if not_tensor:
            im = np.stack(pre_transform(im0s, imgsz, model))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

            im = im.to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if not_tensor:
                im /= 255  # 0 - 255 to 0.0 - 1.0

    # Inference
    with dt[1]:
        visualize = (
                    increment_path(save_dir / Path(paths).stem, mkdir=True)
                    if visualize #and (not source_type.tensor)
                    else False
                    )
        preds = model(im, augment=augment, visualize=visualize, embed=None)  
    # Apply NMS
    with dt[2]:
        if is_seg:
            masks = []
            # p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            p = ops.non_max_suppression(
                                        preds[0],
                                        conf_thres,
                                        iou_thres,
                                        agnostic=agnostic_nms,
                                        max_det=max_det,
                                        nc=len(model.names),
                                        classes=classes,
                                    )
            proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
            return p, im, masks, proto
        else:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            return p, im, None, None
        

# def process_tracking(p, dt, masks, proto, im0s, paths, curr_frames, prev_frames, tracker_list, outputs, faces_data, depth_frame, display_center, f_pixel, known_id, collection_name, samples, client, save_crop, line_thickness, names, im, DEPTH_WIDTH, DEPTH_HEIGHT, save_vid, show_vid, hide_labels, hide_conf, hide_class, windows, seen, pre_velocities_cal, DELTA_T):
def process_tracking(p, dt, masks, proto, im0s, paths, curr_frames, prev_frames, tracker_list, outputs, faces_data, depth_frame, display_center, f_pixel, known_id, save_crop, line_thickness, names, im, DEPTH_WIDTH, DEPTH_HEIGHT, save_vid, show_vid, hide_labels, hide_conf, hide_class, windows, seen, pre_velocities_cal, DELTA_T):
    rs_dicts = []
    curr_velocities = []
    # Process detections
    for i, det in enumerate(p):  # detections per image       
        seen += 1
        p, im0, _ = paths[i], im0s[i].copy(), 0#dataset.count
        p = Path(p)  # to Path
        curr_frames[i] = im0

        # s += '%gx%g ' % im.shape[2:]  # print string
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
                
        if det is not None and len(det):

            print("\n\n#################################################\n\n")
            print("detection is not None")
            print("\n\n#################################################\n\n")

            mask = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            x_extremes = find_x_extremes(mask, det[:, :4], im.shape[2:], im0.shape)
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class

            # pass detections to strongsort
            with dt[3]:
                outputs[i] = tracker_list[i].update(det.cpu(), im0, x_extremes)
            # draw boxes for visualization
            if len(outputs[i]) > 0:

                human_data = pd.DataFrame([output for output in outputs[i]], columns=['x1', 'y1', 'x2', 'y2', 'ID', 'cls', 'conf', 'center_point'])
                human_data['embedding'] = None  # Initialize the new column
                if not faces_data.empty and len(human_data) == len(faces_data):
                    # Calculate IoU
                    faces_boxes = faces_data[['x1', 'y1', 'x2', 'y2']].values
                    human_boxes = human_data[['x1', 'y1', 'x2', 'y2']].values
                    iou_matrix, center_dist_matrix = calculate_iou_and_center_distance(faces_boxes, human_boxes)
                    # # Find the closest bounding box in group B for each box in A
                    # closest_B_indices = np.argmax(iou_matrix, axis=1)
                    # highest_ious = np.max(iou_matrix, axis=1)

                    # Normalize center distances
                    max_dist = np.max(center_dist_matrix)
                    normalized_center_dist = 1 - (center_dist_matrix / max_dist)  # Invert so larger value is better

                    # Combine IoU and normalized center distance
                    weight_iou = 0.6  # Adjust these weights as needed
                    weight_center = 0.4
                    combined_score = weight_iou * iou_matrix + weight_center * normalized_center_dist
                    # Find the closest bounding box in group B for each box in A
                    closest_B_indices = np.argmax(combined_score, axis=1)
                    highest_scores = np.max(combined_score, axis=1)
                    # Add the array from A to B
                    
                    for l, idx in enumerate(closest_B_indices):
                        human_data.at[idx, 'embedding'] = faces_data.at[l, 'embedding']

                for j, (output) in human_data.iterrows():
                    bbox = output[['x1', 'y1', 'x2', 'y2']].tolist()#[0:4]
                    id = output['ID']#[4] # find_id_matched(output[4],id_dic)  
                    cls = output['cls']#[5]
                    conf = output['conf']#[6]

                    # Uncomment this to enable face recognition and milvus search
                    """ if output['embedding'] is not None :
                        query_vectors = np.array([output['embedding']])  

                        top_k = 1
                        hyper_p = math.floor(top_k/2)

                        param = {
                            'collection_name': collection_name,
                            'query_records': query_vectors,
                            'top_k': top_k,
                            'params' : {'nprobe': 16 }
                        }

                        status, results = client.search(**param)

                        th = 0.6
                        pred_name, score, ref = top_k_pred(0,top_k,samples,results) 
                        if score >= th:
                            pass """

                    if output['center_point'] is not None:
                        center = output['center_point']
                        center[0] = max(0, min(center[0], DEPTH_WIDTH-1))
                        center[1] = max(0, min(center[1], DEPTH_HEIGHT-1))
                        d = abs(center[0] - display_center)
                        angle = np.arctan(d/f_pixel) #* 180 / np.pi
                        angle = angle if center[0] > display_center else -angle

                        # Handle depth extraction based on the type of depth frame
                        if isinstance(depth_frame, np.ndarray):
                            # For simulated camera (NumPy array), extract depth using array indexing
                            depth = depth_frame[int(center[1]), int(center[0])]
                        else:
                            # For real RealSense camera, use get_distance() method
                            depth = depth_frame.get_distance(int(center[0]), int(center[1]))
                        real_x_position = depth * np.tan(angle)
                    else:
                        angle = depth = real_x_position = 0
                    # print(output)

                    """
                    ID of box: id from id
                    Box coordinate: (x1, y1, x2, y2) from bbox
                    Depth: depth of bbox from  depth
                    Angle: angle of bbox from center of the screen from a
                    Velocity: velocity of bbox from center of the screen from v        
                    """

                    if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                        c = int(cls)  # integer class
                        id = str(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                            (
                                                                f'{id} {conf:.2f}' if hide_class else f'{id} {conf:.2f} Depth: {depth:.2f}m Angle: {angle:.2f}'))  # (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        color = colors(c, True)
                        annotator.box_label(bbox, label, color=color)
                    if not pre_velocities_cal.empty \
                        and not pre_velocities_cal.loc[pre_velocities_cal['ID'] == id].empty:
                        
                        pre_velocity = pre_velocities_cal.loc[pre_velocities_cal['ID'] == id]
                        velocity_x = (pre_velocity['Real_x_position'].values[0] - real_x_position ) / DELTA_T
                        velocity_y = (pre_velocity['Depth'].values[0] - depth) / DELTA_T
                        rs_dict = {"id": id, "bbox": bbox, "depth": depth, "angle": angle, "velocity": [velocity_x, velocity_y]}
                    else:
                        rs_dict = {"id": id, "bbox": bbox, "depth": depth, "angle": angle, "velocity": None}
                    
                    rs_dicts.append(rs_dict)
                    curr_velocities.append([id, real_x_position, depth])

            
            curr_velocities_cal = pd.DataFrame(curr_velocities, columns=['ID', 'Real_x_position', 'Depth'])
        else:
            curr_velocities_cal = pd.DataFrame(columns=['ID', 'Real_x_position', 'Depth'])
            pass
            # tracker_list[i].tracker.pred_n_update_all_tracks()
        prev_frames[i] = curr_frames[i]
        # Stream results
        im0 = annotator.result()
        if show_vid:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            # depth_image = np.asanyarray(depth_frame.get_data())
            # depth_image = depth_image.astype(np.uint8)
            # cv2.imshow('Depth', depth_image)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        return rs_dicts, curr_velocities_cal
    

@torch.no_grad()
def run(camera_type = 'simulated',
        source = '0', 
        yolo_weights = WEIGHTS / 'yolov8m-seg.engine',  
        reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt',   
        tracking_method='bytetrack',    

        tracking_config = None,    
        imgsz = (640, 640),                 # inference size (height, width)    
        conf_thres = 0.25,                  # confidence threshold
        iou_thres = 0.45,                   # NMS IOU threshold

        max_det = 1000,                     # maximum detections per image
        device = '0',                      # cuda device 
        show_vid = True,             
        save_txt = False,                  # save results to *.txt

        save_conf = False,                  # save confidences in --save-txt labels
        save_crop = False,                 # save cropped prediction boxes
        save_trajectories = False,         # save trajectories for each track
        save_vid = False,             

        nosave = False,                    # do not save images/ videos "192.168.44.250"
        classes = None,                     # filter by class, e.g., [1] for humans
        agnostic_nms = False,              # class-agnostic NMS
        augment = False,                   # augmented inference

        visualize = False,                  # visualize features
        update = False,                     # update all
        project = ROOT / 'runs' / 'track',  # save results to project/name
        name = 'exp',                       # save results to project/name

        exist_ok = False,                   # existing project/name ok, do not increment
        line_thickness = 2,                 # bounding box thickness (pixels)
        hide_labels = False,                # hide labels
        hide_conf = False,                  # hide confidences

        hide_class = False,                 # hide IDs
        half = False,                       # use FP16 half-precision inference
        dnn = False,                        # use OpenCV DNN for ONNX inference
        vid_stride = 1,                     # video frame-rate stride

        retina_masks = False,       
        ):
    
    global depth_image, color_image, display_center
    
    # Prepare the input source information (camera, file, URL, etc)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt') # Save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)   # Download if source is a URL

    print("\n##############################################################################\n")

    if camera_type == 'real':
        # Configure depth and color streams from RealSense Camera
        pipeline = rs.pipeline()
        config = rs.config()

        # Get the device product line for camera information
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        camera_device = pipeline_profile.get_device()    

        found_rgb = False
        for s in camera_device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The RGB camera is not found!")
            exit(0)

    
        # Enable RGB and depth streams
        COLOR_WIDTH = 1280
        COLOR_HEIGHT = 720
        DEPTH_WIDTH = 1280
        DEPTH_HEIGHT = 720
        FPS = 30
        DELTA_T = 1 / FPS
        f_pixel = (COLOR_WIDTH * 0.5) / np.tan(69 * 0.5 * np.pi / 180)
        display_center = COLOR_WIDTH // 2

        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)

        # Enable emitter if supported (optional)
        depth_sensor = camera_device.first_depth_sensor()

        # Check if the emitter is supported by the device
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)  # Enable emitter
            print("Emitter enabled.")

            if depth_sensor.supports(rs.option.laser_power):
                # Set the laser power between 0 and max laser power
                laser_power = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, 360)  # Set laser power to max
                print(f"Laser power set to {150}.")
        pipeline.start(config)

    elif camera_type == 'simulated':
        # rclpy.init()
        print("\nGetting data from the simulated camera.\n")

        # ROS2 node and subscriptions for simulated camera
        node = rclpy.create_node('camera_listener')

        depth_image_sub = node.create_subscription(Image, '/intel_realsense_d415_depth/depth/image_raw', depth_callback, 10)
        color_image_sub = node.create_subscription(Image, '/intel_realsense_d415_depth/image_raw', color_callback, 10)
        depth_camera_info_sub = node.create_subscription(CameraInfo, '/intel_realsense_d415_depth/depth/camera_info', depth_camera_info_callback, 10)
        color_camera_info_sub = node.create_subscription(CameraInfo, '/intel_realsense_d415_depth/camera_info', color_camera_info_callback, 10)
        points_sub = node.create_subscription(Image, '/intel_realsense_d415_depth/points', points_callback, 10)

        # Assuming color image has the same width/height for the simulated camera
        COLOR_WIDTH = 1280
        COLOR_HEIGHT = 720
        DEPTH_WIDTH = 1280
        DEPTH_HEIGHT = 720
        FPS = 30
        DELTA_T = 1 / FPS
        f_pixel = (COLOR_WIDTH * 0.5) / np.tan(69 * 0.5 * np.pi / 180)
        display_center = COLOR_WIDTH // 2  # Set display_center here for simulated camera


    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    # Cluster 3: Model initialization
    bs = 1  # batch size
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)  # Check if it's segmentation
    model = AutoBackend(weights=yolo_weights,
                        device=device,
                        dnn=dnn, 
                        fp16=half,
                        batch=bs,  # Set batch size
                        fuse=True,
                        verbose=True)
    model.eval()  # Set model to evaluation mode
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # Check image size

    # Warmup the model
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # Warmup with dummy data

    # Cluster 4: Tracker initialization
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker,)
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    
    outputs = [None] * bs

    # Cluster 5: Main processing loop
    seen, windows, dt = 0, [], (Profile(device=device), 
                                Profile(device=device), 
                                Profile(device=device), 
                                Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs

    # Uncomment this to enable face detection
    # client, collection_name, samples = init_Milvus()

    known_id = {}
    pre_velocity_cal = pd.DataFrame(columns=['ID', 'Real_x_position', 'Depth'])

    try:
        while True:

            # Get frames based on the camera type
            if camera_type == 'real':
                # Get frames from the RealSense camera
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                # Skip if no valid frames are captured
                if not depth_frame or not color_frame:
                    continue

                # Convert depth and color frames to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
            
            elif camera_type == 'simulated':
                # Spin the ROS2 node to collect simulated data
                rclpy.spin_once(node)

                # ensure that depth and color images are available before processing
                if depth_image is None or color_image is None: 
                    # Sleep for a short time to avoid overloading the CPU
                    time.sleep(0.01)
                    continue

                depth_frame = depth_image

            # Prepare the color image for YOLO object detection
            im0s = [color_image]  # List of color images (batch size = 1)
            paths = source
            faces_data = process_faces(im0s[0])  # Perform face detection

            # YOLO detection
            p, im, masks, proto = process_detection(dt, im0s, paths, model, device, imgsz, 
                                                    save_dir, is_seg, conf_thres, iou_thres, 
                                                    max_det, augment, visualize, classes, agnostic_nms)

            # Tracking
            """ rs_dicts, pre_velocity_cal = process_tracking(p, dt, masks, proto, im0s, paths, curr_frames, 
                                                        prev_frames, tracker_list, outputs, faces_data, 
                                                        depth_frame, display_center, f_pixel, known_id, 
                                                        collection_name, samples, client, save_crop, 
                                                        line_thickness, names, im, DEPTH_WIDTH, 
                                                        DEPTH_HEIGHT, save_vid, show_vid, hide_labels, 
                                                        hide_conf, hide_class, windows, seen, 
                                                        pre_velocity_cal, DELTA_T) """
            
            rs_dicts, pre_velocity_cal = process_tracking(p, dt, masks, proto, im0s, paths, curr_frames, 
                                                        prev_frames, tracker_list, outputs, faces_data, 
                                                        depth_frame, display_center, f_pixel, known_id, 
                                                        save_crop,line_thickness, names, im, DEPTH_WIDTH, 
                                                        DEPTH_HEIGHT, save_vid, show_vid, hide_labels, 
                                                        hide_conf, hide_class, windows, seen, 
                                                        pre_velocity_cal, DELTA_T)
                        
            # Reset the images after processing
            depth_image = None
            color_image = None
            
            # return rs_dicts, pre_velocity_cal

    finally:
        if camera_type == 'real':
            pipeline.stop()  # Stop the camera
        if camera_type == 'simulated':
            pass
            #rclpy.shutdown()
        cv2.destroyAllWindows()  # Close all OpenCV windows


def main():
    # Define the arguments directly for the run() function
    camera_type = 'simulated'  # Can be 'real' or 'simulated'
    source = '0'  # Define the source for the camera or video file
    yolo_weights = WEIGHTS / 'yolov8m-seg.engine'  # Path to the YOLO model weights
    reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'  # Path to ReID model weights
    tracking_method = 'bytetrack'  # Tracking method
    tracking_config = ROOT / 'track_utils' / 'Tracking_Face' / 'yolov8_tracking' / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
    # Optional tracking configuration
    imgsz = (640, 640)  # Image size
    conf_thres = 0.25  # Confidence threshold
    iou_thres = 0.45  # IOU threshold
    max_det = 1000  # Maximum number of detections per image
    device = '0'  # CUDA device (GPU), can also be 'cpu'
    show_vid = True  # Whether to show the video results in a window
    save_txt = False  # Whether to save results in a text file
    save_conf = False  # Whether to save confidence scores
    save_crop = False  # Whether to save cropped bounding boxes
    save_trajectories = False  # Whether to save the trajectories
    save_vid = False  # Whether to save video output
    nosave = False  # If set to True, will not save images/videos
    classes = None  # Filter by class (0 = person)
    agnostic_nms = False  # Class-agnostic non-max suppression
    augment = False  # Whether to use augmented inference
    visualize = False  # Whether to visualize features
    update = False  # Update all models
    project = ROOT / 'runs' / 'track'  # Directory to save results
    name = 'exp'  # Experiment name
    exist_ok = False  # Whether it's okay if the save directory exists
    line_thickness = 2  # Line thickness for bounding boxes
    hide_labels = False  # Whether to hide labels
    hide_conf = False  # Whether to hide confidence scores
    hide_class = False  # Whether to hide IDs
    half = False  # Use FP16 half-precision inference
    dnn = False  # Use OpenCV DNN for ONNX inference
    vid_stride = 1  # Video frame-rate stride
    retina_masks = False  # Whether to use retina masks

    # Call the run() function with the defined arguments
    run(
        camera_type=camera_type,
        source=source,
        yolo_weights=yolo_weights,
        reid_weights=reid_weights,
        tracking_method=tracking_method,
        tracking_config=tracking_config,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        device=device,
        show_vid=show_vid,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        save_trajectories=save_trajectories,
        save_vid=save_vid,
        nosave=nosave,
        classes=classes,
        agnostic_nms=agnostic_nms,
        augment=augment,
        visualize=visualize,
        update=update,
        project=project,
        name=name,
        exist_ok=exist_ok,
        line_thickness=line_thickness,
        hide_labels=hide_labels,
        hide_conf=hide_conf,
        hide_class=hide_class,
        half=half,
        dnn=dnn,
        vid_stride=vid_stride,
        retina_masks=retina_masks
    )

if __name__ == "__main__":
    main()
