import cv2
import pyrealsense2 as rs
import sys
import os

# Calculate the relative path
relative_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'track_utils')

# Append the relative path to sys.path
sys.path.append(relative_path)

import numpy as np
import torch
# torch.backends.cudnn.version = 85096
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align
import pandas as pd
import os
import keyboard
import time
from time import sleep
import pickle

def normalize(emb):
    emb = np.squeeze(emb)
    norm = np.linalg.norm(emb)
    return emb / norm
###########################################
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
database = 'database.pkl'
datas = []
###########################################
user_name = input('Type your user name \n')

count = 0
# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (10, 50)

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (255, 146, 0)

# Line thickness of 2 px
thickness = 2
text = 'Press enter to begin capture'
print(10*'*'+ ' PREPARE TO CAPTURE FACE ' + 10*'*')
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
img_counter = 0
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution 
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
camera_device = pipeline_profile.get_device()
device_product_line = str(camera_device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in camera_device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
f_pixel = (COLOR_WIDTH * 0.5) / np.tan(69 * 0.5 * np.pi / 180)
display_center = COLOR_WIDTH // 2
config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Capture frame-by-frame
        frame = color_image#cap.read()
        img = frame.copy()
        # Using cv2.putText() method
        frame = cv2.putText(frame, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        faces = app.get(img)
        if len(faces) != 0:
            face_crop = face_align.norm_crop(img,faces[0].kps, image_size=112)
            normed_feat = normalize(faces[0].embedding)
            data = {"name": user_name, "emb": normed_feat}
            datas.append(data)

        # with open("sample_{usn}.pkl".format(usn = user_name), "wb") as f:
        #     pickle.dump(datas, f)
        # if keyboard.is_pressed('enter'):
        #     sleep(0.5)
        #     data = {"folder": user_name, "emb": normalize(face[0].embedding)}
        #     datas.append(data)
        # # Display the resulting frame
        frame = app.draw_on(frame, faces)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open("sample_{usn}.pkl".format(usn = user_name), "wb") as f:
                pickle.dump(datas, f)
            # if not os.path.exists(database):
            #     pass
            #     # df.to_pickle(database)
            # else:
            #     old_df = pd.read_pickle(database)
            #     new_df = pd.concat([df, old_df],ignore_index=True)
                # new_df.to_pickle(database)
            break
finally:
    pipeline.stop()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import torch
# torch.backends.cudnn.version = 85096
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
#
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
# img = cv2.imread('huy0.jpg')#ins_get_image('huy0')
# faces = app.get(img)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)
# #'F:\\PROJECT\\KHOA_LUAN_MTMC\\model\\models\\buffalo_l'#