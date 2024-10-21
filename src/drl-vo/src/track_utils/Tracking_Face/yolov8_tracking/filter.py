import cv2
import glob
import os
from os.path import join as osj
from yolov8.blur import blury
import shutil
from yolov8.maskpose.maskpose import MaskPose 
MaskPose = MaskPose()

ROOT = './synthesis'
save_folder = './filter_synthesis'
# for folder in os.listdir(ROOT):
#     if not os.path.exists(osj(save_folder,folder)):
#         os.makedirs(osj(save_folder,folder))
#     save = osj(save_folder,folder)
#     image_files = glob.glob(osj(osj(ROOT,folder), '*.jpg'))
#     tmp = []
#     for im0 in image_files:
#         img = cv2.imread(im0)
#         isblur,blurry_score = blury.is_blur(img)
#         print(blurry_score>100)
#         masked, yaw, pitch, roll = MaskPose.get_pose(img)
#         check = blurry_score>100 and -5<=yaw<=5 and -8<=pitch<=8 and -8<=roll<=8
#         check2 = blurry_score>100 and -7.5<=yaw<=7.5 and -20<=pitch<=20 and -20<=roll<=20
#         if check2:
#             tmp.append(im0)
#             if check:
#                 cv2.imwrite(osj(save,im0.split('/')[-1]),img)
#     if len(os.listdir(save)) == 0:
#         if len(tmp) > 0:
#             shutil.copy(tmp[0],osj(save,tmp[0].split('/')[-1]))
#         else: 
#             with open('log.txt', 'a') as the_file:
#                 the_file.write(folder+'\n')


for folder in os.listdir(save_folder):
    save = osj(save_folder,folder)
    image_files = glob.glob(osj(osj(save_folder,folder), '*.jpg'))
    if len(image_files)>1:
        for im0 in image_files[1:]:
            os.remove(im0)