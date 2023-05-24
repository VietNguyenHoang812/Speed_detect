import cv2
import os
import numpy as np


par_root_folder = "dataset"
sub_root_folder = "type_4"
root_folder = f"{par_root_folder}/{sub_root_folder}"
par_dest_folder = "preprocessed"
if not os.path.isdir(par_dest_folder):
    os.mkdir(par_dest_folder)

sub_dest_folder = f"{par_dest_folder}/{sub_root_folder}"
if not os.path.isdir(sub_dest_folder):
    os.mkdir(sub_dest_folder)

lower = np.array([140, 25, 220])
upper = np.array([179, 255, 255])

list_image_name = os.listdir(root_folder)
for image_name in list_image_name:
    image_pathfile = f"{root_folder}/{image_name}"
    ori_image = cv2.imread(image_pathfile)
    hsv_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, lower, upper)
    preprocessed = cv2.bitwise_and(ori_image, ori_image, mask=mask)
    result_pathfile = f"{sub_dest_folder}/{image_name}"
    cv2.imwrite(result_pathfile, preprocessed)
    print(image_name)