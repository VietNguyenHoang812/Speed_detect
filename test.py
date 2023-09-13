import cv2
import os
import shutil

from yolov5.detect import detect_plate
from yolov5.detect_speed import process
from yolov5.ocr_plate import ocr_plate


weights_plate_pathfile = "weights/license_plate_detection.pt"
weights_plate_ocr_pathfile = "weights/license_plate_ocr.pt"
weights_speed_pathfile = "weights/speed_detection.pt"

val_folder = "/home/vietnh/Documents/project/speed_detect/datasets/type_1_license_plate/images/val"
list_val_images = os.listdir(val_folder)

count = 1
for val_image in list_val_images:
    image_pathfile = f"{val_folder}/{val_image}"
    crop, cropped_savename, plate_detection_score = detect_plate(weights_plate_pathfile, image_pathfile)
    plate_number = ocr_plate(weights_plate_ocr_pathfile, cropped_savename, threshold=0.8)
    speed = process(weights_speed_pathfile, image_pathfile)

    savename = f"{plate_number}_{speed}.jpg"
    savedir = "images/test"
    os.makedirs(savedir, exist_ok=True)
    shutil.copy(image_pathfile, f"{savedir}/{savename}")
    print(f"{count}/{len(list_val_images)}")
    count += 1