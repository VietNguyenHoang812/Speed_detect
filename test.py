import cv2
import os
import shutil
import time

from yolov5.detect_plate import detect_plate
from yolov5.detect_speed import process
from yolov5.ocr_plate import ocr_plate


def pipeline(image_pathfile, weights_plate_pathfile, weights_plate_ocr_pathfile, weights_speed_pathfile, save_pathfile):
    plate_number, speed = "None", 0
    crop, cropped_savename, plate_detection_score = detect_plate(weights_plate_pathfile, image_pathfile, is_saved=True)
    if not plate_detection_score:
        shutil.copy(image_pathfile, save_pathfile)
        return plate_number, speed, None
    plate_number = ocr_plate(weights_plate_ocr_pathfile, cropped_savename, threshold=0.6)
    # print(plate_number)
    speed = process(weights_speed_pathfile, image_pathfile)
    # return speed
    return plate_number, speed, crop

if __name__ == "__main__":
    weights_plate_pathfile = "weights/license_plate_detection.pt"
    weights_plate_ocr_pathfile = "weights/license_plate_ocr.pt"
    weights_speed_pathfile = "weights/speed_detection.pt"

    # Mass inference
    # val_folder = "datasets/type_1_license_plate/images/val"
    # list_val_images = os.listdir(val_folder)

    # process_times = []
    # count = 1
    # savedir = "images/test"
    fail_folder = "images/fail"
    # crop_folder = "images/crop"
    # os.makedirs(savedir, exist_ok=True)
    os.makedirs(fail_folder, exist_ok=True)
    # os.makedirs(crop_folder, exist_ok=True)
    # for val_image in list_val_images:
    #     image_name = val_image.split(".")[0]
    #     image_pathfile = f"{val_folder}/{val_image}"
        
    #     start_time = time.time()
    #     plate_number, speed, crop = pipeline(image_pathfile, weights_plate_pathfile, weights_plate_ocr_pathfile, weights_speed_pathfile, f"{fail_folder}/{val_image}")
    #     end_time = time.time()
    #     savename = f"{image_name}_{plate_number}_{speed}.jpg"
    #     savename = f"{image_name}_{plate_number}_{str(speed)}.jpg"
    #     shutil.copy(image_pathfile, f"{savedir}/{savename}")

    #     process_time = end_time - start_time
    #     process_times.append(process_time)
    #     print(f"{count}/{len(list_val_images)}", process_time)
    #     count += 1
    
    # print(sum(process_times))
    # print(sum(process_times)/len(process_times))

    # Single inference
    val_folder = "datasets/type_1_license_plate/images/val"
    image_name = "P1690088.JPG"
    image_pathfile = f"{val_folder}/{image_name}"

    start_time = time.time()
    plate_number, speed, crop = pipeline(image_pathfile, weights_plate_pathfile, weights_plate_ocr_pathfile, weights_speed_pathfile, f"{fail_folder}/{image_name}")
    end_time = time.time()

    shutil.copy(image_pathfile, f"{image_name.split('.')[0]}_{plate_number}_{speed}.jpg")

    print("Plate number: ", plate_number)
    print("Speed: ", speed)
    print("Processed time: ", end_time - start_time)