import os
import shutil
import time

from typing import List, Tuple
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
    speed = process(weights_speed_pathfile, image_pathfile)

    return plate_number, speed, crop

def mass_inference(
    image_folder: str,
    weights_plate_pathfile: str = "weights/license_plate_detection.pt", 
    weights_plate_ocr_pathfile: str = "weights/license_plate_ocr.pt", 
    weights_speed_pathfile: str = "weights/speed_detection.pt"
) -> List[Tuple[str, int]]:
    results = []
    list_val_images = os.listdir(image_folder)
    fail_folder = "images/fail"
    os.makedirs(fail_folder, exist_ok=True)
    for val_image in list_val_images:
        image_pathfile = f"{val_folder}/{val_image}"
        plate_number, speed, crop = pipeline(image_pathfile, weights_plate_pathfile, weights_plate_ocr_pathfile, weights_speed_pathfile, f"{fail_folder}/{val_image}")
        results.append((plate_number, speed))

    return results

def single_inference(
    image_pathfile: str,
    weights_plate_pathfile: str = "weights/license_plate_detection.pt", 
    weights_plate_ocr_pathfile: str = "weights/license_plate_ocr.pt", 
    weights_speed_pathfile: str = "weights/speed_detection.pt"
) -> Tuple[str, int]:
    fail_folder = "images/fail"
    os.makedirs(fail_folder, exist_ok=True)
    image_name = image_pathfile.split("/")[-1]
    plate_number, speed, crop = pipeline(image_pathfile, weights_plate_pathfile, weights_plate_ocr_pathfile, weights_speed_pathfile, f"{fail_folder}/{image_name}")
    
    return plate_number, speed

if __name__ == "__main__":
    weights_plate_pathfile = "weights/license_plate_detection.pt"
    weights_plate_ocr_pathfile = "weights/license_plate_ocr.pt"
    weights_speed_pathfile = "weights/speed_detection.pt"

    # Mass inference

    # Single inference
    val_folder = "datasets/type_1_license_plate/images/val"
    image_name = "P1690088.JPG"
    image_pathfile = f"{val_folder}/{image_name}"

    start_time = time.time()
    plate_number, speed = single_inference(image_pathfile=image_pathfile)
    end_time = time.time()

    print("Plate number: ", plate_number)
    print("Speed: ", speed)
    print("Processed time: ", end_time - start_time)