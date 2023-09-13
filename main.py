from yolov5.detect import detect_plate
from yolov5.detect_speed import process
from yolov5.ocr_plate import ocr_plate

weights_plate_pathfile = "weights/license_plate_detection.pt"
weights_plate_ocr_pathfile = "weights/license_plate_ocr.pt"
weights_speed_pathfile = "weights/speed_detection.pt"
image_pathfile = "images/P1680346.JPG"

crop, cropped_savename, plate_detection_score = detect_plate(weights_plate_pathfile, image_pathfile)
plate_number = ocr_plate(weights_plate_ocr_pathfile, cropped_savename, threshold=0.8)
speed = process(weights_speed_pathfile, image_pathfile)

result = {
    "BKS": plate_number,
    "Speed": speed
}
print(result)