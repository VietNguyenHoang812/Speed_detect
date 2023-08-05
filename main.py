from yolov5.detect import detect_plate
from yolov5.detect_speed import process
# from paddleocr.inference import ocr_plate
from detectnet_image import ocr_plate


weights_plate_pathfile = "weights/license_plate_detection.pt"
weights_speed_pathfile = "weights/speed_detection.pt"
image_pathfile = "P1680346.JPG"

crop, plate_detection_score = detect_plate(weights_plate_pathfile, image_pathfile)
plate_number = ocr_plate(image=crop)
speed = process(weights_speed_pathfile, image_pathfile)

result = {
    "BKS": plate_number,
    "Speed": speed
}
print(result)