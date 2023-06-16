from yolov5.detect import detect_plate
from paddleocr.inference import ocr_plate


weights_pathfile = "weights/license_plate_detection.pt"
image_pathfile = "test.JPG"

crop, plate_detection_score = detect_plate(weights_pathfile, image_pathfile)
plate_number = ocr_plate(image=crop)
print(plate_number)
