from yolov5.detect import predict
from paddleocr.inference import predict_plate


weights_pathfile = "weights/license_plate_detection.pt"
image_pathfile = "test.JPG"

crop, plate_detection_score = predict(weights_pathfile, image_pathfile)
plate_number = predict_plate(image=crop)
print(plate_number)
