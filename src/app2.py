import sys
import cv2 
import imutils
from yoloDet import YoloTRT

# use path for library and engine file
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

frame = imutils.resize(frame, width=600)
detections, t = model.Inference(frame)

cv2.imshow("Output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()