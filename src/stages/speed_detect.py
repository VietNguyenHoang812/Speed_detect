import cv2
import math
import numpy as np

from typing import Any, List
from src.yoloDet import YoloTRT
from src.utils.singleton import Singleton


library = "weights/speed_detection/libmyplugins.so"
engine = "weights/speed_detection/speed_detection.engine"
categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

class SpeedDetection(metaclass=Singleton):
    def __init__(
            self, 
            library: str = library, 
            engine: str = engine, 
            categories: List[str] = categories, 
            score_threshold: float = 0.5) -> None:
        self.score_threshold = score_threshold
        self.model = YoloTRT(library=library, engine=engine, conf=0.5, yolo_ver="v5", categories=categories)
        
    def preprocess(self, source):
        if isinstance(source, str):
            self.image_name = source.split(".")[0]
            self.image = cv2.imread(source)
        else:
            self.image_name = "untitled"
            self.image = source
        # preprocessed_image_pathfile="temp.jpg"
        lower = np.array([140, 25, 220])
        upper = np.array([179, 255, 255])
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower, upper)
        filtered_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        # cv2.imwrite(preprocessed_image_pathfile, filtered_image)
        return filtered_image

    def process(self, filtered_image):
        detections, t, crops = self.model.Inference(filtered_image)
        return (detections, t, crops)
    
    def postprocess(self, process_results):
        detections, t, crops = process_results
        if len(detections) == 0:
            return None
        
        min_conf, preds_list = 1e7, []
        for i in range(len(detections)):  # per image
            label_name, conf, box = detections[i]["class"], detections[i]["conf"], detections[i]["box"]
            left_width = int(box[0])
            if conf < self.score_threshold:
                continue
            if len(preds_list) <= 2:
                if min_conf > conf:
                    min_conf = conf
                    preds_list.append([left_width, int(label_name), conf])
                else:
                    preds_list.insert(0, [left_width, int(label_name), conf])
            else:
                if min_conf > conf:
                    preds_list.pop()
                    if conf < preds_list[0][2]:
                        min_conf = conf
                        preds_list.append([left_width, int(label_name), conf])
                    else:
                        preds_list.insert(0, [left_width, int(label_name), conf])
                    
        if len(preds_list) != 2:
            return -1
        else:
            (left_1, n1, _), (left2, n2, _) = preds_list
            if left_1 < left2:
                speed = n1*10+n2
            else:
                speed = n2*10+n1
        return speed
    
    def run(self):
        filtered_image = self.preprocess()
        process_results = self.process(filtered_image)
        speed = self.postprocess(process_results)
        return speed
