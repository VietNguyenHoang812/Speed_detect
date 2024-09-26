import cv2
import math
import numpy as np

from typing import Any, List
from src.yoloDet import YoloTRT
from src.utils.calculator import calculate_distance
from src.utils.singleton import Singleton

config = {
    "library": "weights/license_plate_detection/libmyplugins.so",
    "engine": "weights/license_plate_detection/license_plate_detection.engine",
    "categories": ["plate", "other"]
}
class LicensePlateDetection(metaclass=Singleton):    
    def __init__(
            self, 
            library: str = None, 
            engine: str = None, 
            categories: List[str] = None, 
            score_threshold: float = 0.5) -> None:
        library = config["library"] if library is None else library
        engine = config["engine"] if engine is None else engine
        categories = config["categories"] if categories is None else categories
        self.score_threshold = score_threshold
        self.model = YoloTRT(library=library, engine=engine, conf=0.5, yolo_ver="v5", categories=categories)
        
    def preprocess(self, source):
        if isinstance(source, str):
            self.image_name = source.split(".")[0]
            self.image = cv2.imread(source)
        else:
            self.image_name = "untitled"
            self.image = source
        self.ori_height = self.image.shape[0]
        self.ori_width = self.image.shape[1]
        return self.image
    
    def process(self):
        detections, t, crops = self.model.Inference(self.image)
        return (detections, t, crops)
    
    def postprocess(self, process_results, is_saved: bool = False):
        detections, t, crops = process_results
        if len(detections) == 0:
            return None
        
        ho_center, wo_center = self.ori_height/2, self.ori_width/2
        crop_index, min_distance, score = 0, 1e9, 0
        for i in range(len(detections)):  # per image
            conf, box = detections[i]["conf"], detections[i]["box"]
            if conf < self.score_threshold:
                continue
            hcrop_center, wcrop_center = (box[0]+box[2])/2, (box[1]+box[3])/2
            min_distance = 1e9
            center_distance = calculate_distance((ho_center, wo_center), (hcrop_center, wcrop_center))
            if min_distance > center_distance:
                min_distance = center_distance
                crop_index = i
                score = conf

        if is_saved:
            cv2.imwrite(f"{self.image_name}_cropped.jpg", crops[crop_index])
        return crops[crop_index]

    def run(self, source):
        self.preprocess(source)
        process_results = self.process()
        license_plate_box = self.postprocess(process_results)
        return license_plate_box
