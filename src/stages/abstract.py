import cv2
import math

from typing import Any
from src.yoloDet import YoloTRT
from src.utils.calculator import calculate_distance

class AbstractYoloTRT():
    def __init__(self, library: str, engine: str, source: Any, score_threshold: float = 0.5) -> None:
        if isinstance(source, str):
            self.image_name = source.split(".")[0]
            self.image = cv2.imread(source)
        else:
            self.image_name = "untitled"
            self.image = source
        self.ori_height = self.image.shape[0]
        self.ori_width = self.image.shape[1]
        self.score_threshold = score_threshold
        self.model = YoloTRT(library=library, engine=engine, conf=0.5, yolo_ver="v5")
        
    def preprocess(self):
        pass

    def process(self):
        pass
        
    def postprocess(self):
        pass

    def run(self):
        pass


class LicensePlateDetection(AbstractYoloTRT):    
    def process(self):
        detections, t, crops = self.model(self.image)
        return (detections, t, crops)
    
    def postprocess(self, process_results, is_saved: bool = False):
        detections, t, crops = process_results
        if len(detections) == 0:
            return None
        
        ho_center, wo_center = self.ori_height/2, self.ori_width/2
        crop_index, min_distance, score = 0, 1e9, 0
        for i in len(range(detections)):  # per image
            conf, box = detections[i]["conf"], detections[i]["box"]
            if conf < self.score_threshold:
                continue
            hcrop_center, wcrop_center = (box[0]+box[2])/2, (box[1]+box[3])/2
            min_distance = 1e9
            center_distance = self.compare_center((ho_center, wo_center), (hcrop_center, wcrop_center))
            if min_distance > center_distance:
                min_distance = center_distance
                crop_index = i
                score = conf

        if is_saved:
            cv2.imwrite(f"{self.image_name}_cropped.jpg", crops[crop_index])


class LicensePlateOCR(AbstractYoloTRT):
    def __init__(self) -> None:
        pass


class SpeedDetection(AbstractYoloTRT):
    def __init__(self) -> None:
        pass
