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
        detections, t, crops = self.model.Inference(self.image)
        return (detections, t, crops)
    
    def postprocess(self, process_results, is_saved: bool = True):
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

    def run(self):
        process_results = self.process()
        self.postprocess(process_results)


class LicensePlateOCR(AbstractYoloTRT):
    def process(self):
        detections, t, crops = self.model.Inference(self.image)
        return (detections, t, crops)
    
    def postprocess(self, process_results, is_saved: bool = True):
        detections, t, crops = process_results
        if len(detections) == 0:
            return None
        
        predictions = []
        for i in range(len(detections)):  # per image
            label_name, conf, box = detections[i]["class"], detections[i]["conf"], detections[i]["box"]
            left, top = int(box[0]), int(box[1])
            if label_name == "background":
                continue
            if conf < self.score_threshold:
                continue
            prediction = {
                "Value": str(label_name),
                "Score": conf,
                "Top": top,
                "Left": left,
                "Sum": top+left,
            }
            predictions.append(prediction)
        
        # License plate must have at least 7 characters
        if len(predictions) < 7:
            return str(len(predictions))
        
        # Sort characters
        license_plate = ""
        HIGH_ALIGN_RATIO = 2
        LOW_ALIGN_RATIO = 0.5
        sorted_list = sorted(predictions, key=lambda i: (i["Sum"], i["Top"]))
        align_ratio = sorted_list[0]["Top"]/sorted_list[1]["Top"]
        if (align_ratio > HIGH_ALIGN_RATIO or align_ratio < LOW_ALIGN_RATIO) and sorted_list[1]["Left"] > sorted_list[0]["Left"]:
            first = sorted_list.pop(1)
        else:
            first = sorted_list.pop(0)

        license_plate += first["Value"]
        while sorted_list:
            top_first = first["Top"]
            left_first = 9999
            index_pop = -1
            for i, p in enumerate(sorted_list):
                top_p, left_p = p["Top"], p["Left"]
                align_ratio = top_p/top_first
                # check if p and first are on the same line
                if align_ratio < HIGH_ALIGN_RATIO and align_ratio > LOW_ALIGN_RATIO and left_p < left_first:
                    index_pop = i
                    left_first = left_p
            if index_pop == -1:
                sorted_list = sorted(sorted_list, key=lambda i: i["Sum"])
                index_pop = 0

            first = sorted_list.pop(index_pop)
            license_plate += first["Value"]
        
        return license_plate

    def run(self):
        process_results = self.process()
        self.postprocess(process_results)

        
class SpeedDetection(AbstractYoloTRT):
    def __init__(self) -> None:
        pass
