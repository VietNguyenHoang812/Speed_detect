import os
import shutil
import sys
import yaml
import time
from pathlib import Path

import torch
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from typing import List, Dict
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def process(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        threshold = 0.9,
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)

    # Load config
    config_path = "src/yolov5/license_plate_ocr.yaml" 
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        predictions = []
        # Process predictions
        for det in pred:  # per image
            im0 = im0s.copy()
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                det = det.cpu().tolist()
                for box in det:
                    left, top, score, label = box[0], box[1], box[4], int(box[5])
                    label_name = config["names"][label]
                    # print(box[:4], label_name)
                    if label_name == "background":
                        continue
                    if score < threshold:
                        continue
                    prediction = {
                        "Value": str(label_name),
                        "Score": score,
                        "Top": top,
                        "Left": left,
                        "Sum": top+left,
                    }
                    predictions.append(prediction)
            break

        return predictions
    
def postprocess(predictions: List[Dict]) -> str:
    if len(predictions) < 7:
        return str(len(predictions))
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

def ocr_plate(weights, source, threshold = 0.6):
    predictions = process(weights=weights, source=source, threshold=threshold)
    if len(predictions) == 0:
        return "none"
    license_plate = postprocess(predictions)
    
    return license_plate

if __name__ == '__main__':
    weights_ocr_pathfile = "weights/license_plate_ocr.pt"

    # Single inference
    # cropped_image_pathfile = "images/crop/P1690095_cropped.jpg"
    # plate = ocr_plate(weights=weights_ocr_pathfile, source=cropped_image_pathfile, threshold = 0.6)
    # print(plate)

    # Mass inference
    cropped_folder = "images/crop"
    ocr_results_folder = "images/ocr"
    os.makedirs(ocr_results_folder, exist_ok=True)
    list_images = os.listdir(cropped_folder)
    for image_path in list_images:
        image_name = image_path.split("_")[0]
        image_pathfile = f"{cropped_folder}/{image_path}"

        start_time = time.time()
        plate = ocr_plate(weights=weights_ocr_pathfile, source=image_pathfile, threshold = 0.6)
        end_time = time.time()
        print("OCR processed time: ", end_time - start_time)

        shutil.copy(image_pathfile, f"{ocr_results_folder}/{image_name}_{plate}.jpg")