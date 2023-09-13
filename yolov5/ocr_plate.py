import os
import sys
import yaml
from pathlib import Path

import torch
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from PIL import Image
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, cv2, clip_boxes,
                           non_max_suppression, scale_boxes, xyxy2xywh, xywh2xyxy)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def ocr_plate(
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
    config_path = "/home/vietnh/Documents/project/speed_detect/yolov5/data/license_plate_ocr.yaml"
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

        # Process predictions
        plate = ""
        for det in pred:  # per image
            if len(det):
                det = det.cpu().tolist()
                for box in det:
                    score, label = box[4], int(box[5])
                    label_name = config["names"][label]
                    if label_name == "background":
                        continue
                    if score < threshold:
                        continue
                    plate += f"{label_name}"

        return plate


if __name__ == '__main__':
    weights_ocr_pathfile = "weights/license_plate_ocr.pt"
    cropped_image_pathfile = "cropped.jpg"

    plate = ocr_plate(weights=weights_ocr_pathfile, source=cropped_image_pathfile, threshold = 0.8)
    print(plate)