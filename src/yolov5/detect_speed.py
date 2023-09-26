import os
import sys
from pathlib import Path

import torch
import math
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, cv2, clip_boxes,
                           non_max_suppression, scale_boxes, xyxy2xywh, xywh2xyxy)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def detect_speed(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        threshold = 0.4,
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
        # cropped_image, min_distance, score = None, 1e7, None
        # original_image = cv2.imread(source)
        # ho_center, wo_center = original_image.shape[0]/2, original_image.shape[1]/2
        
        min_conf, preds_list = 1e7, []
        speed = ""
        for det in pred:  # per image
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    conf, cls = conf.cpu().numpy(), cls.cpu().numpy()
                    conf = round(conf.tolist(), 2)
                    if conf < threshold:
                        continue
                    _, xyxy = crop_image(xyxy, im0, BGR=True)
                    left_width = xyxy.numpy()[0][0]
                    if len(preds_list) <= 2:
                        if min_conf > conf:
                            min_conf = conf
                            preds_list.append([left_width, int(cls), conf])
                        else:
                            preds_list.insert(0, [left_width, int(cls), conf])
                    else:
                        if min_conf > conf:
                            preds_list.pop()
                            if conf < preds_list[0][2]:
                                min_conf = conf
                                preds_list.append([left_width, int(cls), conf])
                            else:
                                preds_list.insert(0, [left_width, int(cls), conf])
                            
                    # hcrop_center = (xyxy[0][0]+xyxy[0][2])/2
                    # wcrop_center = (xyxy[0][1]+xyxy[0][3])/2

                    # center_distance = compare_center((ho_center, wo_center), (hcrop_center, wcrop_center))
                    # if min_distance > center_distance:
                    #     min_distance = center_distance
                    #     cropped_image = crop
                    #     score = conf
        if len(preds_list) != 2:
            return 0
        else:
            (left_1, n1, _), (left2, n2, _) = preds_list
            if left_1 < left2:
                return n1*10+n2
            else:
                return n2*10+n1

def crop_image(xyxy, im, gain=1.02, pad=10, square=False, BGR=False, save=True):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]

    return crop, xyxy

def preprocess(image_pathfile):
    preprocessed_image_pathfile="temp.jpg"
    lower = np.array([140, 25, 220])
    upper = np.array([179, 255, 255])
    ori_image = cv2.imread(image_pathfile)
    hsv_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)
    preprocessed = cv2.bitwise_and(ori_image, ori_image, mask=mask)
    cv2.imwrite(preprocessed_image_pathfile, preprocessed)

    return preprocessed_image_pathfile

def process(weights_pathfile, source):
    preprocessed_image_pathfile = preprocess(source)
    speed = detect_speed(weights=weights_pathfile, source=preprocessed_image_pathfile)

    return speed

if __name__ == '__main__':
    # opt = parse_opt()
    # main(opt)
    weights_pathfile = "weights/speed_detection.pt"
    test_image_pathfile = "P1690088.JPG"
    # preprocessed_image_pathfile = "temp.jpg"

    speed = process(weights_pathfile, test_image_pathfile)
    print(speed)
