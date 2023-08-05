import jetson_inference
import jetson_utils

from typing import List, Dict


def ocr_plate(img_pathfile: str = "cropped.jpg") -> str:
	predictions = process(img_pathfile)
	license_plate = postprocess(predictions)

	return license_plate

def process(img_pathfile: str = "cropped.jpg") -> List[Dict]:
	threshold = 0.6
	overlay = "box,labels,conf"
	img = jetson_utils.loadImage(img_pathfile)

	# load the object detection network
	net = jetson_inference.detectNet(argv=[
			"--model=networks/az_ocr/az_ocr_ssdmobilenetv1_2.onnx", 
			"--labels=networks/az_ocr/labels.txt", 
			"--input-blob=input_0", 
			"--output-cvg=scores", 
			"--output-bbox=boxes"
		], 
		threshold=threshold
	)

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=overlay)
	print("detected {:d} objects in image".format(len(detections)))

	predictions = []
	for detection in detections:
		prediction = {
			"Value": net.GetClassDesc(detection.ClassID),
			"Left": detection.Left,
			"Top": detection.Top,
			"Sum": detection.Left + detection.Top
		}
		predictions.append(prediction)
	
	return predictions

def postprocess(predictions: List[Dict]) -> str:
	license_plate = ""
	sorted_list = sorted(predictions, key=lambda i: (i["Sum"], i["Top"]))
	if sorted_list[0]["Left"] > sorted_list[1]["Left"]:
		first = sorted_list.pop(0)
	else:
		first = sorted_list.pop(1)

	license_plate += first["Value"]
	while sorted_list:
		top_first = first["Top"]
		left_first = 9999
		index_pop = -1
		for i, p in enumerate(sorted_list):
			top_p, left_p = p["Top"], p["Left"]
			if top_p > 0.8*top_first and top_p < 1.2*top_first and left_p < left_first:
				index_pop = i
				left_first = left_p
		if index_pop == -1:
			sorted_list = sorted(sorted_list, key=lambda i: i["Sum"])
			index_pop = 0

		first = sorted_list.pop(index_pop)
		license_plate += first["Value"]
	
	return license_plate


if __name__ == "__main__":
	print(ocr_plate())