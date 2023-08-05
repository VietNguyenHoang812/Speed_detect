import jetson_inference
import jetson_utils

# import argparse
import sys

# parse the command line
# parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
#                                  formatter_class=argparse.RawTextHelpFormatter, epilog=jetson_inference.detectNet.Usage() +
#                                  jetson_utils.videoSource.Usage() + jetson_utils.videoOutput.Usage() + jetson_utils.logUsage())

# parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
# parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
# parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
# parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
# parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

# is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

# try:
# 	opt = parser.parse_known_args()[0]
# except:
# 	print("")
# 	parser.print_help()
# 	sys.exit(0)

network = "ssd-mobilenet-v2"
threshold = 0.5
overlay = "box,labels,conf"

# load the object detection network
net = jetson_inference.detectNet(argv=["--model=networks/az_ocr/az_ocr_ssdmobilenetv1_2.onnx", "--labels=networks/az_ocr/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)

img = jetson_utils.loadImage("cropped.jpg")

# detect objects in the image (with overlay)
detections = net.Detect(img, overlay=overlay)

# print the detections
print("detected {:d} objects in image".format(len(detections)))

predictions = []
for detection in detections:
	# print(detection)
	prediction = {
		"Value": net.GetClassDesc(detection.ClassID),
		"Left": detection.Left,
		"Top": detection.Top,
		"Sum": detection.Left + detection.Top
	}
	predictions.append(prediction)
	# print(net.GetClassDesc(detection.ClassID))
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

print(license_plate)
# print out performance info
net.PrintProfilerTimes()


