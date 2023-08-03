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

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

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
net = jetson_inference.detectNet(network, sys.argv, threshold)

img = jetson_utils.loadImage("resized.jpg")

# detect objects in the image (with overlay)
detections = net.Detect(img, overlay=overlay)

# print the detections
print("detected {:d} objects in image".format(len(detections)))

for detection in detections:
	print(detection)

# print out performance info
net.PrintProfilerTimes()


