import time

from src.stages.plate_detection import LicensePlateDetection
from src.stages.plate_ocr import LicensePlateOCR
from src.stages.speed_detect import SpeedDetection


def pipeline(image_url: str):
    image_url = image_url.replace("%2F", "/")
    start_time = time.time()
    first_stage_result = LicensePlateDetection().run(image_url)
    second_stage_result = LicensePlateOCR().run(first_stage_result)
    print("Time: ", time.time() - start_time)
    return second_stage_result

if __name__ == "__main__":
    # library = "src/yolov5/build/libmyplugins.so"
    # engine = "src/yolov5/build/license_plate_detection.engine"
    # categories = ["plate", "other"]
    # source = "images/P1690088.JPG"
    # LicensePlateDetection(library, engine, categories, source).run()

    # library = "src/yolov5/build/libmyplugins.so"
    # engine = "src/yolov5/build/license_plate_detection.engine"
    # categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    #               "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
    #               "M", "N", "P", "R", "S", "T", "U", "V", "X", "Y",
    #               "Z", "background"]
    # source = "images/P1690088.JPG"
    # print(LicensePlateOCR(library, engine, categories, source).run())

    library = "weights/speed_detection/libmyplugins.so"
    engine = "weights/speed_detection/speed_detection.engine"
    categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    source = "images/P1690088.JPG"
    print(SpeedDetection(library, engine, categories, source).run())
