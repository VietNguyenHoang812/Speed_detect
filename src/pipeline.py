from src.stages.abstract import LicensePlateDetection, LicensePlateOCR, SpeedDetection


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