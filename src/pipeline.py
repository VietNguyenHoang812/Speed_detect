from src.stages.abstract import LicensePlateDetection


if __name__ == "__main__":
    library = "src/yolov5/build/libmyplugins.so"
    engine = "src/yolov5/build/license_plate_detection.engine"
    source = "images/P1690088.JPG"
    LicensePlateDetection(library, engine, source).run()
