import yaml


def load_yaml(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

if __name__ == "__main__":
    config_path = "/home/vietnh/Documents/project/speed_detect/yolov5/data/license_plate_ocr.yaml"
    config = load_yaml(config_path)
    label = 10
    print(config["names"][label])