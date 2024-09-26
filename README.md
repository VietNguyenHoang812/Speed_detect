# License Plate and Speed Detection on Jetson Nano

## Function
- [x] Detect license plates
- [x] Detect speed number on images

## Pipeline
- Detect license plate: YOLOv5
- OCR license plate: YOLOv5
- Speed detection: Color filtered + YOLOv5

## Data
- Detect license plate: private data
- OCR license plate: lazy in labeling so I'm using data from [Roboflow](https://universe.roboflow.com/nguyen-luan-qagjw). I appreciate 'Nguyen Luan' - a strange comrade - owner of this dataset
- Speed detection: private too

## How to use
### Code
Clone this git repository and check out to 'tensorrt' branch.
'''
git clone https://github.com/VietNguyenHoang812/speed_detect.git
cd speed_detect
git checkout tensorrt
'''

### Weights
Place **weights** folder so that **weights** is subfolder of **speed_detect**.
Oh, where is the **weights** folder? Sorry, private information xD
### Enviroment
I'm using Docker as installing dependencies directly on Jetson Nano is such a pain and very very time-consuming. Virtual enviroment does not have TensorRT or I haven't figured out how to do that properly. 
So here is my Docker images to pull.
```
sudo docker pull vietthevarious/speed_detect:1.1.4
```

### Run
Run docker container
```
docker run -it --rm -p 8000:8000 -v /home/jetson/Documents/speed_detect/:/app --name "speed" vietthevarious/speed_detect:1.1.4 sh build.sh
```
You should place the desired image to infer into **speed_detect/images/** folder, then send api to container



