#!/usr/bin/env bash

# New VM
rm -rf yolov3 weights coco
git clone https://github.com/ultralytics/yolov3
bash yolov3/weights/download_yolov3_weights.sh && cp -r weights yolov3
bash yolov3/data/get_coco_dataset.sh
git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools yolov3
sudo reboot now

# Re-clone
sudo rm -rf yolov3
# git clone https://github.com/ultralytics/yolov3  # master
git clone -b map_update --depth 1 https://github.com/ultralytics/yolov3 yolov3  # branch
cp -r weights yolov3
cp -r cocoapi/PythonAPI/pycocotools yolov3
cd yolov3

# Train
python3 train.py

# Resume
python3 train.py --resume

# Detect
python3 detect.py

# Test
python3 test.py --save-json

# Git pull
git pull https://github.com/ultralytics/yolov3  # master
git pull https://github.com/ultralytics/yolov3 map_update  # branch

# Test Darknet training
python3 test.py --weights ../darknet/backup/yolov3.backup

# Copy latest.pt TO bucket
gsutil cp yolov3/weights/latest1gpu.pt gs://ultralytics

# Copy latest.pt FROM bucket
gsutil cp gs://ultralytics/latest.pt yolov3/weights/latest.pt
wget https://storage.googleapis.com/ultralytics/yolov3/latest_v1_0.pt -O weights/latest_v1_0.pt
wget https://storage.googleapis.com/ultralytics/yolov3/best_v1_0.pt -O weights/best_v1_0.pt

# Debug/Development
sudo rm -rf yolov3
git clone https://github.com/ultralytics/yolov3  # master
# git clone -b hyperparameter_search --depth 1 https://github.com/ultralytics/yolov3 hyperparameter_search  # branch
cp -r weights yolov3
cp -r cocoapi/PythonAPI/pycocotools yolov3
cd yolov3

git pull https://github.com/ultralytics/yolov3 #hyperparameter_search  # branch
python3 train.py --data-cfg data/coco_1cls.data
python3 train.py --data-cfg data/coco_1img.data


