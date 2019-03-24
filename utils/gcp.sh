#!/usr/bin/env bash

# New VM
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3
bash yolov3/data/get_coco_dataset.sh
bash yolov3/weights/download_yolov3_weights.sh
sudo rm -rf cocoapi && git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools yolov3
sudo shutdown

# Convert COCO to *.bmp
cd yolov3
python3
from utils.datasets import *
convert_images2bmp('../coco/images/val2014/')
convert_images2bmp('../coco/images/train2014/')
exit()
sudo shutdown

# Train
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3
cp -r weights yolov3
cd yolov3 && python3 train.py --batch-size 24 --epochs 1
sudo shutdown

# Resume
python3 train.py --resume

# Detect
python3 detect.py

# Clone branch
sudo rm -rf yolov3 && git clone -b multi_gpu --depth 1 https://github.com/ultralytics/yolov3
cp -r weights yolov3
cd yolov3 && python3 train.py --batch-size 24 --epochs 1
sudo shutdown

# Git pull branch
git pull https://github.com/ultralytics/yolov3 multi_gpu

# Test
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3
sudo rm -rf cocoapi && git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools yolov3
cd yolov3 && python3 test.py --save-json --conf-thres 0.005

# Test Darknet training
python3 test.py --img_size 416 --weights ../darknet/backup/yolov3.backup

# Download with wget
wget https://storage.googleapis.com/ultralytics/yolov3.pt -O weights/latest.pt

# Copy latest.pt to bucket
gsutil cp yolov3/weights/latest.pt gs://ultralytics

# Copy latest.pt from bucket
gsutil cp gs://ultralytics/latest.pt yolov3/weights/latest.pt
wget https://storage.googleapis.com/ultralytics/latest.pt

# Trade Studies
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3
cp -r weights yolov3
cd yolov3 && python3 train.py --batch-size 16 --epochs 1
sudo shutdown
