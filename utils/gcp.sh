#!/usr/bin/env bash

# Start
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3 && cd yolov3 && python3 train.py -img_size 416 -epochs 160

# Resume
cd yolov3 && python3 train.py -img_size 416 -resume 1

# Detect
gsutil cp gs://ultralytics/fresh9_5_e201.pt yolov3/checkpoints
cd yolov3 && python3 detect.py

