#!/usr/bin/env bash

# Start
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3 && cd yolov3 && python3 train.py

# Resume
python3 train.py -resume 1

# Detect
gsutil cp gs://ultralytics/yolov3.pt yolov3/weights
python3 detect.py

# Test
python3 test.py -img_size 416 -weights_path weights/latest.pt

# Download and Test
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3 && cd yolov3
wget https://pjreddie.com/media/files/yolov3.weights -P weights
python3 test.py -img_size 416 -weights_path weights/backup5.pt -nms_thres 0.45

# Download and Resume
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3 && cd yolov3
wget https://storage.googleapis.com/ultralytics/yolov3.pt -O weights/latest.pt
python3 train.py -img_size 416 -batch_size 16 -epochs 1 -resume 1
python3 test.py -img_size 416 -weights_path weights/latest.pt -conf_thres 0.5

# Copy latest.pt to bucket
gsutil cp yolov3/weights/latest.pt gs://ultralytics

# Copy latest.pt from bucket
gsutil cp gs://ultralytics/latest.pt yolov3/weights/latest.pt

