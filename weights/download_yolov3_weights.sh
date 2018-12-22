#!/bin/bash

# make '/weights' directory if it does not exist and cd into it
mkdir -p weights && cd weights

# copy weight files, continue '-c' if partially downloaded
# yolov3 darknet weights
wget -c https://pjreddie.com/media/files/yolov3.weights
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
wget -c https://pjreddie.com/media/files/yolov3-spp.weights

# yolov3 pytorch weights
wget -c https://storage.googleapis.com/ultralytics/yolov3.pt
wget -c https://storage.googleapis.com/ultralytics/yolov3-tiny.pt

# darknet53 weights (first 75 layers only)
wget -c https://pjreddie.com/media/files/darknet53.conv.74
