<img src="https://storage.googleapis.com/ultralytics/UltralyticsLogoName1000×676.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC. For more information on Ultralytics projects please visit:
http://www.ultralytics.com  

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO** (https://pjreddie.com/darknet/yolo/) and to **Erik Lindernoren for the pytorch implementation** this work is based on (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch`
- `opencv-python`

# Training

Run `train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Each epoch trains on 120,000 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. An Nvidia GTX 1080 Ti will process ~10 epochs/day with full image augmentation, or ~15 epochs/day with no augmentation. Loss plots for bounding boxes, objectness and classification should appear similar to results shown here (training currently in-progress to 160 epochs).
![Alt](https://github.com/ultralytics/yolov3/blob/master/data/coco_training_loss.png "training loss")

# Inference

Checkpoints will be saved in `/checkpoints` directory. Run `detect.py` to apply trained weights to an image, such as `zidane.jpg` from the `data/samples` folder, shown here.
![Alt](https://github.com/ultralytics/yolov3/blob/master/data/zidane_result.jpg "example")

# Testing

Run `test.py` to test the latest checkpoint on the 5000 validation images. Joseph Redmon's official YOLOv3 weights produce a mAP of .581 using this PyTorch implementation, compared to .579 in darknet (https://arxiv.org/abs/1804.02767).

# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
