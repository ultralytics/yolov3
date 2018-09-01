<img src="https://storage.googleapis.com/ultralytics/UltralyticsLogoName1000×676.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information on Ultralytics projects please visit:
http://www.ultralytics.com  

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO** (https://pjreddie.com/darknet/yolo/) and to **Erik Lindernoren for the pytorch implementation** this work is based on (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch`
- `opencv-python`

# Training

Run `train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Each epoch trains on 120,000 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. An Nvidia GTX 1080 Ti will process ~10 epochs/day with full augmentation, or ~15 epochs/day without input image augmentation. Loss plots for the bounding boxes, objectness and class confidence should appear similar to results shown here (in progress to 160 epochs, will update)
![Alt](https://github.com/ultralytics/yolov3/blob/master/data/coco_training_loss.png "coco training loss")

## Image Augmentation

`datasets.py` applies random augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 20% (vertical and horizontal)
Rotation | +/- 5 degrees
Shear | +/- 3 degrees (vertical and horizontal)
Scale | +/- 20%
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%

![Alt](https://github.com/ultralytics/yolov3/blob/master/data/coco_augmentation_examples.jpg "coco image augmentation")

# Inference

Checkpoints will be saved in `/checkpoints` directory. Run `detect.py` to apply trained weights to an image, such as `zidane.jpg` from the `data/samples` folder, shown here.
![Alt](https://github.com/ultralytics/yolov3/blob/master/data/zidane_result.jpg "inference example")

# Testing

Run `test.py` to test the latest checkpoint on the 5000 validation images. Joseph Redmon's official YOLOv3 weights produce a mAP of .581 using this PyTorch implementation, compared to .579 in darknet (https://arxiv.org/abs/1804.02767).

# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
