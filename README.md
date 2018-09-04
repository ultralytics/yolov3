<img src="https://storage.googleapis.com/ultralytics/UltralyticsLogoName1000×676.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information on Ultralytics projects please visit:
http://www.ultralytics.com

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO** (https://pjreddie.com/darknet/yolo/) and to **Erik Lindernoren for the PyTorch implementation** this work is based on (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.6 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch`
- `opencv-python`

# Training

**Start Training:** Run `train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh` and specifying COCO path on line 37 (local) or line 39 (cloud).

**Resume Training:** Run `train.py -resume 1` to resume training from the most recently saved checkpoint `latest.pt`.

Each epoch trains on 120,000 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. An Nvidia GTX 1080 Ti will process about 10-15 epochs/day depending on image size and augmentation (13 epochs/day at 416 pixels with default augmentation). Loss plots for the bounding boxes, objectness and class confidence should appear similar to results shown here (results in progress to 160 epochs, will update).

![Alt](https://github.com/ultralytics/yolov3/blob/master/data/coco_training_loss.png "coco training loss")

## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

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

Checkpoints are saved in `/checkpoints` directory. Run `detect.py` to apply trained weights to an image, such as `zidane.jpg` from the `data/samples` folder, shown here. Alternatively you can use the official YOLOv3 weights:

-PyTorch format: https://storage.googleapis.com/ultralytics/yolov3.pt
-darknet format: https://pjreddie.com/media/files/yolov3.weights

![Alt](https://github.com/ultralytics/yolov3/blob/master/data/zidane_result.jpg "inference example")

# Testing

Run `test.py` to validate the official YOLOv3 weights `checkpoints/yolov3.weights` against the 5000 validation images. You should obtain a mAP of .581 using this repo (https://github.com/ultralytics/yolov3), compared to .579 as reported in darknet (https://arxiv.org/abs/1804.02767).

Run `test.py -weights_path checkpoints/latest.pt` to validate against the latest training checkpoint.

# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
