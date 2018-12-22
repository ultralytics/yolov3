<img src="https://storage.googleapis.com/ultralytics/UltralyticsLogoName1000×676.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information on Ultralytics projects please visit:
http://www.ultralytics.com.

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO** (https://pjreddie.com/darknet/yolo/) and to **Erik Lindernoren for the PyTorch implementation** this work is based on (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch`
- `opencv-python`

# Training

**Start Training:** Run `train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Training runs about 1 hour per COCO epoch on a 1080 Ti.

**Resume Training:** Run `train.py --resume` to resume training from the most recently saved checkpoint `weights/latest.pt`.

Each epoch trains on 120,000 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. An Nvidia GTX 1080 Ti will process about 10-20 epochs/day depending on image size and augmentation. Loss plots are shown here using default training settings.

![Alt](https://user-images.githubusercontent.com/26833433/49822374-3b27bf00-fd7d-11e8-9180-f0ac9fe2fdb4.png "coco training loss")

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

Run `detect.py` to apply trained weights to an image and visualize results, such as `zidane.jpg` from the `data/samples` folder, shown here. 

**YOLOv3:** `detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.pt`
![Alt](https://github.com/ultralytics/yolov3/blob/master/data/zidane_result.jpg "yolov3 example")

**YOLOv3-tiny:** `detect.py --cfg cfg/yolov3-tiny.cfg --weights weights/yolov3-tiny.pt`
![Alt](https://user-images.githubusercontent.com/26833433/50374155-21427380-05ea-11e9-8d24-f1a4b2bac1ad.jpg "yolov3-tiny example")

# Pretrained Weights

Download official YOLOv3 weights:

**Darknet** format: 
- https://pjreddie.com/media/files/yolov3.weights
- https://pjreddie.com/media/files/yolov3-tiny.weights

**PyTorch** format:
- https://storage.googleapis.com/ultralytics/yolov3.pt 
- https://storage.googleapis.com/ultralytics/yolov3-tiny.pt 

# Validation mAP

Run `test.py` to validate the official YOLOv3 weights `weights/yolov3.weights` against the 5000 validation images. You should obtain a .584 mAP at `--img-size 416`, or .586 at `--img-size 608` using this repo, compared to .579 at 608 x 608 reported in darknet (https://arxiv.org/abs/1804.02767).

Run `test.py --weights weights/latest.pt` to validate against the latest training results. **Default training settings produce a 0.522 mAP at epoch 62.** Hyperparameter settings and loss equation changes affect these results significantly, and additional trade studies may be needed to further improve this.

# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
