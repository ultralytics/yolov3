<table style="width:100%">
  <tr>
    <th>v2.2<img src="https://user-images.githubusercontent.com/26833433/52743528-e6096300-2fe2-11e9-970c-5fee45769fab.jpg" width="400"></th>
        <th>v3.0<img src="https://user-images.githubusercontent.com/26833433/54523854-227d0580-4979-11e9-9801-26a3be239875.jpg" width="400"></th>
    <th><img src="https://storage.googleapis.com/ultralytics/logo/logoname1000.png" width="200">
  <br><br/>
  <p> <a href="https://itunes.apple.com/app/id1452689527">
  <img href="https://itunes.apple.com/app/id1452689527" src="https://user-images.githubusercontent.com/26833433/50044365-9b22ac00-0082-11e9-862f-e77aee7aa7b0.png" width="180"> 
  </a> </p></th> 
  </tr>
</table>


# Introduction

This directory contains python software and an iOS App developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information please visit https://www.ultralytics.com.

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO:** https://pjreddie.com/darknet/yolo/.

# Requirements

Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch >= 1.0.0`
- `opencv-python`
- `tqdm`

# Tutorials

* [GCP Quickstart](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
* [Transfer Learning](https://github.com/ultralytics/yolov3/wiki/Example:-Transfer-Learning)
* [Train Single Image](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Image)
* [Train Single Class](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Class)
* [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)

# Training

**Start Training:** Run `train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`.

**Resume Training:** Run `train.py --resume` resumes training from the latest checkpoint `weights/latest.pt`.

Each epoch trains on 120,000 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. Default training settings produce loss plots below, with **training speed of 0.6 s/batch on a 1080 Ti (18 epochs/day)** or 0.45 s/batch on a 2080 Ti.

`from utils import utils; utils.plot_results()`
![Alt](https://user-images.githubusercontent.com/26833433/53494085-3251aa00-3a9d-11e9-8af7-8c08cf40d70b.png "train.py results")

## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 10% (vertical and horizontal)
Rotation | +/- 5 degrees
Shear | +/- 2 degrees (vertical and horizontal)
Scale | +/- 10%
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%

<img src="https://user-images.githubusercontent.com/26833433/50525037-6cbcbc00-0ad9-11e9-8c38-9fd51af530e0.jpg">

## Speed

https://cloud.google.com/deep-learning-vm/  
**Machine type:** n1-standard-8 (8 vCPUs, 30 GB memory)  
**CPU platform:** Intel Skylake  
**GPUs:** K80 ($0.198/hr), P4 ($0.279/hr), T4 ($0.353/hr), P100 ($0.493/hr), V100 ($0.803/hr)  
**HDD:** 100 GB SSD  
**Dataset:** COCO train 2014 

GPUs | `batch_size` | batch time | epoch time | epoch cost
--- |---| --- | --- | --- 
<i></i> |  (images)  | (s/batch) |  |
1 K80 | 16 | 1.43s  | 175min  | $0.58
1 P4 | 8 | 0.51s  | 125min  | $0.58
1 T4 | 16 | 0.78s  | 94min  | $0.55
1 P100 | 16 | 0.39s  | 48min  | $0.39
2 P100 | 32 | 0.48s | 29min | $0.47
4 P100 | 64 | 0.65s | 20min | $0.65
1 V100 | 16 | 0.25s  | 31min | $0.41
2 V100 | 32 | 0.29s | 18min | $0.48
4 V100 | 64 | 0.41s | 13min | $0.70
8 V100 | 128 | 0.49s | 7min | $0.80

# Inference

Run `detect.py` to apply trained weights to an image, such as `zidane.jpg` from the `data/samples` folder:

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights`
<img src="https://user-images.githubusercontent.com/26833433/50524393-b0adc200-0ad5-11e9-9335-4774a1e52374.jpg" width="600">

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights weights/yolov3-tiny.weights`
<img src="https://user-images.githubusercontent.com/26833433/50374155-21427380-05ea-11e9-8d24-f1a4b2bac1ad.jpg" width="600">

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights weights/yolov3-spp.weights`
<img src="https://user-images.githubusercontent.com/26833433/54747926-e051ff00-4bd8-11e9-8b5d-93a41d871ec7.jpg" width="600">

## Webcam

Run `detect.py` with `webcam=True` to show a live webcam feed.

# Pretrained Weights

- Darknet `*.weights` format: https://pjreddie.com/media/files/yolov3.weights
- PyTorch `*.pt` format: https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI

# mAP

- Use `test.py --weights weights/yolov3.weights` to test the official YOLOv3 weights.
- Use `test.py --weights weights/latest.pt` to test the latest training results.
- Compare to darknet published results https://arxiv.org/abs/1804.02767.

<!---
%<i></i> | ultralytics/yolov3 fastest 5:52@416 (`pycocotools`) | darknet  
--- | --- | ---  
YOLOv3-320 | 51.9 (51.4) | 51.5  
YOLOv3-416 | 55.0 (54.9) | 55.3  
YOLOv3-608 | 57.5 (57.8) | 57.9  

<i></i> | ultralytics/yolov3 MERGE 7:15@416 (`pycocotools`) | darknet  
--- | --- | ---  
YOLOv3-320 | 52.3 (51.7) | 51.5  
YOLOv3-416 | 55.4 (55.3) | 55.3  
YOLOv3-608 | 57.9 (58.1) | 57.9  

<i></i> | ultralytics/yolov3 MERGE+earlier_pred4 8:34@416 (`pycocotools`) | darknet  
--- | --- | ---  
YOLOv3-320 | 52.3 (51.8) | 51.5  
YOLOv3-416 | 55.5 (55.4) | 55.3  
YOLOv3-608 | 57.9 (58.2) | 57.9  
--->
<i></i> | [ultralytics/yolov3](https://github.com/ultralytics/yolov3) with `pycocotools` | [darknet/yolov3](https://arxiv.org/abs/1804.02767) 
--- | --- | ---  
YOLOv3-320 | 51.8 | 51.5  
YOLOv3-416 | 55.4 | 55.3  
YOLOv3-608 | 58.2 | 57.9  

``` bash
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3
# bash yolov3/data/get_coco_dataset.sh
sudo rm -rf cocoapi && git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools yolov3
cd yolov3

python3 test.py --save-json --conf-thres 0.001 --img-size 416
Namespace(batch_size=32, cfg='cfg/yolov3.cfg', conf_thres=0.001, data_cfg='cfg/coco.data', img_size=416, iou_thres=0.5, nms_thres=0.5, save_json=True, weights='weights/yolov3.weights')
Using cuda _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', major=7, minor=0, total_memory=16130MB, multi_processor_count=80)
      Image      Total          P          R        mAP
Calculating mAP: 100%|█████████████████████████████████| 157/157 [08:34<00:00,  2.53s/it]
       5000       5000     0.0896      0.756      0.555
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.317
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.268
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.244
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.587
 
python3 test.py --save-json --conf-thres 0.001 --img-size 608 --batch-size 16
Namespace(batch_size=16, cfg='cfg/yolov3.cfg', conf_thres=0.001, data_cfg='cfg/coco.data', img_size=608, iou_thres=0.5, nms_thres=0.5, save_json=True, weights='weights/yolov3.weights')
Using cuda _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', major=7, minor=0, total_memory=16130MB, multi_processor_count=80)
      Image      Total          P          R        mAP
Calculating mAP: 100%|█████████████████████████████████| 313/313 [08:54<00:00,  1.55s/it]
       5000       5000     0.0966      0.786      0.579
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.198
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.281
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.577
```

# Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)

# Contact

Issues should be raised directly in the repository. For additional questions or comments please email Glenn Jocher at glenn.jocher@ultralytics.com or visit us at https://contact.ultralytics.com/contact.
