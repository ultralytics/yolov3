<table style="width:100%">
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/26833433/56132958-9e279d80-5f8b-11e9-8d23-c34dbb2a39e6.jpg">
    </td>
    <td align="center">
    <a href="https://www.ultralytics.com" target="_blank">
    <img src="https://storage.googleapis.com/ultralytics/logo/logoname1000.png" width="200"></a>
    <a href="https://itunes.apple.com/app/id1452689527" target="_blank">
    <img src="https://user-images.githubusercontent.com/26833433/50044365-9b22ac00-0082-11e9-862f-e77aee7aa7b0.png" width="180"></a>
      <img src="https://user-images.githubusercontent.com/26833433/56132957-9d8f0700-5f8b-11e9-9067-0d91364c57c0.jpg">
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/26833433/56132991-b0094080-5f8b-11e9-89a1-62fb65b98d45.jpg">
    </td>
  </tr>
</table>

# Introduction

This directory contains PyTorch YOLOv3 software and an iOS App developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information please visit https://www.ultralytics.com.

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

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`.

**Resume Training:** `python3 train.py --resume` to resume training from `weights/latest.pt`.

Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. Default training settings produce loss plots below, with **training speed of 0.25 s/batch on a V100 GPU (almost 50 COCO epochs/day)**.

Here we see training results from `coco_1img.data`, `coco_10img.data` and `coco_100img.data`, 3 example files available in the `data/` folder, which train and test on the first 1, 10 and 100 images of the coco2014 trainval dataset.

`from utils import utils; utils.plot_results()`
![results](https://user-images.githubusercontent.com/26833433/56207787-ec9e7000-604f-11e9-94dd-e1fcc374270f.png)

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

## Darknet Conversion

```bash
git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'converted.pt'

# convert cfg/pytorch model to darknet weights
python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'converted.weights'
```

# mAP

- Use `test.py --weights weights/yolov3.weights` to test the official YOLOv3 weights.
- Use `test.py --weights weights/latest.pt` to test the latest training results.
- Compare to darknet published results https://arxiv.org/abs/1804.02767.

<!---
%<i></i> | ultralytics/yolov3 OR-NMS 5:52@416 (`pycocotools`) | darknet  
--- | --- | ---  
YOLOv3-320 | 51.9 (51.4) | 51.5  
YOLOv3-416 | 55.0 (54.9) | 55.3  
YOLOv3-608 | 57.5 (57.8) | 57.9  

<i></i> | ultralytics/yolov3 MERGE-NMS 7:15@416 (`pycocotools`) | darknet  
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
<i></i> | [ultralytics/yolov3](https://github.com/ultralytics/yolov3) | [darknet](https://arxiv.org/abs/1804.02767) 
--- | --- | ---  
`YOLOv3 320` | 51.8 | 51.5  
`YOLOv3 416` | 55.4 | 55.3  
`YOLOv3 608` | 58.2 | 57.9  
`YOLOv3-spp 320` | 52.4 | -  
`YOLOv3-spp 416` | 56.5 | -  
`YOLOv3-spp 608` | 60.7 | 60.6  

``` bash
git clone https://github.com/ultralytics/yolov3
# bash yolov3/data/get_coco_dataset.sh
git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools yolov3
cd yolov3
 
python3 test.py --save-json --img-size 416
Namespace(batch_size=32, cfg='cfg/yolov3-spp.cfg', conf_thres=0.001, data_cfg='data/coco.data', img_size=416, iou_thres=0.5, nms_thres=0.5, save_json=True, weights='weights/yolov3-spp.weights')
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16130MB)
               Class    Images   Targets         P         R       mAP        F1
Calculating mAP: 100%|█████████████████████████████████████████| 157/157 [05:59<00:00,  1.71s/it]
                 all     5e+03  3.58e+04     0.109     0.773      0.57     0.186
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.565
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.620

python3 test.py --save-json --img-size 608 --batch-size 16
Namespace(batch_size=16, cfg='cfg/yolov3-spp.cfg', conf_thres=0.001, data_cfg='data/coco.data', img_size=608, iou_thres=0.5, nms_thres=0.5, save_json=True, weights='weights/yolov3-spp.weights')
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16130MB)
               Class    Images   Targets         P         R       mAP        F1
Computing mAP: 100%|█████████████████████████████████████████| 313/313 [06:11<00:00,  1.01it/s]
                 all     5e+03  3.58e+04      0.12      0.81     0.611     0.203
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.607
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.618

```

# Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)

# Contact

Issues should be raised directly in the repository. For additional questions or comments please email Glenn Jocher at glenn.jocher@ultralytics.com or visit us at https://contact.ultralytics.com.
