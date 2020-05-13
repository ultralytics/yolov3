<table style="width:100%">
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/26833433/61591130-f7beea00-abc2-11e9-9dc0-d6abcf41d713.jpg">
    </td>
    <td align="center">
    <a href="https://www.ultralytics.com" target="_blank">
    <img src="https://storage.googleapis.com/ultralytics/logo/logoname1000.png" width="160"></a>
      <img src="https://user-images.githubusercontent.com/26833433/61591093-2b4d4480-abc2-11e9-8b46-d88eb1dabba1.jpg">
          <a href="https://itunes.apple.com/app/id1452689527" target="_blank">
    <img src="https://user-images.githubusercontent.com/26833433/50044365-9b22ac00-0082-11e9-862f-e77aee7aa7b0.png" width="180"></a>
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/26833433/61591100-55066b80-abc2-11e9-9647-52c0e045b288.jpg">
    </td>
  </tr>
</table>

# Introduction

This directory contains PyTorch YOLOv3 software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information please visit https://www.ultralytics.com.

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO:** https://pjreddie.com/darknet/yolo/.

# Requirements

Python 3.7 or later with all `pip install -U -r requirements.txt` packages including `torch >= 1.5`. Docker images come with all dependencies preinstalled. Docker requirements are: 
- Nvidia Driver >= 440.44
- Docker Engine - CE >= 19.03

# Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data) < highly recommended!!
* [Train Single Class](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Class)
* [Google Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb) with quick training, inference and testing examples
* [GCP Quickstart](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
* [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart) 
* [A TensorRT Implementation of YOLOv3 and YOLOv4](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3-spp) 

# Training

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco2017.sh`. Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set.

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt`.

**Plot Training:** `from utils import utils; utils.plot_results()`

<img src="https://user-images.githubusercontent.com/26833433/78175826-599d4800-7410-11ea-87d4-f629071838f6.png" width="900">

## Image Augmentation

`datasets.py` applies OpenCV-powered (https://opencv.org/) augmentation to the input image. We use a **mosaic dataloader** to increase image variability during training.

<img src="https://user-images.githubusercontent.com/26833433/80769557-6e015d00-8b02-11ea-9c4b-69310eb2b962.jpg" width="900">

## Speed

https://cloud.google.com/deep-learning-vm/  
**Machine type:** preemptible [n1-standard-8](https://cloud.google.com/compute/docs/machine-types) (8 vCPUs, 30 GB memory)   
**CPU platform:** Intel Skylake  
**GPUs:** K80 ($0.14/hr), T4 ($0.11/hr), V100 ($0.74/hr) CUDA with [Nvidia Apex](https://github.com/NVIDIA/apex) FP16/32    
**HDD:** 300 GB SSD  
**Dataset:** COCO train 2014 (117,263 images)  
**Model:** `yolov3-spp.cfg`  
**Command:**  `python3 train.py --data coco2017.data --img 416 --batch 32`

GPU | n | `--batch-size` | img/s | epoch<br>time | epoch<br>cost
--- |--- |--- |--- |--- |---
K80    |1| 32 x 2 | 11  | 175 min  | $0.41
T4     |1<br>2| 32 x 2<br>64 x 1 | 41<br>61 | 48 min<br>32 min | $0.09<br>$0.11
V100   |1<br>2| 32 x 2<br>64 x 1 | 122<br>**178** | 16 min<br>**11 min** | **$0.21**<br>$0.28
2080Ti |1<br>2| 32 x 2<br>64 x 1 | 81<br>140 | 24 min<br>14 min | -<br>-

# Inference

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067835-51d5b500-cc2f-11e9-982e-843f7f9a6ea2.jpg" width="500">

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067834-51d5b500-cc2f-11e9-9357-c485b159a20b.jpg" width="500">

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights yolov3-spp.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067833-51d5b500-cc2f-11e9-8208-6fe197809131.jpg" width="500">


# Pretrained Weights

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'weights/yolov3-spp.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'weights/yolov3-spp.weights'
```

# mAP

<i></i>                      |Size |COCO mAP<br>@0.5...0.95 |COCO mAP<br>@0.5 
---                          | ---         | ---         | ---
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |320 |14.0<br>28.7<br>30.5<br>**37.7** |29.1<br>51.8<br>52.3<br>**56.8**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |416 |16.0<br>31.2<br>33.9<br>**41.2** |33.0<br>55.4<br>56.9<br>**60.6**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |512 |16.6<br>32.7<br>35.6<br>**42.6** |34.9<br>57.7<br>59.5<br>**62.4**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |608 |16.6<br>33.1<br>37.0<br>**43.1** |35.4<br>58.2<br>60.7<br>**62.8**

- mAP@0.5 run at `--iou-thr 0.5`, mAP@0.5...0.95 run at `--iou-thr 0.7`
- Darknet results: https://arxiv.org/abs/1804.02767

```bash
$ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt --img 640 --augment

Namespace(augment=True, batch_size=16, cfg='cfg/yolov3-spp.cfg', conf_thres=0.001, data='coco2014.data', device='', img_size=640, iou_thres=0.6, save_json=True, single_cls=False, task='test', weights='weight
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16130MB)

               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████| 313/313 [03:00<00:00,  1.74it/s]
                 all     5e+03  3.51e+04     0.375     0.743      0.64     0.492

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.647
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.810

Speed: 17.5/2.3/19.9 ms inference/NMS/total per 640x640 image at batch-size 16
```
<!-- Speed: 11.4/2.2/13.6 ms inference/NMS/total per 608x608 image at batch-size 1 -->


# Reproduce Our Results

Run commands below. Training takes about one week on a 2080Ti per model.
```bash
$ python train.py --data coco2014.data --weights '' --batch-size 16 --cfg yolov3-spp.cfg
$ python train.py --data coco2014.data --weights '' --batch-size 32 --cfg yolov3-tiny.cfg
```
<img src="https://user-images.githubusercontent.com/26833433/80831822-57a9de80-8ba0-11ea-9684-c47afb0432dc.png" width="900">

# Reproduce Our Environment

To access an up-to-date working environment (with all dependencies including CUDA/CUDNN, Python and PyTorch preinstalled), consider a:

- **GCP** Deep Learning VM with $300 free credit offer: See our [GCP Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart) 
- **Google Colab Notebook** with 12 hours of free GPU time: [Google Colab Notebook](https://colab.sandbox.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb)
- **Docker Image** from https://hub.docker.com/r/ultralytics/yolov3. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart) 
# Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)

# Contact

**Issues should be raised directly in the repository.** For additional questions or comments please email Glenn Jocher at glenn.jocher@ultralytics.com or visit us at https://contact.ultralytics.com.
