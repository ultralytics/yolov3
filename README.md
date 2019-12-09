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

Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

# Tutorials

* [GCP Quickstart](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
* [Transfer Learning](https://github.com/ultralytics/yolov3/wiki/Example:-Transfer-Learning)
* [Train Single Image](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Image)
* [Train Single Class](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Class)
* [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)

# Jupyter Notebook

Our Jupyter [notebook](https://colab.research.google.com/github/ultralytics/yolov3/blob/master/examples.ipynb) provides quick training, inference and testing examples.

# Training

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set.

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt`.

**Plot Training:** `from utils import utils; utils.plot_results()` plots training results from `coco_16img.data`, `coco_64img.data`, 2 example datasets available in the `data/` folder, which train and test on the first 16 and 64 images of the COCO2014-trainval dataset.

<img src="https://user-images.githubusercontent.com/26833433/63258271-fe9d5300-c27b-11e9-9a15-95038daf4438.png" width="900">

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

<img src="https://user-images.githubusercontent.com/26833433/66699231-27beea80-ece5-11e9-9cad-bdf9d82c500a.jpg" width="900">

## Speed

https://cloud.google.com/deep-learning-vm/  
**Machine type:** [n1-standard-16](https://cloud.google.com/compute/docs/machine-types) (16 vCPUs, 60 GB memory)   
**CPU platform:** Intel Skylake  
**GPUs:** K80 ($0.20/hr), T4 ($0.35/hr), V100 ($0.83/hr) CUDA with [Nvidia Apex](https://github.com/NVIDIA/apex) FP16/32  
**HDD:** 1 TB SSD  
**Dataset:** COCO train 2014 (117,263 images)  
**Model:** `yolov3-spp.cfg`  
**Command:**  `python3 train.py --img 416 --batch 32 --accum 2`

GPU |n| `--batch --accum` | img/s | epoch<br>time | epoch<br>cost
--- |--- |--- |--- |--- |---
K80    |1| 32 x 2 | 11  | 175 min  | $0.58
T4     |1<br>2| 32 x 2<br>64 x 1 | 41<br>61 | 48 min<br>32 min | $0.28<br>$0.36
V100   |1<br>2| 32 x 2<br>64 x 1 | 122<br>**178** | 16 min<br>**11 min** | **$0.23**<br>$0.31
2080Ti |1<br>2| 32 x 2<br>64 x 1 | 81<br>140 | 24 min<br>14 min | -<br>-

# Inference

`detect.py` runs inference on any sources:

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

To run a specific models:

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.weights`  
<img src="https://user-images.githubusercontent.com/26833433/64067835-51d5b500-cc2f-11e9-982e-843f7f9a6ea2.jpg" width="500">

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights`  
<img src="https://user-images.githubusercontent.com/26833433/64067834-51d5b500-cc2f-11e9-9357-c485b159a20b.jpg" width="500">

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights yolov3-spp.weights`  
<img src="https://user-images.githubusercontent.com/26833433/64067833-51d5b500-cc2f-11e9-8208-6fe197809131.jpg" width="500">


# Pretrained Weights

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'converted.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'converted.weights'
```

# mAP

- `test.py --weights weights/yolov3.weights` tests official YOLOv3 weights.
- `test.py --weights weights/last.pt` tests latest checkpoint.
- mAPs on COCO2014 using pycocotools.
- mAP@0.5 run at `--nms-thres 0.5`, mAP@0.5...0.95 run at `--nms-thres 0.7`.
- YOLOv3-SPP ultralytics is `ultralytics68.pt` with `yolov3-spp.cfg`.
- Darknet results published in https://arxiv.org/abs/1804.02767.

<i></i>                      |Size |COCO mAP<br>@0.5...0.95 |COCO mAP<br>@0.5 
---                          | ---         | ---         | ---
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |320 |14.0<br>28.7<br>30.5<br>**35.4** |29.1<br>51.8<br>52.3<br>**54.3**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |416 |16.0<br>31.2<br>33.9<br>**39.0** |33.0<br>55.4<br>56.9<br>**59.2**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |512 |16.6<br>32.7<br>35.6<br>**40.3** |34.9<br>57.7<br>59.5<br>**60.6**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |608 |16.6<br>33.1<br>37.0<br>**40.9** |35.4<br>58.2<br>60.7<br>**60.9**

```bash
$ python3 test.py --save-json --img-size 608 --nms-thres 0.5 --weights ultralytics68.pt

Namespace(batch_size=16, cfg='cfg/yolov3-spp.cfg', conf_thres=0.001, data='data/coco.data', device='1', img_size=608, iou_thres=0.5, nms_thres=0.7, save_json=True, weights='ultralytics68.pt')
Using CUDA device0 _CudaDeviceProperties(name='GeForce RTX 2080 Ti', total_memory=11019MB)

               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████████████████████████████████████| 313/313 [09:46<00:00,  1.09it/s]
                 all     5e+03  3.58e+04    0.0823     0.798     0.595     0.145
              person     5e+03  1.09e+04    0.0999     0.903     0.771      0.18
             bicycle     5e+03       316    0.0491     0.782      0.56    0.0925
                 car     5e+03  1.67e+03    0.0552     0.845     0.646     0.104
          motorcycle     5e+03       391      0.11     0.847     0.704     0.194
            airplane     5e+03       131     0.099     0.947     0.878     0.179
                 bus     5e+03       261     0.142     0.874     0.825     0.244
               train     5e+03       212     0.152     0.863     0.806     0.258
               truck     5e+03       352    0.0849     0.682     0.514     0.151
                boat     5e+03       475    0.0498     0.787     0.504    0.0937
       traffic light     5e+03       516    0.0304     0.752     0.516    0.0584
        fire hydrant     5e+03        83     0.144     0.916     0.882     0.248
           stop sign     5e+03        84    0.0833     0.917     0.809     0.153
       parking meter     5e+03        59    0.0607     0.695     0.611     0.112
               bench     5e+03       473    0.0294     0.685     0.363    0.0564
                bird     5e+03       469    0.0521     0.716     0.524    0.0972
                 cat     5e+03       195     0.252     0.908      0.78     0.395
                 dog     5e+03       223     0.192     0.883     0.829     0.315
               horse     5e+03       305     0.121     0.911     0.843     0.214
               sheep     5e+03       321     0.114     0.854     0.724     0.201
                 cow     5e+03       384     0.105     0.849     0.695     0.187
            elephant     5e+03       284     0.184     0.944     0.912     0.308
                bear     5e+03        53     0.358     0.925     0.875     0.516
               zebra     5e+03       277     0.176     0.935     0.858     0.297
             giraffe     5e+03       170     0.171     0.959     0.892      0.29
            backpack     5e+03       384    0.0426     0.708     0.392    0.0803
            umbrella     5e+03       392    0.0672     0.878      0.65     0.125
             handbag     5e+03       483    0.0238     0.629     0.242    0.0458
                 tie     5e+03       297    0.0419     0.805     0.599    0.0797
            suitcase     5e+03       310    0.0823     0.855     0.628      0.15
             frisbee     5e+03       109     0.126     0.872     0.796     0.221
                skis     5e+03       282    0.0473     0.748     0.454     0.089
           snowboard     5e+03        92    0.0579     0.804     0.559     0.108
         sports ball     5e+03       236     0.057     0.733     0.622     0.106
                kite     5e+03       399     0.087     0.852     0.645     0.158
        baseball bat     5e+03       125    0.0496     0.776     0.603    0.0932
      baseball glove     5e+03       139    0.0511     0.734     0.563    0.0956
          skateboard     5e+03       218    0.0655     0.844      0.73     0.122
           surfboard     5e+03       266    0.0709     0.827     0.651     0.131
       tennis racket     5e+03       183    0.0694     0.858     0.759     0.128
              bottle     5e+03       966    0.0484     0.812     0.513    0.0914
          wine glass     5e+03       366    0.0735     0.738     0.543     0.134
                 cup     5e+03       897    0.0637     0.788     0.538     0.118
                fork     5e+03       234    0.0411     0.662     0.487    0.0774
               knife     5e+03       291    0.0334     0.557     0.292    0.0631
               spoon     5e+03       253    0.0281     0.621     0.307    0.0537
                bowl     5e+03       620    0.0624     0.795     0.514     0.116
              banana     5e+03       371     0.052      0.83      0.41    0.0979
               apple     5e+03       158    0.0293     0.741     0.262    0.0564
            sandwich     5e+03       160    0.0913     0.725     0.522     0.162
              orange     5e+03       189    0.0382     0.688      0.32    0.0723
            broccoli     5e+03       332    0.0513      0.88     0.445     0.097
              carrot     5e+03       346    0.0398     0.766     0.362    0.0757
             hot dog     5e+03       164    0.0958     0.646     0.494     0.167
               pizza     5e+03       224    0.0886     0.875     0.699     0.161
               donut     5e+03       237    0.0925     0.827      0.64     0.166
                cake     5e+03       241    0.0658      0.71     0.539      0.12
               chair     5e+03  1.62e+03    0.0432     0.793     0.489    0.0819
               couch     5e+03       236     0.118     0.801     0.584     0.205
        potted plant     5e+03       431    0.0373     0.852     0.505    0.0714
                 bed     5e+03       195     0.149     0.846     0.693     0.253
        dining table     5e+03       634    0.0546      0.82      0.49     0.102
              toilet     5e+03       179     0.161      0.95      0.81     0.275
                  tv     5e+03       257    0.0922     0.903      0.79     0.167
              laptop     5e+03       237     0.127     0.869     0.744     0.222
               mouse     5e+03        95    0.0648     0.863     0.732      0.12
              remote     5e+03       241    0.0436     0.788     0.535    0.0827
            keyboard     5e+03       117    0.0668     0.923     0.755     0.125
          cell phone     5e+03       291    0.0364     0.704     0.436    0.0692
           microwave     5e+03        88     0.154     0.841     0.743     0.261
                oven     5e+03       142    0.0618     0.803     0.576     0.115
             toaster     5e+03        11    0.0565     0.636     0.191     0.104
                sink     5e+03       211    0.0439     0.853     0.544    0.0835
        refrigerator     5e+03       107    0.0791     0.907     0.742     0.145
                book     5e+03  1.08e+03    0.0399     0.667     0.233    0.0753
               clock     5e+03       292    0.0542     0.836     0.733     0.102
                vase     5e+03       353    0.0675     0.799     0.591     0.125
            scissors     5e+03        56    0.0397      0.75     0.461    0.0755
          teddy bear     5e+03       245    0.0995     0.882     0.669     0.179
          hair drier     5e+03        11   0.00508    0.0909    0.0475   0.00962
          toothbrush     5e+03        77    0.0371      0.74     0.418    0.0706

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.600
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.243
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.707
```

# Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)

# Contact

**Issues should be raised directly in the repository.** For additional questions or comments please email Glenn Jocher at glenn.jocher@ultralytics.com or visit us at https://contact.ultralytics.com.
