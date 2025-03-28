<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

<div align="center">
  <p>
    <a align="center" href="https://www.ultralytics.com/yolo" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png"></a>
  </p>

[中文](https://docs.ultralytics.com/zh) | [한국어](https://docs.ultralytics.com/ko) | [日本語](https://docs.ultralytics.com/ja) | [Русский](https://docs.ultralytics.com/ru) | [Deutsch](https://docs.ultralytics.com/de) | [Français](https://docs.ultralytics.com/fr) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt) | [Türkçe](https://docs.ultralytics.com/tr) | [Tiếng Việt](https://docs.ultralytics.com/vi) | [العربية](https://docs.ultralytics.com/ar)

<div>
    <a href="https://github.com/ultralytics/yolov3/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov3/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv3 CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv3 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/yolov3"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov3?logo=docker" alt="Docker Pulls"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://reddit.com/r/ultralytics"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

Ultralytics YOLOv3 🚀 is a significant iteration in the YOLO (You Only Look Once) family of real-time [object detection](https://docs.ultralytics.com/tasks/detect/) models. Originally developed by Joseph Redmon, YOLOv3 improved upon its predecessors by enhancing accuracy, particularly for smaller objects, through techniques like multi-scale predictions and a more complex backbone network (Darknet-53). This repository represents [Ultralytics'](https://www.ultralytics.com/) implementation, building upon the foundational work and incorporating best practices learned through extensive research in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl).

We hope the resources here help you leverage YOLOv3 effectively. Explore the [Ultralytics YOLOv3 Docs](https://docs.ultralytics.com/models/yolov3/) for detailed guides, raise issues on [GitHub](https://github.com/ultralytics/yolov3/issues/new/choose) for support, and join our vibrant [Discord community](https://discord.com/invite/ultralytics) for questions and discussions!

To request an Enterprise License for commercial use, please complete the form at [Ultralytics Licensing](https://www.ultralytics.com/license).

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>
</div>
<br>

## <div align="center">🚀 Ultralytics YOLO11 is Here!</div>

We are thrilled to announce the release of Ultralytics YOLO11 🚀, the next generation in state-of-the-art (SOTA) vision models! Now available at the main **[Ultralytics repository](https://github.com/ultralytics/ultralytics)**, YOLO11 continues our commitment to speed, [accuracy](https://www.ultralytics.com/glossary/accuracy), and user-friendliness. Whether your focus is [object detection](https://docs.ultralytics.com/tasks/detect/), [image segmentation](https://docs.ultralytics.com/tasks/segment/), or [image classification](https://docs.ultralytics.com/tasks/classify/), YOLO11 offers unparalleled performance and versatility for a wide range of applications.

Start exploring YOLO11 today! Visit the [Ultralytics Docs](https://docs.ultralytics.com/) for comprehensive guides and resources.

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics)

```bash
pip install ultralytics
```

<div align="center">
  <a href="https://www.ultralytics.com/yolo" target="_blank">
  <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png"></a>
</div>

## <div align="center">📚 Documentation</div>

See the [Ultralytics YOLOv3 Docs](https://docs.ultralytics.com/models/yolov3/) for full documentation on training, validation, inference, and deployment. Quickstart examples are provided below.

<details open>
<summary>Install</summary>

Clone the repository and install dependencies using `requirements.txt` in a **Python>=3.7.0** environment. Ensure you have **PyTorch>=1.7**.

```bash
git clone https://github.com/ultralytics/yolov3 # clone repository
cd yolov3
pip install -r requirements.txt # install dependencies
```

</details>

<details>
<summary>Inference</summary>

Perform inference using YOLOv3 models loaded via [PyTorch Hub](https://pytorch.org/hub/). Models are automatically downloaded from the latest YOLOv3 release.

```python
import torch

# Load a YOLOv3 model (e.g., yolov3, yolov3-spp)
model = torch.hub.load("ultralytics/yolov3", "yolov3", pretrained=True)  # specify 'yolov3' or other variants

# Define image source (URL, file path, PIL image, OpenCV image, numpy array, list of images)
img_source = "https://ultralytics.com/images/zidane.jpg"

# Perform inference
results = model(img_source)

# Process and display results
results.print()  # Print results to console
# results.show()  # Display results in a window
# results.save()  # Save results to runs/detect/exp
# results.crop()  # Save cropped detections
# results.pandas().xyxy[0] # Access results as pandas DataFrame
```

</details>

<details>
<summary>Inference with detect.py</summary>

The `detect.py` script runs inference on various sources. It automatically downloads required [models](https://github.com/ultralytics/yolov5/tree/master/models) from the latest YOLOv3 release and saves the output to `runs/detect`.

```bash
# Run inference using detect.py with different sources
python detect.py --weights yolov3.pt --source 0                              # Webcam
python detect.py --weights yolov3.pt --source image.jpg                      # Single image
python detect.py --weights yolov3.pt --source video.mp4                      # Video file
python detect.py --weights yolov3.pt --source screen                         # Screen capture
python detect.py --weights yolov3.pt --source path/                          # Directory of images/videos
python detect.py --weights yolov3.pt --source list.txt                       # Text file with image paths
python detect.py --weights yolov3.pt --source list.streams                   # Text file with stream URLs
python detect.py --weights yolov3.pt --source 'path/*.jpg'                   # Glob pattern for images
python detect.py --weights yolov3.pt --source 'https://youtu.be/LNwODJXcvt4' # YouTube video URL
python detect.py --weights yolov3.pt --source 'rtsp://example.com/media.mp4' # RTSP, RTMP, HTTP stream URL
```

</details>

<details>
<summary>Training</summary>

The following command demonstrates training YOLOv3 on the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). Models and datasets are automatically downloaded from the latest YOLOv3 release. Training times vary depending on the model size and hardware; for instance, YOLOv5 variants (often used as a reference) take 1-8 days on a V100 GPU. Use the largest possible `--batch-size` or utilize `--batch-size -1` for YOLOv3 [AutoBatch](https://docs.ultralytics.com/reference/utils/autobatch/). Batch sizes shown are indicative for a V100-16GB GPU.

```bash
# Train YOLOv3 on COCO dataset for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3.yaml --batch-size 128
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>📚 Tutorials</summary>

- [Train Custom Data](https://docs.ultralytics.com/datasets/detect/) 🚀 **RECOMMENDED**: Learn how to train YOLO models on your own datasets.
- [Tips for Best Training Results](https://docs.ultralytics.com/guides/model-training-tips/) ☘️: Improve your model's performance with expert tips.
- [Multi-GPU Training](https://docs.ultralytics.com/usage/cli/#multi-gpu-training): Scale your training across multiple GPUs.
- [PyTorch Hub Integration](https://docs.ultralytics.com/integrations/pytorch-hub/): Load models easily using PyTorch Hub. 🌟 **NEW**
- [Model Export](https://docs.ultralytics.com/modes/export/): Export models to various formats like TFLite, ONNX, CoreML, TensorRT. 🚀
- [NVIDIA Jetson Deployment](https://docs.ultralytics.com/guides/nvidia-jetson/): Deploy models on NVIDIA Jetson devices. 🌟 **NEW**
- [Test-Time Augmentation (TTA)](https://docs.ultralytics.com/modes/val/#inference-augmentation): Enhance prediction accuracy using TTA.
- [Model Ensembling](https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.ensemble): Combine multiple models for better robustness.
- [Model Pruning/Sparsity](https://docs.ultralytics.com/integrations/neural-magic/): Optimize models for size and speed.
- [Hyperparameter Evolution](https://docs.ultralytics.com/guides/hyperparameter-tuning/): Automatically tune hyperparameters for optimal performance.
- [Transfer Learning](https://docs.ultralytics.com/guides/transfer-learning/): Fine-tune pretrained models on your custom data.
- [Architecture Summary](https://docs.ultralytics.com/models/yolov5/#architecture): Understand the underlying model architecture. 🌟 **NEW**
- [Ultralytics HUB Training](https://docs.ultralytics.com/hub/cloud-training/) 🚀 **RECOMMENDED**: Train and deploy YOLO models easily using Ultralytics HUB.
- [ClearML Logging](https://docs.ultralytics.com/integrations/clearml/): Integrate experiment tracking with ClearML.
- [Neural Magic DeepSparse Integration](https://docs.ultralytics.com/integrations/neural-magic/): Accelerate inference with DeepSparse.
- [Comet Logging](https://docs.ultralytics.com/integrations/comet/): Log and visualize experiments using Comet. 🌟 **NEW**

</details>

## <div align="center">🤝 Integrations</div>

Our key integrations with leading AI platforms extend Ultralytics' capabilities, enhancing tasks like dataset labeling, training, visualization, and model management. Explore how Ultralytics works with [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/), [Comet ML](https://docs.ultralytics.com/integrations/comet/), [Roboflow](https://docs.ultralytics.com/integrations/roboflow/), and [Intel OpenVINO](https://docs.ultralytics.com/integrations/openvino/) to streamline your AI workflow.

<br>
<a href="https://www.ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations"></a>
<br>
<br>

<div align="center">
  <a href="https://www.ultralytics.com/hub">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-ultralytics-hub.png" width="10%" alt="Ultralytics HUB logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://docs.ultralytics.com/integrations/weights-biases/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-wb.png" width="10%" alt="Weights & Biases logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://docs.ultralytics.com/integrations/comet/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" alt="Comet ML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://docs.ultralytics.com/integrations/neural-magic/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" alt="NeuralMagic logo"></a>
</div>

|                                                       Ultralytics HUB 🚀                                                        |                                                                 W&B                                                                 |                                                                              Comet ⭐ NEW                                                                              |                                                        Neural Magic                                                         |
| :-----------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
| Streamline YOLO workflows: Label, train, and deploy effortlessly with [Ultralytics HUB](https://hub.ultralytics.com/). Try now! | Track experiments, hyperparameters, and results with [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/). | Free forever, [Comet](https://docs.ultralytics.com/integrations/comet/) lets you save YOLO models, resume training, and interactively visualize and debug predictions. | Run YOLO inference up to 6x faster with [Neural Magic DeepSparse](https://docs.ultralytics.com/integrations/neural-magic/). |

## <div align="center">☁️ Ultralytics HUB</div>

Experience seamless AI development with [Ultralytics HUB](https://hub.ultralytics.com/) ⭐, your all-in-one platform for data visualization, YOLO 🚀 model training, and deployment—no coding required. Convert images into actionable insights and realize your AI projects effortlessly using our advanced platform and intuitive [Ultralytics App](https://www.ultralytics.com/app-install). Begin your journey for **Free** today!

<a align="center" href="https://hub.ultralytics.com/" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png"></a>

## <div align="center">💡 Why YOLOv3?</div>

YOLOv3 marked a significant step in the evolution of real-time object detectors. Its key contributions include:

- **Multi-Scale Predictions:** Detecting objects at three different scales using feature pyramids, improving accuracy for objects of varying sizes, especially small ones.
- **Improved Backbone:** Utilizing Darknet-53, a deeper and more complex network than its predecessor (Darknet-19), enhancing feature extraction capabilities.
- **Class Prediction:** Using logistic classifiers instead of softmax for class predictions, allowing for multi-label classification where an object can belong to multiple categories.

While newer models like [YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [YOLOv8](https://docs.ultralytics.com/models/yolov8/) offer further improvements in speed and accuracy, YOLOv3 remains a foundational model in the field and is still widely used and studied. The table below shows comparisons with later YOLOv5 models for context.

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv3-P5 640 Figure</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>Figure Notes</summary>

- **COCO AP val** denotes [mAP@0.5:0.95](https://www.ultralytics.com/glossary/mean-average-precision-map) metric measured on the 5000-image COCO val2017 dataset over various inference sizes from 256 to 1536. See [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/).
- **GPU Speed** measures average inference time per image on the COCO val2017 dataset using an [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p4/) V100 instance at batch-size 32.
- **EfficientDet** data from [google/automl](https://github.com/google/automl) repository at batch size 8.
- **Reproduce** benchmark results using `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### Pretrained Checkpoints (YOLOv5 Comparison)

This table shows YOLOv5 models trained on the COCO dataset, often used as benchmarks.

| Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)              | 640                   | 28.0                 | 45.7              | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)              | 640                   | 37.4                 | 56.8              | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)              | 640                   | 45.4                 | 64.1              | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)              | 640                   | 49.0                 | 67.3              | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)              | 640                   | 50.7                 | 68.9              | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                 |                       |                      |                   |                              |                               |                                |                    |                        |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)            | 1280                  | 36.0                 | 54.4              | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt)            | 1280                  | 44.8                 | 63.7              | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt)            | 1280                  | 51.3                 | 69.3              | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt)            | 1280                  | 53.7                 | 71.3              | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt)<br>+ [TTA] | 1280<br>1536          | 55.0<br>**55.8**     | 72.7<br>**72.7**  | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |

<details>
  <summary>Table Notes</summary>

- All YOLOv5 checkpoints shown are trained to 300 epochs with default settings. Nano and Small models use `hyp.scratch-low.yaml` hyperparameters, others use `hyp.scratch-high.yaml`. See [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/).
- **mAP<sup>val</sup>** values are for single-model single-scale evaluation on the [COCO val2017 dataset](https://docs.ultralytics.com/datasets/detect/coco/).<br>Reproduce using `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`.
- **Speed** metrics averaged over COCO val images using an [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p4/) instance. NMS times (~1 ms/img) are not included.<br>Reproduce using `python val.py --data coco.yaml --img 640 --task speed --batch 1`.
- **TTA** ([Test Time Augmentation](https://docs.ultralytics.com/modes/val/#inference-augmentation)) includes reflection and scale augmentations.<br>Reproduce using `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`.

</details>

## <div align="center">🖼️ Segmentation</div>

While YOLOv3 primarily focused on object detection, later Ultralytics models like YOLOv5 introduced instance segmentation capabilities. The YOLOv5 [release v7.0](https://github.com/ultralytics/yolov5/releases/v7.0) included segmentation models that achieved state-of-the-art performance. These models are easy to train, validate, and deploy. See the [YOLOv5 Release Notes](https://github.com/ultralytics/yolov5/releases/v7.0) and the [YOLOv5 Segmentation Colab Notebook](https://github.com/ultralytics/yolov5/blob/master/segment/tutorial.ipynb) for more details and tutorials.

<details>
  <summary>Segmentation Checkpoints (YOLOv5)</summary>

<div align="center">
<a align="center" href="https://www.ultralytics.com/yolo" target="_blank">
<img width="800" src="https://user-images.githubusercontent.com/61612323/204180385-84f3aca9-a5e9-43d8-a617-dda7ca12e54a.png"></a>
</div>

YOLOv5 segmentation models were trained on the [COCO-segments dataset](https://docs.ultralytics.com/datasets/segment/coco/) for 300 epochs at an image size of 640 using A100 GPUs. Models were exported to ONNX FP32 for CPU speed tests and TensorRT FP16 for GPU speed tests on Google [Colab Pro](https://colab.research.google.com/signup).

| Model                                                                                      | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Train time<br><sup>300 epochs<br>A100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TRT A100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ------------------------------------------------------------------------------------------ | --------------------- | -------------------- | --------------------- | --------------------------------------------- | ------------------------------ | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt) | 640                   | 27.6                 | 23.4                  | 80:17                                         | **62.7**                       | **1.2**                        | **2.0**            | **7.1**                |
| [YOLOv5s-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt) | 640                   | 37.6                 | 31.7                  | 88:16                                         | 173.3                          | 1.4                            | 7.6                | 26.4                   |
| [YOLOv5m-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt) | 640                   | 45.0                 | 37.1                  | 108:36                                        | 427.0                          | 2.2                            | 22.0               | 70.8                   |
| [YOLOv5l-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt) | 640                   | 49.0                 | 39.9                  | 66:43 (2x)                                    | 857.4                          | 2.9                            | 47.9               | 147.7                  |
| [YOLOv5x-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt) | 640                   | **50.7**             | **41.4**              | 62:56 (3x)                                    | 1579.2                         | 4.5                            | 88.8               | 265.7                  |

- All checkpoints trained for 300 epochs with SGD optimizer (`lr0=0.01`, `weight_decay=5e-5`) at image size 640 using default settings. Runs logged at [W&B YOLOv5_v70_official](https://wandb.ai/glenn-jocher/YOLOv5_v70_official).
- **Accuracy** values are for single-model, single-scale on the COCO dataset.<br>Reproduce with `python segment/val.py --data coco.yaml --weights yolov5s-seg.pt`.
- **Speed** averaged over 100 inference images on a [Colab Pro](https://colab.research.google.com/signup) A100 High-RAM instance (inference only, NMS adds ~1ms/image).<br>Reproduce with `python segment/val.py --data coco.yaml --weights yolov5s-seg.pt --batch 1`.
- **Export** to ONNX (FP32) and TensorRT (FP16) using `export.py`.<br>Reproduce with `python export.py --weights yolov5s-seg.pt --include engine --device 0 --half`.

</details>

<details>
  <summary>Segmentation Usage Examples (YOLOv5) &nbsp;<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/segment/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></summary>

### Train

Train YOLOv5 segmentation models. Use `--data coco128-seg.yaml` for auto-download or manually download COCO-segments with `bash data/scripts/get_coco.sh --train --val --segments` then use `--data coco.yaml`.

```bash
# Single-GPU Training
python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640

# Multi-GPU DDP Training
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3
```

### Val

Validate YOLOv5s-seg mask mAP on the COCO dataset:

```bash
# Download COCO val segments split (780MB, 5000 images)
bash data/scripts/get_coco.sh --val --segments
# Validate performance
python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640
```

### Predict

Use a pretrained YOLOv5m-seg model for prediction:

```bash
# Predict objects and masks in an image
python segment/predict.py --weights yolov5m-seg.pt --data data/images/bus.jpg
```

```python
# Load model via PyTorch Hub (Note: Segmentation inference might require specific handling)
# model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5m-seg.pt")
```

| ![Zidane Segmentation Example](https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg) | ![Bus Segmentation Example](https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg) |
| ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |

### Export

Export a YOLOv5s-seg model to ONNX and TensorRT formats:

```bash
# Export model for deployment
python export.py --weights yolov5s-seg.pt --include onnx engine --img 640 --device 0
```

</details>

## <div align="center">🏷️ Classification</div>

Similar to segmentation, image classification capabilities were formally introduced in later Ultralytics YOLO versions, specifically YOLOv5 [release v6.2](https://github.com/ultralytics/yolov5/releases/v6.2). These models allow for training, validation, and deployment for classification tasks. Check the [YOLOv5 Release Notes](https://github.com/ultralytics/yolov5/releases/v6.2) and the [YOLOv5 Classification Colab Notebook](https://github.com/ultralytics/yolov5/blob/master/classify/tutorial.ipynb) for detailed information and examples.

<details>
  <summary>Classification Checkpoints (YOLOv5 & Others)</summary>

<br>

YOLOv5-cls models were trained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) for 90 epochs using a 4xA100 instance. ResNet and EfficientNet models were trained alongside for comparison using the same settings. Models were exported to ONNX FP32 (CPU speed) and TensorRT FP16 (GPU speed) and tested on Google [Colab Pro](https://colab.research.google.com/signup).

| Model                                                                                              | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Training<br><sup>90 epochs<br>4xA100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TensorRT V100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@224 (B) |
| -------------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | -------------------------------------------- | ------------------------------ | ----------------------------------- | ------------------ | ---------------------- |
| [YOLOv5n-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-cls.pt)         | 224                   | 64.6             | 85.4             | 7:59                                         | **3.3**                        | **0.5**                             | **2.5**            | **0.5**                |
| [YOLOv5s-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt)         | 224                   | 71.5             | 90.2             | 8:09                                         | 6.6                            | 0.6                                 | 5.4                | 1.4                    |
| [YOLOv5m-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-cls.pt)         | 224                   | 75.9             | 92.9             | 10:06                                        | 15.5                           | 0.9                                 | 12.9               | 3.9                    |
| [YOLOv5l-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-cls.pt)         | 224                   | 78.0             | 94.0             | 11:56                                        | 26.9                           | 1.4                                 | 26.5               | 8.5                    |
| [YOLOv5x-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-cls.pt)         | 224                   | **79.0**         | **94.4**         | 15:04                                        | 54.3                           | 1.8                                 | 48.1               | 15.9                   |
|                                                                                                    |                       |                  |                  |                                              |                                |                                     |                    |                        |
| [ResNet18](https://github.com/ultralytics/yolov5/releases/download/v7.0/resnet18.pt)               | 224                   | 70.3             | 89.5             | **6:47**                                     | 11.2                           | 0.5                                 | 11.7               | 3.7                    |
| [ResNet34](https://github.com/ultralytics/yolov5/releases/download/v7.0/resnet34.pt)               | 224                   | 73.9             | 91.8             | 8:33                                         | 20.6                           | 0.9                                 | 21.8               | 7.4                    |
| [ResNet50](https://github.com/ultralytics/yolov5/releases/download/v7.0/resnet50.pt)               | 224                   | 76.8             | 93.4             | 11:10                                        | 23.4                           | 1.0                                 | 25.6               | 8.5                    |
| [ResNet101](https://github.com/ultralytics/yolov5/releases/download/v7.0/resnet101.pt)             | 224                   | 78.5             | 94.3             | 17:10                                        | 42.1                           | 1.9                                 | 44.5               | 15.9                   |
|                                                                                                    |                       |                  |                  |                                              |                                |                                     |                    |                        |
| [EfficientNet_b0](https://github.com/ultralytics/yolov5/releases/download/v7.0/efficientnet_b0.pt) | 224                   | 75.1             | 92.4             | 13:03                                        | 12.5                           | 1.3                                 | 5.3                | 1.0                    |
| [EfficientNet_b1](https://github.com/ultralytics/yolov5/releases/download/v7.0/efficientnet_b1.pt) | 224                   | 76.4             | 93.2             | 17:04                                        | 14.9                           | 1.6                                 | 7.8                | 1.5                    |
| [EfficientNet_b2](https://github.com/ultralytics/yolov5/releases/download/v7.0/efficientnet_b2.pt) | 224                   | 76.6             | 93.4             | 17:10                                        | 15.9                           | 1.6                                 | 9.1                | 1.7                    |
| [EfficientNet_b3](https://github.com/ultralytics/yolov5/releases/download/v7.0/efficientnet_b3.pt) | 224                   | 77.7             | 94.0             | 19:19                                        | 18.9                           | 1.9                                 | 12.2               | 2.4                    |

<details>
  <summary>Table Notes (click to expand)</summary>

- All checkpoints trained for 90 epochs with SGD optimizer (`lr0=0.001`, `weight_decay=5e-5`) at image size 224 using default settings. Runs logged at [W&B YOLOv5-Classifier-v6-2](https://wandb.ai/glenn-jocher/YOLOv5-Classifier-v6-2).
- **Accuracy** values are for single-model, single-scale on the [ImageNet-1k dataset](https://www.image-net.org/index.php).<br>Reproduce with `python classify/val.py --data ../datasets/imagenet --img 224`.
- **Speed** averaged over 100 inference images using a Google [Colab Pro](https://colab.research.google.com/signup) V100 High-RAM instance.<br>Reproduce with `python classify/val.py --data ../datasets/imagenet --img 224 --batch 1`.
- **Export** to ONNX (FP32) and TensorRT (FP16) using `export.py`.<br>Reproduce with `python export.py --weights yolov5s-cls.pt --include engine onnx --imgsz 224`.

</details>
</details>

<details>
  <summary>Classification Usage Examples (YOLOv5) &nbsp;<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/classify/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></summary>

### Train

Train YOLOv5 classification models. Datasets like [MNIST](https://docs.ultralytics.com/datasets/classify/mnist/), [Fashion-MNIST](https://docs.ultralytics.com/datasets/classify/fashion-mnist/), [CIFAR10](https://docs.ultralytics.com/datasets/classify/cifar10/), [CIFAR100](https://docs.ultralytics.com/datasets/classify/cifar100/), [Imagenette](https://docs.ultralytics.com/datasets/classify/imagenette/), [Imagewoof](https://docs.ultralytics.com/datasets/classify/imagewoof/), and [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) can be auto-downloaded using the `--data` argument (e.g., `--data mnist`).

```bash
# Single-GPU Training on CIFAR-100
python classify/train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP Training on ImageNet
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

### Val

Validate YOLOv5m-cls accuracy on the ImageNet-1k validation set:

```bash
# Download ImageNet validation split (6.3G, 50000 images)
bash data/scripts/get_imagenet.sh --val
# Validate model accuracy
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224
```

### Predict

Use a pretrained YOLOv5s-cls model to classify an image:

```bash
# Classify an image
python classify/predict.py --weights yolov5s-cls.pt --data data/images/bus.jpg
```

```python
# Load model via PyTorch Hub
# model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s-cls.pt")
```

### Export

Export trained classification models (YOLOv5s-cls, ResNet50, EfficientNet-B0) to ONNX and TensorRT formats:

```bash
# Export models for deployment
python export.py --weights yolov5s-cls.pt resnet50.pt efficientnet_b0.pt --include onnx engine --img 224
```

</details>

## <div align="center">🌎 Environments</div>

Get started quickly with our verified environments. Click the icons below for setup details.

<div align="center">
  <a href="https://docs.ultralytics.com/integrations/paperspace/">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-gradient.png" width="10%" alt="Run on Gradient"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/google-colab/">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-colab-small.png" width="10%" alt="Open In Colab"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/kaggle/">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-kaggle-small.png" width="10%" alt="Open In Kaggle"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/guides/docker-quickstart/">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-docker-small.png" width="10%" alt="Docker Image"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/amazon-sagemaker/">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-aws-small.png" width="10%" alt="AWS Marketplace"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/google-cloud/">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-gcp-small.png" width="10%" alt="GCP Quickstart"/></a>
</div>

## <div align="center">💖 Contribute</div>

We welcome your contributions! Making contributions to YOLOv3 should be easy and transparent. Please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for instructions on getting started. We also encourage you to fill out the [Ultralytics Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to share your feedback. A huge thank you to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## <div align="center">📜 License</div>

Ultralytics provides two licensing options to suit different needs:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts, this [OSI-approved](https://opensource.org/license/agpl-v3) open-source license encourages open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/yolov3/blob/master/LICENSE) file for details.
- **Enterprise License**: Tailored for commercial applications, this license allows the integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. For commercial use cases, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## <div align="center">📞 Contact</div>

For bug reports and feature requests related to YOLOv3, please visit [GitHub Issues](https://github.com/ultralytics/yolov3/issues). For general questions, discussions, and community interaction, join our [Discord server](https://discord.com/invite/ultralytics)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
