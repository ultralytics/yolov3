<div align="center">
  <p>
    <a href="https://www.ultralytics.com/" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png" alt="Ultralytics YOLOv3 banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [العربية](https://docs.ultralytics.com/ar/)

<div>
    <a href="https://github.com/ultralytics/yolov3/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov3/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv3 CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv3 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/yolov3"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov3?logo=docker" alt="Docker Pulls"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </div>
  <br>

[Ultralytics](https://www.ultralytics.com/) YOLOv3 is a PyTorch implementation of the YOLOv3 (You Only Look Once, version 3) real-time [object detection](https://docs.ultralytics.com/tasks/detect/) model. YOLOv3 frames detection as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one forward pass — making it fast, accurate, and straightforward to train and deploy.

This repository packages the three classic YOLOv3 detection models — **YOLOv3**, **YOLOv3-SPP**, and **YOLOv3-tiny** — with training, validation, inference, and export tooling, and reuses shared utilities from the [`ultralytics`](https://github.com/ultralytics/ultralytics) package.

Find detailed guidance in the [Ultralytics YOLOv3 Docs](https://docs.ultralytics.com/models/yolov3/). Get support via [GitHub Issues](https://github.com/ultralytics/yolov3/issues/new/choose), and join the conversation on [Discord](https://discord.com/invite/ultralytics), [Reddit](https://www.reddit.com/r/ultralytics/), and the [Ultralytics Forums](https://community.ultralytics.com/).

For commercial use, request an Enterprise License at [Ultralytics Licensing](https://www.ultralytics.com/license).

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>
</div>
<br>

## 📚 Documentation

See the [Ultralytics YOLOv3 Docs](https://docs.ultralytics.com/models/yolov3/) for full documentation. The quickstart examples below cover installation, inference, and training with this repository.

<details open>
<summary>Install</summary>

Clone the repository and install the dependencies from `requirements.txt` in a [**Python>=3.8.0**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
# Clone the YOLOv3 repository
git clone https://github.com/ultralytics/yolov3

# Navigate to the cloned directory
cd yolov3

# Install required packages
pip install -r requirements.txt
```

</details>

<details open>
<summary>Inference with PyTorch Hub</summary>

Load YOLOv3 directly through [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/). Weights download automatically on first use.

```python
import torch

# Load a YOLOv3 model (choices: 'yolov3', 'yolov3_spp', 'yolov3_tiny')
model = torch.hub.load("ultralytics/yolov3", "yolov3", pretrained=True)

# Run inference on an image (local file, URL, PIL image, OpenCV frame, or numpy array)
results = model("https://ultralytics.com/images/zidane.jpg")

# Inspect the results
results.print()  # print detections to the console
results.show()  # display the annotated image
results.save()  # save the annotated image to runs/detect/exp
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a wide range of sources, downloading models automatically and saving results to `runs/detect`.

```bash
python detect.py --weights yolov3.pt --source 0                              # webcam
python detect.py --weights yolov3.pt --source img.jpg                        # image
python detect.py --weights yolov3.pt --source vid.mp4                        # video
python detect.py --weights yolov3.pt --source screen                         # screenshot
python detect.py --weights yolov3.pt --source path/                          # directory
python detect.py --weights yolov3.pt --source 'path/*.jpg'                   # glob
python detect.py --weights yolov3.pt --source 'https://youtu.be/LNwODJXcvt4' # YouTube
python detect.py --weights yolov3.pt --source 'rtsp://example.com/media.mp4' # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Train YOLOv3 on the [COCO](https://docs.ultralytics.com/datasets/detect/coco/) dataset. Models and datasets download automatically. Use the largest `--batch-size` your hardware allows.

```bash
# Train YOLOv3-tiny
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3-tiny.yaml --batch-size 64

# Train YOLOv3
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3.yaml --batch-size 32

# Train YOLOv3-SPP
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3-spp.yaml --batch-size 16
```

Validate accuracy with `python val.py --weights yolov3.pt --data coco.yaml`, and export to other formats (TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TFLite, and more) with `python export.py --weights yolov3.pt --include onnx`.

</details>

<details>
<summary>Tutorials</summary>

These guides cover the shared Ultralytics training framework and apply to YOLOv3:

- [Train Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) — train on your own dataset.
- [Tips for Best Training Results](https://docs.ultralytics.com/guides/model-training-tips/) — get the most out of training.
- [Multi-GPU Training](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training/) — scale training across GPUs.
- [PyTorch Hub Loading](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/) — load models programmatically.
- [Model Export](https://docs.ultralytics.com/yolov5/tutorials/model_export/) — deploy to ONNX, TensorRT, CoreML, TFLite, and more.
- [Test-Time Augmentation (TTA)](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/) — improve accuracy at inference.
- [Model Ensembling](https://docs.ultralytics.com/yolov5/tutorials/model_ensembling/) — combine models for better results.
- [Hyperparameter Evolution](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution/) — tune hyperparameters automatically.
- [Transfer Learning with Frozen Layers](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/) — adapt pretrained models efficiently.

</details>

## 🧠 Architecture

YOLOv3 builds detection on a few core ideas that make it both accurate and fast:

- **Darknet-53 backbone** — a 53-layer convolutional feature extractor with residual (skip) connections, deeper and more accurate than the Darknet-19 backbone of YOLOv2 while staying efficient.
- **Multi-scale detection** — predictions are made at three feature-map scales using a feature-pyramid-style design (upsampling and concatenating earlier feature maps), so the model detects small, medium, and large objects well.
- **Anchor boxes** — boxes are predicted relative to dimension-cluster anchor priors, with three anchors per scale (nine total) and a sigmoid offset parameterization for stable training.
- **Independent class prediction** — each class is scored with an independent logistic classifier rather than a softmax, so one box can carry multiple non-mutually-exclusive labels.

The repository ships three variants of this architecture:

- **YOLOv3** — the full Darknet-53 model; the best balance of speed and accuracy.
- **YOLOv3-SPP** — adds a Spatial Pyramid Pooling block that pools features at multiple kernel sizes for a larger effective receptive field and a small accuracy gain.
- **YOLOv3-tiny** — a compact backbone with detection at two scales, optimized for CPU and edge devices where speed matters most.

Models are defined declaratively in [`models/*.yaml`](https://github.com/ultralytics/yolov3/tree/master/models) and built by `parse_model()` in `models/yolo.py`, so the architecture can be inspected and modified without writing Python.

## 🏋️ Pretrained Checkpoints

All three models are trained on [COCO](https://docs.ultralytics.com/datasets/detect/coco/) (80 classes) and download automatically from the [YOLOv3 release assets](https://github.com/ultralytics/yolov3/releases) on first use.

| Model                                                                                           | Description                                                                       |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [yolov3-tiny.pt](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-tiny.pt) | Lightweight two-scale model — the fastest option, ideal for CPU and edge devices. |
| [yolov3.pt](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt)           | The original Darknet-53 model — a strong balance of speed and accuracy.           |
| [yolov3-spp.pt](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-spp.pt)   | Adds Spatial Pyramid Pooling for a larger receptive field and improved accuracy.  |

## 🧩 Integrations

Ultralytics integrates with leading AI platforms to extend dataset labeling, training, visualization, and model management. Explore how partners such as [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/), [Comet ML](https://docs.ultralytics.com/integrations/comet/), [Roboflow](https://docs.ultralytics.com/integrations/roboflow/), and [Intel OpenVINO](https://docs.ultralytics.com/integrations/openvino/) can streamline your workflow at [Ultralytics Integrations](https://docs.ultralytics.com/integrations/).

<a href="https://docs.ultralytics.com/integrations/" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations">
</a>

## 🤔 Why YOLOv3?

YOLOv3 was a landmark in real-time object detection and remains a dependable, well-understood baseline:

- **Real-time single-stage detection** — one forward pass produces all detections, with no separate region-proposal stage.
- **Strong across object sizes** — multi-scale predictions handle small, medium, and large objects.
- **Multi-label friendly** — independent logistic classifiers allow overlapping class labels.
- **Simple and portable** — a fully-convolutional design that trains and exports cleanly to many deployment formats.

For the broader family of Ultralytics YOLO models, see the [Ultralytics repository](https://github.com/ultralytics/ultralytics).

## ☁️ Environments

Get started quickly with pre-configured environments. Click an icon below for setup details.

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
  <a href="https://docs.ultralytics.com/integrations/google-colab/">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-gcp-small.png" width="10%" alt="GCP Quickstart"/></a>
</div>

## 🤝 Contribute

Contributions are welcome! Please see the [Contributing Guide](https://docs.ultralytics.com/help/contributing/) to get started, and share your feedback through the [Ultralytics Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). Thank you to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/yolov3/graphs/contributors)

## 📜 License

Ultralytics offers two licensing options:

- **AGPL-3.0 License**: An [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license ideal for research and collaboration. See the [LICENSE](https://github.com/ultralytics/yolov3/blob/master/LICENSE) file for details.
- **Enterprise License**: For commercial use, this license allows integration of Ultralytics software and models into commercial products without AGPL-3.0 obligations. Contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📧 Contact

For bug reports and feature requests, please use [GitHub Issues](https://github.com/ultralytics/yolov3/issues). For questions and discussion, join our [Discord](https://discord.com/invite/ultralytics), [Reddit](https://www.reddit.com/r/ultralytics/), and the [Ultralytics Forums](https://community.ultralytics.com/).

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
