<div align="center">
  <p>
    <a href="https://www.ultralytics.com/yolo" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png" alt="Ultralytics YOLOv3 banner"></a>
  </p>

[English](https://docs.ultralytics.com/en) | [한국어](https://docs.ultralytics.com/ko) | [日本語](https://docs.ultralytics.com/ja) | [Русский](https://docs.ultralytics.com/ru) | [Deutsch](https://docs.ultralytics.com/de) | [Français](https://docs.ultralytics.com/fr) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt) | [Türkçe](https://docs.ultralytics.com/tr) | [Tiếng Việt](https://docs.ultralytics.com/vi) | [العربية](https://docs.ultralytics.com/ar)

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

Ultralytics YOLOv3 是由 [Ultralytics](https://www.ultralytics.com/) 开发的一款强大而高效的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型。该实现基于 [PyTorch](https://pytorch.org/) 框架，建立在原始 YOLOv3 架构之上。与之前的版本相比，YOLOv3 以其在[目标检测](https://www.ultralytics.com/glossary/object-detection)速度和准确性方面的显著改进而闻名。它融合了广泛研究和开发的见解与最佳实践，使其成为各种视觉 AI 任务的可靠选择。

我们希望这里的资源能帮助您充分利用 YOLOv3。请浏览 [Ultralytics 文档](https://docs.ultralytics.com/)获取详细信息（注意：特定的 YOLOv3 文档可能有限，请参考通用的 YOLO 原则），在 [GitHub](https://github.com/ultralytics/yolov5/issues/new/choose) 上提出问题以获得支持，并加入我们的 [Discord 社区](https://discord.com/invite/ultralytics)进行提问和讨论！

如需申请企业许可证，请填写 [Ultralytics 许可](https://www.ultralytics.com/license)表格。

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>
</div>
<br>

## 🚀 YOLO11：下一代进化

我们激动地宣布推出 **Ultralytics YOLO11** 🚀，这是我们最先进（SOTA）视觉模型的最新进展！YOLO11 现已在 [Ultralytics YOLO GitHub 仓库](https://github.com/ultralytics/ultralytics)发布，它继承了我们在速度、精度和易用性方面的传统。无论您是处理[目标检测](https://docs.ultralytics.com/tasks/detect/)、[实例分割](https://docs.ultralytics.com/tasks/segment/)、[姿态估计](https://docs.ultralytics.com/tasks/pose/)、[图像分类](https://docs.ultralytics.com/tasks/classify/)还是[旋转目标检测 (OBB)](https://docs.ultralytics.com/tasks/obb/)，YOLO11 都能提供在各种应用中脱颖而出所需的性能和多功能性。

立即开始，释放 YOLO11 的全部潜力！访问 [Ultralytics 文档](https://docs.ultralytics.com/)获取全面的指南和资源：

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics)

```bash
# 安装 ultralytics 包
pip install ultralytics
```

<div align="center">
  <a href="https://www.ultralytics.com/yolo" target="_blank">
  <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png" alt="Ultralytics YOLO Performance Comparison"></a>
</div>

## 📚 文档

请参阅 [Ultralytics 文档](https://docs.ultralytics.com/models/yolov3/)，了解使用 Ultralytics 框架进行训练、测试和部署的完整文档。虽然特定的 YOLOv3 文档可能有限，但通用原则仍然适用。请参阅下方为 YOLOv3 概念改编的快速入门示例。

<details open>
<summary>安装</summary>

克隆仓库并在 [**Python>=3.8.0**](https://www.python.org/) 环境中从 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) 安装依赖项。确保您已安装 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。（注意：此仓库最初是 YOLOv5 的，依赖项应兼容，但建议针对 YOLOv3 进行专门测试）。

```bash
# 克隆 YOLOv3 仓库
git clone https://github.com/ultralytics/yolov3

# 导航到克隆的目录
cd yolov3

# 安装所需的包
pip install -r requirements.txt
```

</details>

<details open>
<summary>使用 PyTorch Hub 进行推理</summary>

通过 [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/) 使用 YOLOv3 进行推理。[模型](https://github.com/ultralytics/yolov5/tree/master/models)如 `yolov3.pt`、`yolov3-spp.pt`、`yolov3-tiny.pt` 可以被加载。

```python
import torch

# 加载 YOLOv3 模型（例如，yolov3, yolov3-spp）
model = torch.hub.load("ultralytics/yolov3", "yolov3", pretrained=True)  # 指定 'yolov3' 或其他变体

# 定义输入图像源（URL、本地文件、PIL 图像、OpenCV 帧、numpy 数组或列表）
img = "https://ultralytics.com/images/zidane.jpg"  # 示例图像

# 执行推理
results = model(img)

# 处理结果（选项：.print(), .show(), .save(), .crop(), .pandas()）
results.print()  # 将结果打印到控制台
results.show()  # 在窗口中显示结果
results.save()  # 将结果保存到 runs/detect/exp
```

</details>

<details>
<summary>使用 detect.py 进行推理</summary>

`detect.py` 脚本在各种来源上运行推理。使用 `--weights yolov3.pt` 或其他 YOLOv3 变体。它会自动下载模型并将结果保存到 `runs/detect`。

```bash
# 使用 yolov3-tiny 和网络摄像头运行推理
python detect.py --weights yolov3-tiny.pt --source 0

# 使用 yolov3 在本地图像文件上运行推理
python detect.py --weights yolov3.pt --source img.jpg

# 使用 yolov3-spp 在本地视频文件上运行推理
python detect.py --weights yolov3-spp.pt --source vid.mp4

# 在屏幕截图上运行推理
python detect.py --weights yolov3.pt --source screen

# 在图像目录上运行推理
python detect.py --weights yolov3.pt --source path/to/images/

# 在列出图像路径的文本文件上运行推理
python detect.py --weights yolov3.pt --source list.txt

# 在列出流 URL 的文本文件上运行推理
python detect.py --weights yolov3.pt --source list.streams

# 使用 glob 模式对图像运行推理
python detect.py --weights yolov3.pt --source 'path/to/*.jpg'

# 在 YouTube 视频 URL 上运行推理
python detect.py --weights yolov3.pt --source 'https://youtu.be/LNwODJXcvt4'

# 在 RTSP、RTMP 或 HTTP 流上运行推理
python detect.py --weights yolov3.pt --source 'rtsp://example.com/media.mp4'
```

</details>

<details>
<summary>训练</summary>

以下命令展示了如何在 [COCO 数据集](https://docs.ultralytics.com/datasets/detect/coco/)上训练 YOLOv3 模型。模型和数据集会自动下载。请使用您硬件允许的最大 `--batch-size`。

```bash
# 在 COCO 上训练 YOLOv3-tiny 300 个周期（示例设置）
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3-tiny.yaml --batch-size 64

# 在 COCO 上训练 YOLOv3 300 个周期（示例设置）
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3.yaml --batch-size 32

# 在 COCO 上训练 YOLOv3-SPP 300 个周期（示例设置）
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3-spp.yaml --batch-size 16
```

</details>

<details open>
<summary>教程</summary>

注意：这些教程主要使用 YOLOv5 示例，但其原理通常适用于 Ultralytics 框架内的 YOLOv3。

- **[训练自定义数据](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)** 🚀 **推荐**：学习如何在您自己的数据集上训练模型。
- **[获得最佳训练结果的技巧](https://docs.ultralytics.com/guides/model-training-tips/)** ☘️：利用专家技巧提高模型性能。
- **[多 GPU 训练](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training/)**：使用多个 GPU 加速训练。
- **[PyTorch Hub 集成](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/)** 🌟 **新增**：使用 PyTorch Hub 轻松加载模型。
- **[模型导出 (TFLite, ONNX, CoreML, TensorRT)](https://docs.ultralytics.com/yolov5/tutorials/model_export/)** 🚀：将您的模型转换为各种部署格式。
- **[NVIDIA Jetson 部署](https://docs.ultralytics.com/yolov5/tutorials/running_on_jetson_nano/)** 🌟 **新增**：在 NVIDIA Jetson 设备上部署模型。
- **[测试时增强 (TTA)](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/)**：使用 TTA 提高预测准确性。
- **[模型集成](https://docs.ultralytics.com/yolov5/tutorials/model_ensembling/)**：组合多个模型以获得更好的性能。
- **[模型剪枝/稀疏化](https://docs.ultralytics.com/yolov5/tutorials/model_pruning_and_sparsity/)**：优化模型的大小和速度。
- **[超参数进化](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution/)**：自动找到最佳训练超参数。
- **[使用冻结层的迁移学习](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/)**：高效地将预训练模型应用于新任务。
- **[架构总结](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/)** 🌟 **新增**：理解模型架构（侧重于 YOLOv3 原理）。
- **[Ultralytics HUB 训练](https://www.ultralytics.com/hub)** 🚀 **推荐**：使用 Ultralytics HUB 训练和部署 YOLO 模型。
- **[ClearML 日志记录](https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration/)**：与 ClearML 集成以进行实验跟踪。
- **[Neural Magic DeepSparse 集成](https://docs.ultralytics.com/yolov5/tutorials/neural_magic_pruning_quantization/)**：使用 DeepSparse 加速推理。
- **[Comet 日志记录](https://docs.ultralytics.com/yolov5/tutorials/comet_logging_integration/)** 🌟 **新增**：使用 Comet ML 记录实验。

</details>

## 🧩 集成

我们与领先 AI 平台的关键集成扩展了 Ultralytics 产品的功能，增强了诸如数据集标注、训练、可视化和模型管理等任务。了解 Ultralytics 如何与 [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/)、[Comet ML](https://docs.ultralytics.com/integrations/comet/)、[Roboflow](https://docs.ultralytics.com/integrations/roboflow/) 和 [Intel OpenVINO](https://docs.ultralytics.com/integrations/openvino/) 等合作伙伴协作，优化您的 AI 工作流程。在 [Ultralytics 集成](https://docs.ultralytics.com/integrations/) 探索更多信息。

<a href="https://docs.ultralytics.com/integrations/" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations">
</a>
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
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" alt="Neural Magic logo"></a>
</div>

|                                            Ultralytics HUB 🌟                                            |                                              Weights & Biases                                               |                                                           Comet                                                           |                                                      Neural Magic                                                       |
| :------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
| 简化 YOLO 工作流程：使用 [Ultralytics HUB](https://hub.ultralytics.com) 轻松标注、训练和部署。立即试用！ | 使用 [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/) 跟踪实验、超参数和结果。 | 永久免费，[Comet ML](https://docs.ultralytics.com/integrations/comet/) 让您保存 YOLO 模型、恢复训练并交互式地可视化预测。 | 使用 [Neural Magic DeepSparse](https://docs.ultralytics.com/integrations/neural-magic/) 将 YOLO 推理速度提高多达 6 倍。 |

## ⭐ Ultralytics HUB

通过 [Ultralytics HUB](https://www.ultralytics.com/hub) ⭐ 体验无缝的 AI 开发，这是构建、训练和部署计算机视觉模型的终极平台。无需编写任何代码，即可可视化数据集、训练 YOLOv3、YOLOv5 和 YOLOv8 🚀 模型，并将它们部署到实际应用中。使用我们尖端的工具和用户友好的 [Ultralytics App](https://www.ultralytics.com/app-install)，将图像转化为可操作的见解。立即开始您的**免费**旅程！

<a align="center" href="https://www.ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png" alt="Ultralytics HUB Platform Screenshot"></a>

## 🤔 为何选择 YOLOv3？

YOLOv3 在发布时代表了实时目标检测领域的一大进步。其主要优势包括：

- **提高准确性：** 与 YOLOv2 相比，对小目标的检测效果更好。
- **多尺度预测：** 在三个不同尺度上检测目标，提高了对各种尺寸目标的性能。
- **类别预测：** 使用逻辑分类器预测目标类别，而不是 softmax，允许进行多标签分类。
- **特征提取器：** 与 YOLOv2 中使用的 Darknet-19 相比，使用了更深的网络（Darknet-53）。

虽然像 YOLOv5 和 YOLO11 这样的更新模型提供了进一步的改进，但 YOLOv3 仍然是一个坚实且被广泛理解的基准，由 Ultralytics 在 PyTorch 中高效实现。

## ☁️ 环境

使用我们预配置的环境快速开始。点击下面的图标查看设置详情。

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

## 🤝 贡献

我们欢迎您的贡献！让 YOLO 模型易于使用且高效是社区共同努力的目标。请参阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)开始。通过 [Ultralytics 调查](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享您的反馈。感谢所有为使 Ultralytics YOLO 变得更好而做出贡献的人！

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/yolov5/graphs/contributors)

## 📜 许可证

Ultralytics 提供两种许可选项以满足不同需求：

- **AGPL-3.0 许可证**：一种经 [OSI 批准](https://opensource.org/license/agpl-v3)的开源许可证，非常适合学术研究、个人项目和测试。它促进开放合作和知识共享。详情请参阅 [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) 文件。
- **企业许可证**：专为商业应用量身定制，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品和服务中，绕过 AGPL-3.0 的开源要求。对于商业用途，请通过 [Ultralytics 许可](https://www.ultralytics.com/license)与我们联系。

## 📧 联系

有关 Ultralytics YOLO 实现的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/ultralytics/yolov5/issues)。有关一般问题、讨论和社区支持，请加入我们的 [Discord 服务器](https://discord.com/invite/ultralytics)！

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
