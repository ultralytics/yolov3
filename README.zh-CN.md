<div align="center">
  <p>
    <a href="https://www.ultralytics.com/" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png" alt="Ultralytics YOLOv3 banner"></a>
  </p>

[English](https://docs.ultralytics.com/) | [한국어](https://docs.ultralytics.com/ko) | [日本語](https://docs.ultralytics.com/ja) | [Русский](https://docs.ultralytics.com/ru) | [Deutsch](https://docs.ultralytics.com/de) | [Français](https://docs.ultralytics.com/fr) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt) | [Türkçe](https://docs.ultralytics.com/tr) | [Tiếng Việt](https://docs.ultralytics.com/vi) | [العربية](https://docs.ultralytics.com/ar)

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

[Ultralytics](https://www.ultralytics.com/) YOLOv3 是 YOLOv3（You Only Look Once，第 3 版）实时[目标检测](https://docs.ultralytics.com/tasks/detect)模型的 PyTorch 实现。YOLOv3 将检测视为单一回归问题，在一次前向传播中直接从整幅图像预测边界框和类别概率，因而快速、准确，且易于训练和部署。

本仓库提供三款经典的 YOLOv3 检测模型——**YOLOv3**、**YOLOv3-SPP** 和 **YOLOv3-tiny**——并配套训练、验证、推理和导出工具，同时复用 [`ultralytics`](https://github.com/ultralytics/ultralytics) 包中的共享工具。

详细指南请参阅 [Ultralytics YOLOv3 文档](https://docs.ultralytics.com/models/yolov3)。如需支持，请提交 [GitHub Issue](https://github.com/ultralytics/yolov3/issues/new/choose)，并欢迎加入 [Discord](https://discord.com/invite/ultralytics)、[Reddit](https://www.reddit.com/r/ultralytics/) 和 [Ultralytics 论坛](https://community.ultralytics.com/) 参与讨论。

如需商业使用，请通过 [Ultralytics 许可](https://www.ultralytics.com/license) 申请企业许可证。

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

## 📚 文档

完整文档请参阅 [Ultralytics YOLOv3 文档](https://docs.ultralytics.com/models/yolov3)。以下快速入门示例涵盖本仓库的安装、推理和训练。

<details open>
<summary>安装</summary>

在 [**Python>=3.8.0**](https://www.python.org/) 环境（已安装 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)）中克隆仓库并从 `requirements.txt` 安装依赖。

```bash
# 克隆 YOLOv3 仓库
git clone https://github.com/ultralytics/yolov3

# 进入目录
cd yolov3

# 安装依赖
pip install -r requirements.txt
```

</details>

<details open>
<summary>使用 PyTorch Hub 进行推理</summary>

通过 [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) 可直接加载 YOLOv3，权重在首次使用时自动下载。

```python
import torch

# 加载 YOLOv3 模型 (可选: 'yolov3'、'yolov3_spp'、'yolov3_tiny')
model = torch.hub.load("ultralytics/yolov3", "yolov3", pretrained=True)

# 对图像进行推理 (本地文件、URL、PIL 图像、OpenCV 帧或 numpy 数组)
results = model("https://ultralytics.com/images/zidane.jpg")

# 查看结果
results.print()  # 在控制台打印检测结果
results.show()  # 显示标注后的图像
results.save()  # 将标注图像保存到 runs/detect/exp
```

</details>

<details>
<summary>使用 detect.py 进行推理</summary>

`detect.py` 支持多种输入源推理，自动下载模型并将结果保存至 `runs/detect`。

```bash
python detect.py --weights yolov3.pt --source 0                              # 摄像头
python detect.py --weights yolov3.pt --source img.jpg                        # 图像
python detect.py --weights yolov3.pt --source vid.mp4                        # 视频
python detect.py --weights yolov3.pt --source screen                         # 屏幕截图
python detect.py --weights yolov3.pt --source path/                          # 目录
python detect.py --weights yolov3.pt --source 'path/*.jpg'                   # glob 匹配
python detect.py --weights yolov3.pt --source 'rtsp://example.com/media.mp4' # RTSP、RTMP、HTTP 流
```

</details>

<details>
<summary>训练</summary>

在 [COCO](https://docs.ultralytics.com/datasets/detect/coco) 数据集上训练 YOLOv3。模型和数据集会自动下载。请根据硬件选择尽可能大的 `--batch-size`。

```bash
# 训练 YOLOv3-tiny
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3-tiny.yaml --batch-size 64

# 训练 YOLOv3
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3.yaml --batch-size 32

# 训练 YOLOv3-SPP
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov3-spp.yaml --batch-size 16
```

使用 `python val.py --weights yolov3.pt --data coco.yaml` 验证精度，并使用 `python export.py --weights yolov3.pt --include onnx` 导出为其他格式（TorchScript、ONNX、OpenVINO、TensorRT、CoreML、PaddlePaddle）。

</details>

<details>
<summary>教程</summary>

以下指南基于 Ultralytics 通用训练框架，同样适用于 YOLOv3：

- [训练自定义数据](https://docs.ultralytics.com/modes/train) — 在自有数据集上训练。
- [最佳训练技巧](https://docs.ultralytics.com/guides/model-training-tips) — 充分发挥训练效果。
- [多 GPU 训练](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training) — 跨 GPU 扩展训练。
- [PyTorch Hub 加载](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) — 以编程方式加载模型。
- [模型导出](https://docs.ultralytics.com/modes/export) — 部署到 ONNX、TensorRT、CoreML 等。
- [测试时增强 (TTA)](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation) — 提升推理精度。
- [模型集成](https://docs.ultralytics.com/yolov5/tutorials/model_ensembling) — 融合多个模型以获得更好结果。
- [超参数调优](https://docs.ultralytics.com/guides/hyperparameter-tuning) — 自动调优超参数。
- [冻结层迁移学习](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers) — 高效迁移预训练模型。

</details>

## 🧠 架构

YOLOv3 的检测能力建立在几个核心思想之上，使其兼具准确与高效：

- **Darknet-53 主干** — 53 层卷积特征提取网络，带有残差（跳跃）连接；比 YOLOv2 的 Darknet-19 主干更深、更准确，同时保持高效。
- **多尺度检测** — 采用类特征金字塔的设计（上采样并拼接较早的特征图），在三个特征图尺度上进行预测，从而很好地检测大、中、小目标。
- **锚框** — 边界框相对于由维度聚类得到的锚框先验进行预测，每个尺度三个锚框（共九个），并使用 sigmoid 偏移参数化以稳定训练。
- **独立类别预测** — 每个类别使用独立的逻辑分类器而非 softmax 进行打分，因此单个框可携带多个互不排斥的标签。

本仓库提供该架构的三种变体：

- **YOLOv3** — 完整的 Darknet-53 模型；速度与精度的最佳平衡。
- **YOLOv3-SPP** — 增加空间金字塔池化模块，在多个核大小上池化特征，扩大有效感受野并小幅提升精度。
- **YOLOv3-tiny** — 紧凑主干、两个尺度检测，针对以速度为先的 CPU 和边缘设备优化。

模型在 [`models/*.yaml`](https://github.com/ultralytics/yolov3/tree/master/models) 中以声明式定义，并由 `models/yolo.py` 中的 `parse_model()` 构建，因此无需编写 Python 即可查看和修改架构。

## 🏋️ 预训练权重

三款模型均在 [COCO](https://docs.ultralytics.com/datasets/detect/coco)（80 类）上训练，并在首次使用时从 [YOLOv3 版本资源](https://github.com/ultralytics/yolov3/releases)自动下载。

| 模型                                                                                            | 说明                                                |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [yolov3-tiny.pt](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-tiny.pt) | 轻量的两尺度模型——最快的选择，适合 CPU 和边缘设备。 |
| [yolov3.pt](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt)           | 原始 Darknet-53 模型——速度与精度的有力平衡。        |
| [yolov3-spp.pt](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-spp.pt)   | 增加空间金字塔池化，扩大感受野并提升精度。          |

## 🧩 集成

Ultralytics 与领先的 AI 平台集成，扩展数据集标注、训练、可视化和模型管理能力。在 [Ultralytics 集成](https://docs.ultralytics.com/integrations) 探索 [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases)、[Comet ML](https://docs.ultralytics.com/integrations/comet)、[Roboflow](https://docs.ultralytics.com/integrations/roboflow) 和 [Intel OpenVINO](https://docs.ultralytics.com/integrations/openvino) 等合作伙伴如何简化您的工作流。

<a href="https://docs.ultralytics.com/integrations" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations">
</a>

## 🤔 为何选择 YOLOv3？

YOLOv3 是实时目标检测的里程碑，至今仍是可靠且广为理解的基准：

- **实时单阶段检测** — 一次前向传播即可产生所有检测，无需独立的候选区域生成阶段。
- **跨尺寸表现稳健** — 多尺度预测可处理大、中、小目标。
- **支持多标签** — 独立的逻辑分类器允许重叠的类别标签。
- **简单且可移植** — 全卷积设计，训练顺畅，可干净地导出到多种部署格式。

如需了解更广泛的 Ultralytics YOLO 模型系列，请访问 [Ultralytics 仓库](https://github.com/ultralytics/ultralytics)。

## ☁️ 环境

使用预配置环境快速上手。点击下方图标了解各平台设置详情。

<div align="center">
  <a href="https://docs.ultralytics.com/integrations/paperspace">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-gradient.png" width="10%" alt="Run on Gradient"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/google-colab">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-colab-small.png" width="10%" alt="Open In Colab"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/kaggle">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-kaggle-small.png" width="10%" alt="Open In Kaggle"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/guides/docker-quickstart">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-docker-small.png" width="10%" alt="Docker Image"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/amazon-sagemaker">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-aws-small.png" width="10%" alt="AWS Marketplace"/></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://docs.ultralytics.com/integrations/google-colab">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/logo-gcp-small.png" width="10%" alt="GCP Quickstart"/></a>
</div>

## 🤝 贡献

欢迎贡献！请参阅[贡献指南](https://docs.ultralytics.com/help/contributing)开始参与，并通过 [Ultralytics 调查](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) 分享您的反馈。感谢所有贡献者！

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/yolov3/graphs/contributors)

## 📜 许可证

Ultralytics 提供两种许可选项：

- **AGPL-3.0 许可证**：经 [OSI 批准](https://opensource.org/license/agpl-3.0)的开源协议，适合科研与协作。详情见 [LICENSE](https://github.com/ultralytics/yolov3/blob/master/LICENSE)。
- **企业许可证**：面向商业用途，可将 Ultralytics 软件和模型集成到商业产品中，无需遵守 AGPL-3.0 的开源义务。请通过 [Ultralytics 许可](https://www.ultralytics.com/license) 联系我们。

## 📧 联系

如需报告 bug 或提出功能请求，请使用 [GitHub Issues](https://github.com/ultralytics/yolov3/issues)。如有问题或想讨论，欢迎加入我们的 [Discord](https://discord.com/invite/ultralytics)、[Reddit](https://www.reddit.com/r/ultralytics/) 和 [Ultralytics 论坛](https://community.ultralytics.com/)。

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
