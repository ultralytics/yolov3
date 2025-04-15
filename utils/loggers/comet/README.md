<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

<img src="https://cdn.comet.ml/img/notebook_logo.png">

# YOLOv3 Integration with Comet

This guide explains how to seamlessly integrate YOLOv3 with [Comet experiment tracking](https://www.comet.com/site/?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github) for enhanced experiment management, model optimization, and collaborative workflows.

## ℹ️ About Comet

[Comet](https://www.comet.com/site/?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github) is a leading platform for tracking, visualizing, and optimizing machine learning and deep learning experiments. It empowers data scientists, engineers, and teams to:

- Monitor model metrics in real time
- Save and version hyperparameters, datasets, and model checkpoints
- Visualize predictions using [Comet Custom Panels](https://www.comet.com/docs/v2/guides/comet-dashboard/code-panels/about-panels/?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github)
- Collaborate and share results efficiently

Comet ensures your work is always accessible and simplifies team collaboration.

## 🚀 Getting Started

### Install Comet

Install Comet using pip:

```shell
pip install comet_ml
```

### Configure Comet Credentials

You can set up Comet credentials for YOLOv3 in two ways:

1. **Environment Variables**  
   Set your credentials in your environment:

   ```shell
   export COMET_API_KEY=YOUR_COMET_API_KEY
   export COMET_PROJECT_NAME=YOUR_COMET_PROJECT_NAME # Defaults to 'yolov3' if not set
   ```

2. **Comet Configuration File**  
   Create a `.comet.config` file in your working directory:

   ```
   [comet]
   api_key=YOUR_API_KEY
   project_name=YOUR_PROJECT_NAME # Defaults to 'yolov3' if not set
   ```

### Run the Training Script

Run the [Ultralytics training script](https://docs.ultralytics.com/modes/train/) as usual. Comet will automatically integrate with YOLOv3.

```shell
# Train YOLOv3 on COCO128 for 5 epochs
python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov3.pt
```

Comet will automatically log your hyperparameters, command-line arguments, training metrics, and validation metrics. You can analyze your runs in the Comet UI. For more on metrics like mAP and Recall, see the [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

<img width="1920" alt="Comet UI showing YOLO training run" src="https://user-images.githubusercontent.com/26833433/202851203-164e94e1-2238-46dd-91f8-de020e9d6b41.png">

## ✨ Try an Example!

Explore a [completed YOLO run in the Comet UI](https://www.comet.com/examples/comet-example-yolov5/a0e29e0e9b984e4a822db2a62d0cb357?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github).

Or, try it yourself in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/integrations/model-training/yolov5/notebooks/Comet_and_YOLOv5.ipynb)

## 📊 Automatic Logging

By default, Comet logs the following during YOLOv3 training:

### Metrics

- Box Loss, Object Loss, Classification Loss (training and validation)
- mAP<sub>0.5</sub>, mAP<sub>0.5:0.95</sub> (validation)
- Precision and Recall (validation)

### Parameters

- All model hyperparameters
- All command-line options used during training

### Visualizations

- Confusion matrix of model predictions on validation data
- PR and F1 curves for all classes
- Correlogram of class labels

## ⚙️ Configure Comet Logging

You can customize Comet logging using environment variables:

```shell
# Comet Logging Configuration
export COMET_MODE=online                                    # 'online' or 'offline'. Defaults to online.
export COMET_MODEL_NAME=YOUR_MODEL_NAME                     # Name for the saved model. Defaults to yolov3.
export COMET_LOG_CONFUSION_MATRIX=false                     # Disable confusion matrix logging. Defaults to true.
export COMET_MAX_IMAGE_UPLOADS=NUMBER                       # Max prediction images to log. Defaults to 100.
export COMET_LOG_PER_CLASS_METRICS=true                     # Log per-class metrics. Defaults to false.
export COMET_DEFAULT_CHECKPOINT_FILENAME=your_checkpoint.pt # Checkpoint for resuming. Defaults to 'last.pt'.
export COMET_LOG_BATCH_LEVEL_METRICS=true                   # Log batch-level metrics. Defaults to false.
export COMET_LOG_PREDICTIONS=true                           # Set to false to disable prediction logging. Defaults to true.
```

### Logging Checkpoints with Comet

By default, [model checkpoints](https://docs.ultralytics.com/guides/model-training-tips/#checkpoints) are not uploaded to Comet. Enable checkpoint logging by using the `--save-period` argument:

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --save-period 1 # Save checkpoints every epoch
```

### Logging Model Predictions

Model predictions (images, ground truth, bounding boxes) are logged to Comet by default. Control frequency with the `--bbox_interval` argument (log every Nth batch per epoch). Visualize predictions using Comet's Object Detection Custom Panel.

**Note:** The YOLOv3 validation dataloader defaults to a batch size of 32. Adjust logging frequency as needed.

See an [example Comet project using the Object Detection Panel](https://www.comet.com/examples/comet-example-yolov5?shareable=YcwMiJaZSXfcEXpGOHDD12vA1&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github).

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --bbox_interval 2 # Log predictions every 2nd batch per epoch
```

#### Controlling the Number of Prediction Images Logged

Comet logs up to 100 validation images by default. Adjust this with the `COMET_MAX_IMAGE_UPLOADS` variable:

```shell
env COMET_MAX_IMAGE_UPLOADS=200 python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --bbox_interval 1
```

#### Logging Class-Level Metrics

Enable per-class mAP, precision, recall, and F1-score logging:

```shell
env COMET_LOG_PER_CLASS_METRICS=true python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt
```

## 💾 Uploading a Dataset to Comet Artifacts

Store your [datasets](https://docs.ultralytics.com/datasets/) using [Comet Artifacts](https://www.comet.com/docs/v2/guides/data-management/using-artifacts/#learn-more?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github) by adding the `--upload_dataset` flag. Ensure your dataset follows the structure in the [Ultralytics dataset guide](https://docs.ultralytics.com/datasets/). The dataset config YAML file must match the format of `coco128.yaml`.

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --upload_dataset # Uploads the dataset specified in coco128.yaml
```

Find uploaded datasets in the Artifacts tab in your Comet Workspace.
<img width="1073" alt="Comet Artifacts tab showing uploaded dataset" src="https://user-images.githubusercontent.com/7529846/186929193-162718bf-ec7b-4eb9-8c3b-86b3763ef8ea.png">

Preview data directly in the Comet UI.
<img width="1082" alt="Comet UI previewing dataset artifact" src="https://user-images.githubusercontent.com/7529846/186929215-432c36a9-c109-4eb0-944b-84c2786590d6.png">

Artifacts are versioned and support metadata. Comet automatically logs metadata from your dataset YAML file.
<img width="963" alt="Comet Artifact metadata view" src="https://user-images.githubusercontent.com/7529846/186929256-9d44d6eb-1a19-42de-889a-bcbca3018f2e.png">

### Using a Saved Artifact

To use a dataset stored in Comet Artifacts, update the `path` variable in your dataset YAML file to the Artifact resource URL:

```yaml
# contents of artifact.yaml
path: "comet://<workspace name>/<artifact name>:<artifact version or alias>"
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
# ... other dataset configurations
```

Then, pass this config file to your training script:

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data artifact.yaml \
  --weights yolov3.pt
```

Artifacts enable tracking data lineage throughout your workflow. The graph below shows experiments using the uploaded dataset.
<img width="1391" alt="Comet Artifact lineage graph" src="https://user-images.githubusercontent.com/7529846/186929264-4c4014fa-fe51-4f3c-a5c5-f6d24649b1b4.png">

## ▶️ Resuming a Training Run

If your training run is interrupted, resume it with the `--resume` flag and the Comet Run Path (`comet://<your workspace name>/<your project name>/<experiment id>`). This restores the model state, hyperparameters, arguments, and downloads necessary Comet Artifacts. Logging continues to the same Comet Experiment.

```shell
python train.py \
  --resume "comet://YOUR_WORKSPACE/YOUR_WORKSPACE/EXPERIMENT_ID"
```

## 🔍 Hyperparameter Search with the Comet Optimizer

YOLOv3 integrates with Comet's Optimizer for [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) and visualization.

### Configuring an Optimizer Sweep

Create a JSON config file for the sweep (e.g., `utils/loggers/comet/optimizer_config.json`):

```json
{
  "spec": {
    "maxCombo": 10,
    "objective": "minimize",
    "metric": "metrics/mAP_0.5",
    "algorithm": "bayes",
    "parameters": {
      "lr0": { "type": "float", "min": 0.001, "max": 0.01 },
      "momentum": { "type": "float", "min": 0.85, "max": 0.95 }
    }
  },
  "name": "YOLOv3 Hyperparameter Sweep",
  "trials": 1
}
```

Run the sweep with the `hpo.py` script:

```shell
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config utils/loggers/comet/optimizer_config.json
```

The `hpo.py` script accepts the same arguments as `train.py`. Add any additional arguments as needed:

```shell
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config utils/loggers/comet/optimizer_config.json \
  --save-period 1 \
  --bbox_interval 1
```

### Running a Sweep in Parallel

Use the `comet optimizer` command to run the sweep with multiple workers:

```shell
comet optimizer -j \
  utils/loggers/comet/hpo.py NUMBER_OF_WORKERS utils/loggers/comet/optimizer_config.json
```

### Visualizing Results

Comet provides rich visualizations for sweep results. Explore a [project with a completed sweep](https://www.comet.com/examples/comet-example-yolov5/view/PrlArHGuuhDTKC1UuBmTtOSXD/panels?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github).

<img width="1626" alt="Comet UI showing hyperparameter optimization results" src="https://user-images.githubusercontent.com/7529846/186914869-7dc1de14-583f-4323-967b-c9a66a29e495.png">

## 🤝 Contributing

Contributions to this integration are welcome! See the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing/) for details on how to get involved. Thank you for helping improve the Ultralytics ecosystem!
