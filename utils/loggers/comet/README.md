<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

<img src="https://cdn.comet.ml/img/notebook_logo.png">

# YOLOv3 Integration with Comet

This guide details how to effectively use YOLOv3 with [Comet](https://bit.ly/yolov5-readme-comet2) for experiment tracking and model optimization.

## ℹ️ About Comet

[Comet](https://www.comet.com/site/?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github) provides tools designed to help data scientists, engineers, and team leaders accelerate and optimize machine learning and deep learning models.

With Comet, you can track and visualize model metrics in real-time, save hyperparameters, datasets, and model checkpoints, and visualize model predictions using [Comet Custom Panels](https://www.comet.com/docs/v2/guides/comet-dashboard/code-panels/about-panels/?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github). Comet ensures your work is never lost and simplifies sharing results and collaboration across teams of any size.

## 🚀 Getting Started

### Install Comet

Install the necessary package using pip:

```shell
pip install comet_ml
```

### Configure Comet Credentials

You can configure Comet credentials for YOLOv3 in two ways:

1.  **Environment Variables**: Set your credentials directly in your environment.

    ```shell
    export COMET_API_KEY=<Your Comet API Key>
    export COMET_PROJECT_NAME=<Your Comet Project Name> # Defaults to 'yolov3' if not set
    ```

2.  **Comet Configuration File**: Create a `.comet.config` file in your working directory and add your credentials there.

    ```
    [comet]
    api_key=<Your Comet API Key>
    project_name=<Your Comet Project Name> # Defaults to 'yolov3' if not set
    ```

### Run the Training Script

Execute the [Ultralytics training script](https://docs.ultralytics.com/modes/train/) as you normally would. Comet automatically integrates with YOLOv3.

```shell
# Train YOLOv3 on COCO128 for 5 epochs
python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov3.pt
```

That's it! Comet will automatically log your hyperparameters, command-line arguments, training metrics, and validation metrics. You can visualize and analyze your runs in the Comet UI. For more details on metrics like mAP and Recall, see the [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

<img width="1920" alt="Comet UI showing YOLO training run" src="https://user-images.githubusercontent.com/26833433/202851203-164e94e1-2238-46dd-91f8-de020e9d6b41.png">

## ✨ Try an Example!

Explore an example of a [completed run in the Comet UI](https://www.comet.com/examples/comet-example-yolov5/a0e29e0e9b984e4a822db2a62d0cb357?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github).

Or, try it yourself using this Colab Notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/integrations/model-training/yolov5/notebooks/Comet_and_YOLOv5.ipynb)

## 📊 Automatic Logging

By default, Comet automatically logs the following:

### Metrics

- Box Loss, Object Loss, Classification Loss (Training and Validation)
- mAP_0.5, mAP_0.5:0.95 (Validation)
- Precision and Recall (Validation)

### Parameters

- Model Hyperparameters
- All command-line options passed during training

### Visualizations

- Confusion Matrix of model predictions on validation data
- Plots for PR and F1 curves across all classes
- Correlogram of Class Labels

## ⚙️ Configure Comet Logging

You can configure Comet to log additional data using command-line flags or environment variables.

```shell
# Environment Variables for Comet Configuration
export COMET_MODE=online # 'online' or 'offline'. Defaults to online.
export COMET_MODEL_NAME=<your_model_name> # Name for the saved model. Defaults to yolov3.
export COMET_LOG_CONFUSION_MATRIX=false # Disable logging Confusion Matrix. Defaults to true.
export COMET_MAX_IMAGE_UPLOADS=<number> # Max prediction images to log. Defaults to 100.
export COMET_LOG_PER_CLASS_METRICS=true # Log evaluation metrics per class. Defaults to false.
export COMET_DEFAULT_CHECKPOINT_FILENAME=<your_checkpoint.pt> # Checkpoint for resuming. Defaults to 'last.pt'.
export COMET_LOG_BATCH_LEVEL_METRICS=true # Log training metrics per batch. Defaults to false.
export COMET_LOG_PREDICTIONS=true # Set to false to disable logging model predictions. Defaults to true.
```

### Logging Checkpoints with Comet

Logging [model checkpoints](https://docs.ultralytics.com/guides/model-training-tips/#checkpoints) to Comet is disabled by default. Enable it by passing the `save-period` argument to the training script. Checkpoints will be saved to Comet at the specified interval.

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --save-period 1 # Saves checkpoints every epoch
```

### Logging Model Predictions

Model predictions (images, ground truth labels, bounding boxes) are logged to Comet by default. Control the frequency using the `bbox_interval` argument. This value corresponds to logging every Nth batch per epoch. Predictions can be visualized using Comet's Object Detection Custom Panel.

**Note:** The YOLOv3 validation dataloader defaults to a batch size of 32. Adjust the logging frequency accordingly.

See an [example project using the Panel here](https://www.comet.com/examples/comet-example-yolov5?shareable=YcwMiJaZSXfcEXpGOHDD12vA1&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github).

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --bbox_interval 2 # Logs predictions every 2nd batch per epoch
```

#### Controlling the Number of Prediction Images Logged

Comet logs images associated with predictions. The default maximum is 100 validation images. Adjust this using the `COMET_MAX_IMAGE_UPLOADS` environment variable.

```shell
env COMET_MAX_IMAGE_UPLOADS=200 python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --bbox_interval 1
```

#### Logging Class Level Metrics

Use the `COMET_LOG_PER_CLASS_METRICS` environment variable to log mAP, precision, recall, and F1-score for each class.

```shell
env COMET_LOG_PER_CLASS_METRICS=true python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt
```

## 💾 Uploading a Dataset to Comet Artifacts

Store your [datasets](https://docs.ultralytics.com/datasets/) using [Comet Artifacts](https://www.comet.com/docs/v2/guides/data-management/using-artifacts/#learn-more?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github) by using the `upload_dataset` flag. Ensure your dataset follows the structure described in the [Ultralytics dataset guide](https://docs.ultralytics.com/datasets/). The dataset config `yaml` file must match the format of `coco128.yaml`.

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov3.pt \
  --upload_dataset # Uploads the dataset specified in coco128.yaml
```

Find the uploaded dataset in the Artifacts tab in your Comet Workspace.
<img width="1073" alt="Comet Artifacts tab showing uploaded dataset" src="https://user-images.githubusercontent.com/7529846/186929193-162718bf-ec7b-4eb9-8c3b-86b3763ef8ea.png">

Preview the data directly in the Comet UI.
<img width="1082" alt="Comet UI previewing dataset artifact" src="https://user-images.githubusercontent.com/7529846/186929215-432c36a9-c109-4eb0-944b-84c2786590d6.png">

Artifacts are versioned and support metadata. Comet automatically logs metadata from your dataset `yaml` file.
<img width="963" alt="Comet Artifact metadata view" src="https://user-images.githubusercontent.com/7529846/186929256-9d44d6eb-1a19-42de-889a-bcbca3018f2e.png">

### Using a Saved Artifact

To use a dataset stored in Comet Artifacts, update the `path` variable in your dataset `yaml` file to the Artifact resource URL:

```yaml
# contents of artifact.yaml file
path: "comet://<workspace name>/<artifact name>:<artifact version or alias>"
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
# ... other dataset configurations
```

Then, pass this configuration file to your training script:

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data artifact.yaml \
  --weights yolov3.pt
```

Artifacts also enable tracking data lineage throughout your experimentation workflow. The graph below shows experiments that used the uploaded dataset.
<img width="1391" alt="Comet Artifact lineage graph" src="https://user-images.githubusercontent.com/7529846/186929264-4c4014fa-fe51-4f3c-a5c5-f6d24649b1b4.png">

## ▶️ Resuming a Training Run

If a training run is interrupted, resume it using the `--resume` flag with the Comet Run Path (`comet://<your workspace name>/<your project name>/<experiment id>`). This restores the model state, hyperparameters, training arguments, and downloads necessary Comet Artifacts. Logging continues to the existing Comet Experiment.

```shell
python train.py \
  --resume "comet://<your_workspace>/<your_project>/<experiment_id>"
```

## 🔍 Hyperparameter Search with the Comet Optimizer

YOLOv3 integrates with Comet's Optimizer for easy [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) and visualization.

### Configuring an Optimizer Sweep

Create a JSON configuration file for the sweep (e.g., `utils/loggers/comet/optimizer_config.json`).

```json
// Example optimizer_config.json
{
  "spec": {
    "maxCombo": 10, // Max number of experiments to run
    "objective": "minimize", // "minimize" or "maximize"
    "metric": "metrics/mAP_0.5", // Metric to optimize
    "algorithm": "bayes", // Optimization algorithm
    "parameters": {
      // Hyperparameters to tune
      "lr0": { "type": "float", "min": 0.001, "max": 0.01 },
      "momentum": { "type": "float", "min": 0.85, "max": 0.95 }
      // Add other hyperparameters from train.py args
    }
  },
  "name": "YOLOv3 Hyperparameter Sweep", // Name of the sweep
  "trials": 1 // Number of trials per experiment combination
}
```

Run the sweep using the `hpo.py` script:

```shell
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config utils/loggers/comet/optimizer_config.json
```

The `hpo.py` script accepts the same arguments as `train.py`. Add any additional arguments needed for the sweep:

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
  utils/loggers/comet/hpo.py < number_of_workers > utils/loggers/comet/optimizer_config.json
```

### Visualizing Results

Comet offers various visualizations for sweep results. Explore a [project with a completed sweep here](https://www.comet.com/examples/comet-example-yolov5/view/PrlArHGuuhDTKC1UuBmTtOSXD/panels?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github).

<img width="1626" alt="Comet UI showing hyperparameter optimization results" src="https://user-images.githubusercontent.com/7529846/186914869-7dc1de14-583f-4323-967b-c9a66a29e495.png">

## 🤝 Contributing

Contributions to enhance this integration are welcome! Please see the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more information on how to get involved. Thank you for helping improve the Ultralytics ecosystem!
