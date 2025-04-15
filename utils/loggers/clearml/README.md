<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# ClearML Integration for Ultralytics YOLO

This guide details how to integrate [ClearML](https://clear.ml/), a leading open-source MLOps platform, with your Ultralytics YOLO projects. ClearML streamlines the entire machine learning lifecycle—from experiment tracking to deployment—making it easier to manage and scale your computer vision workflows.

<img align="center" src="https://github.com/thepycoder/clearml_screenshots/raw/main/logos_dark.png#gh-light-mode-only" alt="Clear|ML"><img align="center" src="https://github.com/thepycoder/clearml_screenshots/raw/main/logos_light.png#gh-dark-mode-only" alt="Clear|ML">

## ✨ About ClearML

[ClearML](https://clear.ml/) is an [open-source MLOps suite](https://github.com/clearml/clearml) that enables you to manage, automate, and orchestrate machine learning workflows efficiently. Integrating ClearML with Ultralytics YOLO unlocks several advantages:

- **Experiment Management**: Automatically track every YOLO training run, including code versions, configurations, metrics, and outputs in a centralized dashboard. Explore more about [Ultralytics experiment tracking integrations](https://docs.ultralytics.com/integrations/).
- **Data Versioning**: Manage and access your custom training datasets with ClearML Data Versioning. See how [Ultralytics datasets](https://docs.ultralytics.com/datasets/) are structured.
- **Remote Execution**: Train and monitor your YOLO models remotely using ClearML Agent on any machine or cloud instance. Learn about [model deployment options](https://docs.ultralytics.com/guides/model-deployment-options/).
- **Hyperparameter Optimization**: Use ClearML's HPO tools to optimize your model configurations and improve [mean average precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map). Review the [Ultralytics Hyperparameter Tuning guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/).
- **Model Deployment**: Deploy trained YOLO models as scalable APIs with ClearML Serving in just a few steps.

You can leverage any combination of these tools to fit your project requirements.

![ClearML scalars dashboard](https://raw.githubusercontent.com/thepycoder/clearml_screenshots/main/experiment_manager_with_compare.gif)

## 🦾 Setting Up ClearML

To use ClearML, connect the SDK to a ClearML Server instance. You have two main options:

1. **ClearML Hosted Service**: Register for a free account at the [ClearML Hosted Service](https://app.clear.ml/).
2. **Self-Hosted Server**: Deploy your own [ClearML Server](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server) for full control and data privacy.

Follow these steps to get started:

1. Install the `clearml` Python package:

   ```bash
   pip install clearml
   ```

2. Connect the ClearML SDK to your server. Generate credentials in the ClearML Web UI (Settings → Workspace → Create new credentials) and run:

   ```bash
   clearml-init
   ```

   Follow the prompts to complete setup.

Once configured, ClearML is ready to integrate with your YOLO workflows! 😎

## 🚀 Training YOLO With ClearML

Enabling ClearML experiment tracking for YOLO is simple. Ensure the `clearml` package is installed:

```bash
pip install clearml > =1.2.0
```

With ClearML installed, every YOLO [training run](https://docs.ultralytics.com/modes/train/) is automatically logged.

By default, experiments are organized under the `YOLO` project with the task name `Training`. You can customize these using the `--project` and `--name` arguments in your training command. ClearML uses `/` as a delimiter for subprojects.

**Example Training Command:**

```bash
# Train with default project/task names
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
```

**Example with Custom Names:**

```bash
# Train with custom project and task names
python train.py --project my_yolo_project --name experiment_001 --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
```

ClearML will automatically capture:

- Git repository details (URL, commit ID, entry point) and local code changes
- Installed Python packages and versions
- [Hyperparameters](https://www.ultralytics.com/glossary/hyperparameter-tuning) and script arguments
- [Model checkpoints](https://www.ultralytics.com/glossary/model-weights) (use `--save-period n` to save every `n` epochs)
- Console output (stdout and stderr)
- Performance [metrics and scalars](https://docs.ultralytics.com/guides/yolo-performance-metrics/) such as mAP<sub>0.5</sub>, mAP<sub>0.5:0.95</sub>, precision, recall, losses, and learning rates
- Machine details, runtime, and creation date
- Generated plots like label correlograms and [confusion matrices](https://www.ultralytics.com/glossary/confusion-matrix)
- Debug samples: images with bounding boxes, mosaic visualizations, and validation images per epoch

This comprehensive tracking allows you to visualize progress in the ClearML UI, compare experiments, and easily identify the best-performing models by sorting based on metrics like [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map).

## 🔗 Dataset Version Management

Versioning datasets is essential for reproducibility and collaboration in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) projects. ClearML Data helps manage datasets efficiently. YOLO supports using ClearML dataset IDs directly in the training command.

![ClearML Dataset Interface](https://raw.githubusercontent.com/thepycoder/clearml_screenshots/main/clearml_data.gif)

### Prepare Your Dataset

YOLO uses YAML files to define dataset configurations. Datasets are typically stored in a `../datasets` directory relative to your repository root. For example, the [COCO128 dataset](https://docs.ultralytics.com/datasets/detect/coco128/) structure:

```
../
├── yolov3/          # Your repository
└── datasets/
    └── coco128/     # Dataset root folder
        ├── images/
        ├── labels/
        ├── coco128.yaml  # Dataset configuration file <--- IMPORTANT
        ├── LICENSE
        └── README.txt
```

Ensure your custom dataset follows a similar structure.

⚠️ **Important**: Copy the dataset `.yaml` configuration file into the **root directory** of your dataset folder (e.g., `datasets/coco128/coco128.yaml`). This YAML file must include keys like `path`, `train`, `val`, `test`, `nc` (number of classes), and `names` (class names list) for ClearML integration to function correctly.

### Upload Your Dataset

Navigate to your dataset's root folder and use the `clearml-data` CLI tool to upload and version it:

```bash
# Navigate to the dataset directory
cd ../datasets/coco128

# Sync the dataset with ClearML (creates a versioned dataset)
clearml-data sync --project "YOLO Datasets" --name coco128 --folder .
```

This command creates a new ClearML dataset (or a new version if it exists) named `coco128` within the `YOLO Datasets` project.

Alternatively, use granular commands:

```bash
# Create a new dataset task
clearml-data create --project "YOLO Datasets" --name coco128

# Add files to the dataset (use '.' for current folder)
clearml-data add --files .

# Finalize and upload the dataset version
clearml-data close
```

### Run Training Using a ClearML Dataset

Once your dataset is versioned in ClearML, you can reference it directly in your YOLO training command using its unique ID. ClearML will automatically download the dataset if it's not present locally.

```bash
# Replace <your_dataset_id> with the actual ID from ClearML
python train.py --img 640 --batch 16 --epochs 3 --data clearml:// yolov5s.pt --cache < your_dataset_id > --weights
```

The dataset ID used will be logged as a parameter in your ClearML experiment, ensuring full traceability.

## 👀 Hyperparameter Optimization

ClearML's experiment tracking captures all the information needed to reproduce a run, forming the foundation for effective [hyperparameter optimization (HPO)](https://docs.ultralytics.com/guides/hyperparameter-tuning/). ClearML allows you to clone experiments, modify hyperparameters, and rerun them automatically.

To run HPO locally, Ultralytics provides a sample script. You'll need the ID of a previously executed training task (the "template task") to use as a base.

1. Locate the HPO script at `utils/loggers/clearml/hpo.py`.
2. Edit the script to include the `template task` ID.
3. Optionally, install [Optuna](https://optuna.org/) (`pip install optuna`) for advanced optimization strategies, or use the default `RandomSearch`.
4. Run the script:

   ```bash
   python utils/loggers/clearml/hpo.py
   ```

This script clones the template task, applies new hyperparameters suggested by the optimizer, and executes the modified task locally (`task.execute_locally()`). To run HPO remotely, change this to `task.execute()` to enqueue the tasks for a ClearML Agent.

![HPO in ClearML UI](https://raw.githubusercontent.com/thepycoder/clearml_screenshots/main/hpo.png)

## 🤯 Remote Execution (Advanced)

ClearML Agent enables running experiments on remote machines, such as on-premises servers or cloud GPUs. The agent fetches tasks from a queue, replicates the original environment (code, packages, uncommitted changes), executes the task, and reports results back to the ClearML Server.

- **Learn More**: Watch the [ClearML Agent Introduction](https://www.youtube.com/watch?v=MX3BrXnaULs) or read the [ClearML Agent documentation](https://clear.ml/docs/latest/docs/clearml_agent).

Turn any machine into a ClearML Agent by running:

```bash
# Replace <queues_to_listen_to> with your queue(s) name(s)
clearml-agent daemon --queue < queues_to_listen_to > [--docker] # Use --docker to run in a Docker container
```

### Cloning, Editing, and Enqueuing Tasks

You can manage remote execution tasks through the ClearML Web UI:

1. **Clone**: Right-click an existing experiment to clone it.
2. **Edit**: Modify hyperparameters or other configurations in the cloned task.
3. **Enqueue**: Right-click the modified task and select "Enqueue" to assign it to a specific queue monitored by your agents.

![Enqueue a task from the ClearML UI](https://raw.githubusercontent.com/thepycoder/clearml_screenshots/main/enqueue.gif)

### Executing a Task Remotely via Code

Alternatively, modify your training script to automatically enqueue the task for remote execution. Add `task.execute_remotely()` after the ClearML logger is initialized in `train.py`:

```python
# ... inside train.py ...

# Loggers setup
if RANK in {-1, 0}:
    # Initialize loggers, including ClearML
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)

    if loggers.clearml:
        # Add this line to send the task to a queue for remote execution
        loggers.clearml.task.execute_remotely(queue_name="my_default_queue")

        # Get dataset dictionary if using ClearML datasets
        data_dict = loggers.clearml.data_dict
# ... rest of the script ...
```

When you run the modified `train.py`, the script execution will pause, package the code and environment, and send the task to the specified queue (`my_default_queue`). A ClearML Agent listening to that queue will then pick it up and run it.

### Autoscaling Agents

ClearML also provides **Autoscalers** that automatically provision and manage cloud instances (AWS, GCP, Azure) as ClearML Agents based on queue load. Machines spin up when tasks are pending and shut down when idle, optimizing resource usage and cost.

Learn how to set up autoscalers:

[![Watch the Autoscaler setup video](https://img.youtube.com/vi/j4XVMAaUt3E/0.jpg)](https://youtu.be/j4XVMAaUt3E)

## 👋 Contribute

Contributions are welcome! If you'd like to improve this integration or suggest features, please see the Ultralytics [Contributing Guide](https://docs.ultralytics.com/help/contributing/) and submit a Pull Request. Thank you to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)
