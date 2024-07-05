# Ultralytics YOLOv3 🚀, AGPL-3.0 license

import glob
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

try:
    import comet_ml

    # Project Configuration
    config = comet_ml.config.get_config()
    COMET_PROJECT_NAME = config.get_string(os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5")
except ImportError:
    comet_ml = None
    COMET_PROJECT_NAME = None

import PIL
import torch
import torchvision.transforms as T
import yaml

from utils.dataloaders import img2label_paths
from utils.general import check_dataset, scale_boxes, xywh2xyxy
from utils.metrics import box_iou

COMET_PREFIX = "comet://"

COMET_MODE = os.getenv("COMET_MODE", "online")

# Model Saving Settings
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")

# Dataset Artifact Settings
COMET_UPLOAD_DATASET = os.getenv("COMET_UPLOAD_DATASET", "false").lower() == "true"

# Evaluation Settings
COMET_LOG_CONFUSION_MATRIX = os.getenv("COMET_LOG_CONFUSION_MATRIX", "true").lower() == "true"
COMET_LOG_PREDICTIONS = os.getenv("COMET_LOG_PREDICTIONS", "true").lower() == "true"
COMET_MAX_IMAGE_UPLOADS = int(os.getenv("COMET_MAX_IMAGE_UPLOADS", 100))

# Confusion Matrix Settings
CONF_THRES = float(os.getenv("CONF_THRES", 0.001))
IOU_THRES = float(os.getenv("IOU_THRES", 0.6))

# Batch Logging Settings
COMET_LOG_BATCH_METRICS = os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true"
COMET_BATCH_LOGGING_INTERVAL = os.getenv("COMET_BATCH_LOGGING_INTERVAL", 1)
COMET_PREDICTION_LOGGING_INTERVAL = os.getenv("COMET_PREDICTION_LOGGING_INTERVAL", 1)
COMET_LOG_PER_CLASS_METRICS = os.getenv("COMET_LOG_PER_CLASS_METRICS", "false").lower() == "true"

RANK = int(os.getenv("RANK", -1))

to_pil = T.ToPILImage()


class CometLogger:
    """Log metrics, parameters, source code, models and much more with Comet."""

    def __init__(self, opt, hyp, run_id=None, job_type="Training", **experiment_kwargs) -> None:
        """
        Initialize the CometLogger instance with experiment configurations and hyperparameters for logging.

        Args:
            opt (Namespace): Namespace object containing the experiment options and configurations.
            hyp (dict): Dictionary of hyperparameters used for the experiment.
            run_id (str | None): Unique identifier for the experiment run. Defaults to None.
            job_type (str): The type of job being executed, default is "Training".
            **experiment_kwargs: Additional keyword arguments to pass to the Comet Experiment object.

        Returns:
            None

        Notes:
            - This method sets up the Comet experiment, configures logging settings, and initializes
              various experiment-specific parameters. For more information on Comet.ml integration,
              refer to https://www.comet.ml/docs/.
            - It logs several configurations and hyperparameters to Comet, enabling robust tracking
              and comparison of YOLOv3 experiments.

        Example:
            ```python
            from ultralytics import CometLogger
            import argparse

            # Example experiment options and hyperparameters
            opt = argparse.Namespace(save_period=5, upload_dataset=True, resume=False, data="my_dataset.yaml",
                                     name="my_experiment", conf_thres=0.25, iou_thres=0.45, bbox_interval=-1,
                                     epochs=50)

            hyp = {"lr0": 0.01, "momentum": 0.937, "weight_decay": 0.0005}

            # Initialize CometLogger
            comet_logger = CometLogger(opt, hyp)
            ```
        """
        self.job_type = job_type
        self.opt = opt
        self.hyp = hyp

        # Comet Flags
        self.comet_mode = COMET_MODE

        self.save_model = opt.save_period > -1
        self.model_name = COMET_MODEL_NAME

        # Batch Logging Settings
        self.log_batch_metrics = COMET_LOG_BATCH_METRICS
        self.comet_log_batch_interval = COMET_BATCH_LOGGING_INTERVAL

        # Dataset Artifact Settings
        self.upload_dataset = self.opt.upload_dataset or COMET_UPLOAD_DATASET
        self.resume = self.opt.resume

        # Default parameters to pass to Experiment objects
        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
            "project_name": COMET_PROJECT_NAME,
        }
        self.default_experiment_kwargs.update(experiment_kwargs)
        self.experiment = self._get_experiment(self.comet_mode, run_id)
        self.experiment.set_name(self.opt.name)

        self.data_dict = self.check_dataset(self.opt.data)
        self.class_names = self.data_dict["names"]
        self.num_classes = self.data_dict["nc"]

        self.logged_images_count = 0
        self.max_images = COMET_MAX_IMAGE_UPLOADS

        if run_id is None:
            self.experiment.log_other("Created from", "YOLOv3")
            if not isinstance(self.experiment, comet_ml.OfflineExperiment):
                workspace, project_name, experiment_id = self.experiment.url.split("/")[-3:]
                self.experiment.log_other(
                    "Run Path",
                    f"{workspace}/{project_name}/{experiment_id}",
                )
            self.log_parameters(vars(opt))
            self.log_parameters(self.opt.hyp)
            self.log_asset_data(
                self.opt.hyp,
                name="hyperparameters.json",
                metadata={"type": "hyp-config-file"},
            )
            self.log_asset(
                f"{self.opt.save_dir}/opt.yaml",
                metadata={"type": "opt-config-file"},
            )

        self.comet_log_confusion_matrix = COMET_LOG_CONFUSION_MATRIX

        if hasattr(self.opt, "conf_thres"):
            self.conf_thres = self.opt.conf_thres
        else:
            self.conf_thres = CONF_THRES
        if hasattr(self.opt, "iou_thres"):
            self.iou_thres = self.opt.iou_thres
        else:
            self.iou_thres = IOU_THRES

        self.log_parameters({"val_iou_threshold": self.iou_thres, "val_conf_threshold": self.conf_thres})

        self.comet_log_predictions = COMET_LOG_PREDICTIONS
        if self.opt.bbox_interval == -1:
            self.comet_log_prediction_interval = 1 if self.opt.epochs < 10 else self.opt.epochs // 10
        else:
            self.comet_log_prediction_interval = self.opt.bbox_interval

        if self.comet_log_predictions:
            self.metadata_dict = {}
            self.logged_image_names = []

        self.comet_log_per_class_metrics = COMET_LOG_PER_CLASS_METRICS

        self.experiment.log_others(
            {
                "comet_mode": COMET_MODE,
                "comet_max_image_uploads": COMET_MAX_IMAGE_UPLOADS,
                "comet_log_per_class_metrics": COMET_LOG_PER_CLASS_METRICS,
                "comet_log_batch_metrics": COMET_LOG_BATCH_METRICS,
                "comet_log_confusion_matrix": COMET_LOG_CONFUSION_MATRIX,
                "comet_model_name": COMET_MODEL_NAME,
            }
        )

        # Check if running the Experiment with the Comet Optimizer
        if hasattr(self.opt, "comet_optimizer_id"):
            self.experiment.log_other("optimizer_id", self.opt.comet_optimizer_id)
            self.experiment.log_other("optimizer_objective", self.opt.comet_optimizer_objective)
            self.experiment.log_other("optimizer_metric", self.opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_parameters", json.dumps(self.hyp))

    def _get_experiment(self, mode, experiment_id=None):
        """
        Returns a comet_ml Experiment object, either online or offline, existing or new, based on mode and
        experiment_id.

        Args:
          mode (str): Specifies whether to create an online or offline experiment. Acceptable values are 'online'
            and 'offline'.
          experiment_id (str | None): Optional; ID of an existing experiment to resume. If None, a new experiment
            is created.

        Returns:
          comet_ml.Experiment | comet_ml.ExistingExperiment | comet_ml.OfflineExperiment |
          comet_ml.ExistingOfflineExperiment: The Comet experiment object initialized based on provided mode and
            experiment_id.

        Raises:
          ValueError: If experiment creation fails due to unset Comet credentials in 'online' mode.

        Notes:
          For more information on how to set up and configure Comet experiments, visit the
          [Comet documentation](https://www.comet.com/docs/).
        """
        if mode == "offline":
            return (
                comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )
                if experiment_id is not None
                else comet_ml.OfflineExperiment(
                    **self.default_experiment_kwargs,
                )
            )
        try:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )

            return comet_ml.Experiment(**self.default_experiment_kwargs)

        except ValueError:
            logger.warning(
                "COMET WARNING: "
                "Comet credentials have not been set. "
                "Comet will default to offline logging. "
                "Please set your credentials to enable online logging."
            )
            return self._get_experiment("offline", experiment_id)

        return

    def log_metrics(self, log_dict, **kwargs):
        """
        Logs metrics to the current Comet experiment using a dictionary of metric names and values.

        Args:
            log_dict (dict): A dictionary where keys are metric names (str) and values are the corresponding metric values (int | float).
            **kwargs: Additional keyword arguments to pass to the comet_ml.Experiment.log_metrics() method.

        Returns:
            None

        Notes:
            Ensure that Comet.ml is properly configured and initialized before using this method. This function is
            typically used during the training and evaluation phases to log important performance metrics.

        Examples:
            ```python
            comet_logger = CometLogger(opt, hyp)
            metrics = {
                "accuracy": 0.95,
                "loss": 0.05
            }
            comet_logger.log_metrics(metrics)
            ```
        """
        self.experiment.log_metrics(log_dict, **kwargs)

    def log_parameters(self, log_dict, **kwargs):
        """
        Log parameters to the current Comet experiment.

        Args:
            log_dict (dict): A dictionary containing parameter names as keys and their corresponding values.
            kwargs (dict): Additional keyword arguments to pass to the Comet `log_parameters` method.

        Returns:
            None
        """
        self.experiment.log_parameters(log_dict, **kwargs)

    def log_asset(self, asset_path, **kwargs):
        """log_asset(asset_path: str, file_name: str = None, overwrite: bool = False) -> None:"""Logs a file or directory to the current Comet experiment.
        
            Args:
                asset_path (str): Path to the file or directory to be logged.
                file_name (str, optional): Optional new name for the logged file or directory. Defaults to None.
                overwrite (bool, optional): Whether to overwrite the existing asset if it exists. Defaults to False.
        
            Returns:
                None
        
            Examples:
                ```python
                import os
                from ultralytics import CometLogger
        
                # Initialize the CometLogger
                opt = {}
                hyp = {}
                logger = CometLogger(opt, hyp)
        
                # Log a file
                logger.log_asset(os.path.join('models', 'yolov5s.pt'))
                ```
            """
            self.experiment.log_asset(asset_path, file_name=file_name, overwrite=overwrite)
        """
        self.experiment.log_asset(asset_path, **kwargs)

    def log_asset_data(self, asset, **kwargs):
        """log_asset_data(asset: bytes | str, **kwargs) -> None:"""
            Log binary asset data to the current experiment.
        
            This method allows logging raw binary data or data referenced by a filepath as an asset in the Comet experiment.
            The asset can be accompanied by additional metadata or specifications passed through keyword arguments.
        
            Args:
                asset (bytes | str): The binary asset data or the path to the asset file to log.
                **kwargs: Additional keyword arguments to pass to the `experiment.log_asset_data` method. These can include
                    metadata fields such as:
                    - `name` (str): The name of the asset.
                    - `file_name` (str): The filename to associate with the asset.
                    - `overwrite` (bool): Whether to overwrite an existing asset with the same name.
        
            Returns:
                None
        
            Example:
                ```python
                logger = CometLogger(opt, hyp)
                with open('config.yaml', 'rb') as f:
                    config_data = f.read()
                logger.log_asset_data(config_data, name='config.yaml', metadata={'type': 'config-file'})
                ```
        """
        self.experiment.log_asset_data(asset, **kwargs)

    def log_image(self, img, **kwargs):
        """
        Logs an image to the current experiment with optional additional parameters for enhanced detail.
        
        Args:
            img (PIL.Image.Image | str): The image to log, which can be a PIL Image object or a file path to the image.
            kwargs (dict): Optional additional logging parameters, such as `name`, `step` and `overwrite`.
                           For example:
                           - `name` (str): Identifier name for the image.
                           - `step` (int): Step at which to log the image.
                           - `overwrite` (bool): If True, overwrites the image if it exists.
        
        Returns:
            None
        
        Example:
            ```python
            from PIL import Image
            from comet_ml import Experiment
            from ultralytics import CometLogger
        
            # Initialize Comet experiment
            experiment = Experiment(api_key="your_api_key", project_name="your_project_name")
        
            # Initialize CometLogger
            opt = {"save_period": 1, "upload_dataset": False}
            hyp = {"lr0": 0.01}
            logger = CometLogger(opt, hyp)
        
            # Log an example image
            img = Image.open("example.jpg")
            logger.log_image(img, name="example_image", step=5)
            ```
        
        Notes:
            - Ensure you have initialized the Comet experiment properly before attempting to log images.
            - Adequate image pre-processing might be necessary depending on the input format.
        """
        self.experiment.log_image(img, **kwargs)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log a machine learning model to Comet's experiment management platform.
        
        Args:
            path (str): The directory path where the model file is saved.
            opt (argparse.Namespace): Parsed command-line arguments containing training configurations.
            epoch (int): The current epoch number during training.
            fitness_score (list[float]): A list of fitness scores, typically evaluated metrics per epoch.
            best_model (bool): A boolean flag (default: False) indicating whether the model is the best-performing model so far.
        
        Returns:
            None
        
        Note:
            This function leverages functionalities from the `comet_ml` library to upload model states and metadata to an 
            online experiment dashboard. Ensure `comet_ml` is properly set up and configured. 
        
        Example:
            ```python
            logger = CometLogger(opt, hyp)
            logger.log_model(path='runs/train/exp', opt=opt, epoch=5, fitness_score=[0.5, 0.6, 0.7], best_model=True)
            ```
        """
        if not self.save_model:
            return

        model_metadata = {
            "fitness_score": fitness_score[-1],
            "epochs_trained": epoch + 1,
            "save_period": opt.save_period,
            "total_epochs": opt.epochs,
        }

        model_files = glob.glob(f"{path}/*.pt")
        for model_path in model_files:
            name = Path(model_path).name

            self.experiment.log_model(
                self.model_name,
                file_or_folder=model_path,
                file_name=name,
                metadata=model_metadata,
                overwrite=True,
            )

    def check_dataset(self, data_file):
        """
        Loads and validates the dataset configuration from a YAML file.
        
        Args:
            data_file (str): Path to the dataset configuration YAML file.
        
        Returns:
            dict: A dictionary containing the dataset configuration parameters.
        
        Notes:
            If the dataset path in the configuration file starts with 'comet://', it will initiate a download process
            for the specified dataset artifact.
        
        Example:
            ```python
            comet_logger = CometLogger(opt, hyp)
            dataset = comet_logger.check_dataset('config/dataset.yaml')
            ```
        """
        with open(data_file) as f:
            data_config = yaml.safe_load(f)

        path = data_config.get("path")
        if path and path.startswith(COMET_PREFIX):
            path = data_config["path"].replace(COMET_PREFIX, "")
            return self.download_dataset_artifact(path)
        self.log_asset(self.opt.data, metadata={"type": "data-config-file"})

        return check_dataset(data_file)

    def log_predictions(self, image, labelsn, path, shape, predn):
        """
        Logs filtered predictions with Intersection over Union (IoU) above a threshold, discarding if maximum
        image log count is reached.
        
        Args:
            image (PIL.Image.Image): The image on which predictions were made.
            labelsn (torch.Tensor): Ground truth labels in normalized xywh format with shape (n, 5), where
                n is the number of labels.
            path (str): Path to the image file.
            shape (tuple): Shape of the image in (height, width) format.
            predn (torch.Tensor): Predicted labels in normalized xywh format with confidence score and class ID,
                shape (m, 6), where m is the number of predictions.
        
        Returns:
            None
        
        Notes:
            - This function only logs images to the Comet experiment if the current logged image count 
              (`self.logged_images_count`) is less than the maximum allowed uploads (`self.max_images`).
            - Predictions and ground truth labels are filtered based on IoU and confidence thresholds.
            - Images are logged with metadata containing ground truth and predicted bounding boxes and confidence scores.
        
        Examples:
            ```python
            image = PIL.Image.open('path/to/image.jpg')
            labelsn = torch.tensor([[0, 0.5, 0.5, 0.8, 0.8]])
            path = 'path/to/image.jpg'
            shape = (1024, 1024)
            predn = torch.tensor([[0.5, 0.5, 0.8, 0.8, 0.9, 0]])
        
            comet_logger.log_predictions(image, labelsn, path, shape, predn)
            ```
        """
        if self.logged_images_count >= self.max_images:
            return
        detections = predn[predn[:, 4] > self.conf_thres]
        iou = box_iou(labelsn[:, 1:], detections[:, :4])
        mask, _ = torch.where(iou > self.iou_thres)
        if len(mask) == 0:
            return

        filtered_detections = detections[mask]
        filtered_labels = labelsn[mask]

        image_id = path.split("/")[-1].split(".")[0]
        image_name = f"{image_id}_curr_epoch_{self.experiment.curr_epoch}"
        if image_name not in self.logged_image_names:
            native_scale_image = PIL.Image.open(path)
            self.log_image(native_scale_image, name=image_name)
            self.logged_image_names.append(image_name)

        metadata = [
            {
                "label": f"{self.class_names[int(cls)]}-gt",
                "score": 100,
                "box": {"x": xyxy[0], "y": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
            }
            for cls, *xyxy in filtered_labels.tolist()
        ]
        metadata.extend(
            {
                "label": f"{self.class_names[int(cls)]}",
                "score": conf * 100,
                "box": {"x": xyxy[0], "y": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
            }
            for *xyxy, conf, cls in filtered_detections.tolist()
        )
        self.metadata_dict[image_name] = metadata
        self.logged_images_count += 1

        return

    def preprocess_prediction(self, image, labels, shape, pred):
        """
        Preprocesses predictions by adjusting the shapes of labels and predictions to match the dimensions of the input image.
        
        Args:
          image (torch.Tensor): Input image tensor.
          labels (torch.Tensor): Ground truth labels tensor with shape (N x 5), where N is the number of labels.
          shape (tuple[int, int]): Tuple containing height and width of the original image.
          pred (torch.Tensor): Model predictions tensor with shape (M x 6), where M is the number of predictions.
        
        Returns:
          labelsn (torch.Tensor | None): Adjusted ground truth labels in native image space. None if no labels.
          predn (torch.Tensor): Adjusted model predictions in native image space.
        
        Notes:
          - This method scales the bounding boxes of the ground truth labels and model predictions to the dimensions
            of the input image.
          - When `opt.single_cls` is set to True, all predictions are treated as belonging to a single class.
        
        Examples:
          ```python
          comet_logger = CometLogger(opt, hyp)
          image = torch.randn(3, 640, 480)  # Example image tensor
          labels = torch.randn(5, 5)  # Example labels tensor
          shape = (1024, 768)  # Original image dimensions
          pred = torch.randn(10, 6)  # Example predictions tensor
        
          labelsn, predn = comet_logger.preprocess_prediction(image, labels, shape, pred)
          ```
        """
        nl, _ = labels.shape[0], pred.shape[0]

        # Predictions
        if self.opt.single_cls:
            pred[:, 5] = 0

        predn = pred.clone()
        scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])

        labelsn = None
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(image.shape[1:], tbox, shape[0], shape[1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])  # native-space pred

        return predn, labelsn

    def add_assets_to_artifact(self, artifact, path, asset_path, split):
        """
        Adds asset images and labels from a specified directory to a Comet artifact.
        
        Args:
            artifact (comet_ml.Artifact): The Comet artifact to which the assets will be added.
            path (str): The base path to be used as the relative root for logical paths.
            asset_path (str): The directory containing the asset images and labels.
            split (str): The data subset to which the assets belong (e.g., 'train', 'val').
        
        Returns:
            None
        
        Notes:
            Paths within `asset_path` will be sorted, and corresponding image and label files will be added to the 
            `artifact` with their paths made relative to `path`. If any errors occur while adding files, those files 
            will be skipped, and errors will be logged.
        
        Example:
            ```python
            artifact = comet_ml.Artifact("dataset", type="dataset")
            comet_logger.add_assets_to_artifact(artifact, "/base/path", "/assets", "train")
            ```
        """
        img_paths = sorted(glob.glob(f"{asset_path}/*"))
        label_paths = img2label_paths(img_paths)

        for image_file, label_file in zip(img_paths, label_paths):
            image_logical_path, label_logical_path = map(lambda x: os.path.relpath(x, path), [image_file, label_file])

            try:
                artifact.add(
                    image_file,
                    logical_path=image_logical_path,
                    metadata={"split": split},
                )
                artifact.add(
                    label_file,
                    logical_path=label_logical_path,
                    metadata={"split": split},
                )
            except ValueError as e:
                logger.error("COMET ERROR: Error adding file to Artifact. Skipping file.")
                logger.error(f"COMET ERROR: {e}")
                continue

        return artifact

    def upload_dataset_artifact(self):
        """
        Uploads dataset to Comet as an artifact, defaulting to the 'yolov5-dataset' name if unspecified.
        
        Args:
            None
        
        Returns:
            None
        
        Notes:
            - Ensure that `self.data_dict` contains the appropriate dataset paths and metadata keys, such as 'train', 'val', and 
              'test', before invoking this method. The method uses these keys to locate dataset splits and log them as part of 
              the artifact.
        
        Example:
            ```python
            logger = CometLogger(opt, hyp)
            logger.upload_dataset_artifact()
            ```
        """
        dataset_name = self.data_dict.get("dataset_name", "yolov5-dataset")
        path = str((ROOT / Path(self.data_dict["path"])).resolve())

        metadata = self.data_dict.copy()
        for key in ["train", "val", "test"]:
            split_path = metadata.get(key)
            if split_path is not None:
                metadata[key] = split_path.replace(path, "")

        artifact = comet_ml.Artifact(name=dataset_name, artifact_type="dataset", metadata=metadata)
        for key in metadata.keys():
            if key in ["train", "val", "test"]:
                if isinstance(self.upload_dataset, str) and (key != self.upload_dataset):
                    continue

                asset_path = self.data_dict.get(key)
                if asset_path is not None:
                    artifact = self.add_assets_to_artifact(artifact, path, asset_path, key)

        self.experiment.log_artifact(artifact)

        return

    def download_dataset_artifact(self, artifact_path):
        """
        Downloads a dataset artifact to a specified directory, given its path.
        
        Args:
            artifact_path (str): The path of the artifact to be downloaded.
        
        Returns:
            None
        Notes:
            - The dataset artifact will be downloaded and saved in a subdirectory under the directory specified by 
              'self.opt.save_dir'.
            - The downloaded dataset's metadata will be used to update the 'data_dict' attribute.
            - Supports updating the 'names' field in the metadata, which maps class indices to class names, and can be
              either a list or a dictionary.
            
        Examples:
            ```python
            comet_logger = CometLogger(opt, hyp)
            comet_logger.download_dataset_artifact("path/to/artifact")
            ```
        """
        logged_artifact = self.experiment.get_artifact(artifact_path)
        artifact_save_dir = str(Path(self.opt.save_dir) / logged_artifact.name)
        logged_artifact.download(artifact_save_dir)

        metadata = logged_artifact.metadata
        data_dict = metadata.copy()
        data_dict["path"] = artifact_save_dir

        metadata_names = metadata.get("names")
        if isinstance(metadata_names, dict):
            data_dict["names"] = {int(k): v for k, v in metadata.get("names").items()}
        elif isinstance(metadata_names, list):
            data_dict["names"] = {int(k): v for k, v in zip(range(len(metadata_names)), metadata_names)}
        else:
            raise "Invalid 'names' field in dataset yaml file. Please use a list or dictionary"

        return self.update_data_paths(data_dict)

    def update_data_paths(self, data_dict):
        """
        Updates 'path' attribute in the provided data dictionary with the given base path, adjusting train, validation, and test
        path entries accordingly.
        
        Args:
           data_dict (dict): Dictionary containing dataset configuration, including 'path', 'train', 'val', and 'test' keys.
               'path' (str | None): Base path for the dataset. If not provided or empty, entries remain unchanged.
               'train' (str | list[str] | None): Path(s) for training data.
               'val' (str | list[str] | None): Path(s) for validation data.
               'test' (str | list[str] | None): Path(s) for test data.
        
        Returns:
           dict: Modified data dictionary with updated paths.
        
        Notes:
           This function alters the paths in the 'train', 'val', and 'test' keys to be relative to the base path provided in
           the 'path' key. If 'train', 'val', or 'test' entries are lists, each element in the list is updated accordingly.
        
        Examples:
           To update a data dictionary with new base paths:
           
           ```python
           data_config = {
               "path": "/data/yolo",
               "train": "train/images",
               "val": "val/images",
               "test": "test/images"
           }
        
           updated_data_config = comet_logger_instance.update_data_paths(data_config)
           ```
        
           This would result in:
           ```python
           {
               "path": "/data/yolo",
               "train": "/data/yolo/train/images",
               "val": "/data/yolo/val/images",
               "test": "/data/yolo/test/images"
           }
           ```
        """
        path = data_dict.get("path", "")

        for split in ["train", "val", "test"]:
            if data_dict.get(split):
                split_path = data_dict.get(split)
                data_dict[split] = (
                    f"{path}/{split_path}" if isinstance(split, str) else [f"{path}/{x}" for x in split_path]
                )

        return data_dict

    def on_pretrain_routine_end(self, paths):
        """
        Called at the end of the pretraining routine to handle paths modification if `opt.resume` is False.
        
        Args:
            paths (list[Path | str]): A list of file paths to be logged as assets.
        
        Returns:
            None
        
        Notes:
            - If `opt.resume` is set to `True`, this method will not log any assets.
            - If `self.upload_dataset` is `True` and `self.resume` is `False`, the dataset will be uploaded as an artifact.
        """
        if self.opt.resume:
            return

        for path in paths:
            self.log_asset(str(path))

        if self.upload_dataset and not self.resume:
            self.upload_dataset_artifact()

        return

    def on_train_start(self):
        """
        Logs hyperparameter settings and initial configurations at the start of training.
        
        Returns:
            None
        
        Raises:
            ValueError: If Comet credentials are not set and online logging is attempted without them.
        
        Notes:
            This method is called automatically at the start of training to log the initial configurations and
            hyperparameters of the experiment to Comet.
            
        Example:
            ```python
            logger = CometLogger(opt, hyp)
            logger.on_train_start()
            ```
        """
        self.log_parameters(self.hyp)

    def on_train_epoch_start(self):
        """
        Callback function executed at the start of each training epoch.
        
        Parameters:
            None
        
        Returns:
            None
        
        Notes:
            This method is typically used for initializing or resetting variables needed at the start of each new epoch 
            during the training process. Modify this function to include any actions that should be executed at the 
            beginning of each epoch.
        """
        return

    def on_train_epoch_end(self, epoch):
        """
        Callback function executed at the end of each training epoch, updating the current epoch in the experiment.
        
        Args:
            epoch (int): The current epoch number in the training process.
        
        Returns:
            None: This function does not return any value.
        
        Notes:
            The function is intended to maintain synchronization between the training loop and the Comet experiment by
            tracking the current epoch.
        """
        self.experiment.curr_epoch = epoch

        return

    def on_train_batch_start(self):
        """
        _train_batch_start()
        """
        Callback executed at the start of each training batch.

        This method is called at the beginning of each training batch to facilitate any additional processes, logging,
        or operations that need to be performed at the start of every batch during the training phase.

        Returns:
            None
        """
        return

    def on_train_batch_end(self, log_dict, step):
        """
        Handles the end of a training batch, logging metrics if specified conditions are met.

        Args:
            log_dict (dict): Dictionary containing metric names and their corresponding values to log.
            step (int): Current training step or iteration.

        Returns:
            None

        Notes:
            This method updates the `curr_step` attribute of the experiment to the current step. If the `log_batch_metrics`
            attribute is `True` and the current step is divisible by `comet_log_batch_interval`, it logs the metrics contained
            in `log_dict`.
        """
        self.experiment.curr_step = step
        if self.log_batch_metrics and (step % self.comet_log_batch_interval == 0):
            self.log_metrics(log_dict, step=step)

        return

    def on_train_end(self, files, save_dir, last, best, epoch, results):
        """
        Callback executed at the end of training to log final assets, results, and model to Comet.

        Args:
          files (list[str]): List of file paths to log as assets.
          save_dir (str): Directory where training results are saved.
          last (Path): Path to the last model checkpoint.
          best (Path): Path to the best model checkpoint.
          epoch (int): The current epoch number.
          results (dict): Dictionary of logged metrics and their values at the end of training.

        Returns:
          None

        Notes:
          If `comet_log_predictions` is set to True, image metadata is logged to Comet at the end of training. Final model state
          is logged, including checkpoints if `save_model` is enabled.

        Example usage:
          ```python
          comet_logger = CometLogger(opt, hyp)
          comet_logger.on_train_end(files, save_dir, last, best, epoch, results)
          ```
        """
        if self.comet_log_predictions:
            curr_epoch = self.experiment.curr_epoch
            self.experiment.log_asset_data(self.metadata_dict, "image-metadata.json", epoch=curr_epoch)

        for f in files:
            self.log_asset(f, metadata={"epoch": epoch})
        self.log_asset(f"{save_dir}/results.csv", metadata={"epoch": epoch})

        if not self.opt.evolve:
            model_path = str(best if best.exists() else last)
            name = Path(model_path).name
            if self.save_model:
                self.experiment.log_model(
                    self.model_name,
                    file_or_folder=model_path,
                    file_name=name,
                    overwrite=True,
                )

        # Check if running Experiment with Comet Optimizer
        if hasattr(self.opt, "comet_optimizer_id"):
            metric = results.get(self.opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_metric_value", metric)

        self.finish_run()

    def on_val_start(self):
        """
        Prepares the environment for the validation phase, logging necessary details and settings.

        Returns:
            None: This function does not return any value.

        Notes:
            This method is intended for use as a callback at the start of the validation phase in an ML experiment.
            Ensures that the validation environment is correctly set up and any specific configurations or initializations
            needed for logging with Comet are performed.

            Example usage:

            ```python
            comet_logger = CometLogger(opt, hyp)
            comet_logger.on_val_start()
            ```
        """
        return

    def on_val_batch_start(self):
        """
        Starts with a clear, concise summary line.

        Describes all parameters with types and brief descriptions.
        
        Specifies the return value and its type, always in parentheses.
        
        Preserves existing links or URLs from the current docstring.
        
        Follows Google Python Style Guide conventions.
        
        Includes minimal Examples (using ```python, not >>>) or extremely brief Notes sections if relevant.
        
        Uses zero indentation.
        
        Wraps text at 106 characters per line.
        
        Uses a vertical line for union types, e.g., '(int | str)'.
        
        Employs technical language suitable for ML engineers.
        
        Called at the start of each validation batch to prepare the batch environment.
        
        Args:
          None
        
        Returns:
          None
        
        Notes:
          This method does not perform any operations. It serves as a placeholder for batch-start callback handling
          during the validation phase.
        """
        return

    def on_val_batch_end(self, batch_i, images, targets, paths, shapes, outputs):
        """
        Handles end of each validation batch, optionally logging predictions to Comet.ml based on specified conditions.

        Args:
            batch_i (int): Index of the current batch.
            images (torch.Tensor): Tensor of images from the current batch.
            targets (torch.Tensor): Ground truth labels for the current batch.
            paths (list[str]): List of file paths for images in the current batch.
            shapes (list[tuple[int, int]]): Original shapes of the images in the current batch.
            outputs (list[torch.Tensor]): Predicted outputs from the model for the current batch.

        Returns:
            None

        Notes:
            - This method will log predictions to Comet.ml if `comet_log_predictions` is set to True and the batch index
            meets the specified logging interval condition (`comet_log_prediction_interval`).
        """
        if not (self.comet_log_predictions and ((batch_i + 1) % self.comet_log_prediction_interval == 0)):
            return

        for si, pred in enumerate(outputs):
            if len(pred) == 0:
                continue

            image = images[si]
            labels = targets[targets[:, 0] == si, 1:]
            shape = shapes[si]
            path = paths[si]
            predn, labelsn = self.preprocess_prediction(image, labels, shape, pred)
            if labelsn is not None:
                self.log_predictions(image, labelsn, path, shape, predn)

        return

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        """
        Logs per-class metric statistics to Comet.ml at the end of the validation phase, including confusion matrix if
        necessary.

        Args:
            nt (list[int]): Number of true instances for each class.
            tp (list[int]): Number of true positive instances for each class.
            fp (list[int]): Number of false positive instances for each class.
            p (list[float]): Precision values for each class.
            r (list[float]): Recall values for each class.
            f1 (list[float]): F1 scores for each class.
            ap (list[float]): Average precision values for each class.
            ap50 (list[float]): Average precision values at IoU=0.5 for each class.
            ap_class (list[int]): Class indices corresponding to the AP values.
            confusion_matrix (ConfusionMatrix): ConfusionMatrix object to be logged.

        Returns:
            None
        """
        if self.comet_log_per_class_metrics and self.num_classes > 1:
            for i, c in enumerate(ap_class):
                class_name = self.class_names[c]
                self.experiment.log_metrics(
                    {
                        "mAP@.5": ap50[i],
                        "mAP@.5:.95": ap[i],
                        "precision": p[i],
                        "recall": r[i],
                        "f1": f1[i],
                        "true_positives": tp[i],
                        "false_positives": fp[i],
                        "support": nt[c],
                    },
                    prefix=class_name,
                )

        if self.comet_log_confusion_matrix:
            epoch = self.experiment.curr_epoch
            class_names = list(self.class_names.values())
            class_names.append("background")
            num_classes = len(class_names)

            self.experiment.log_confusion_matrix(
                matrix=confusion_matrix.matrix,
                max_categories=num_classes,
                labels=class_names,
                epoch=epoch,
                column_label="Actual Category",
                row_label="Predicted Category",
                file_name=f"confusion-matrix-epoch-{epoch}.json",
            )

    def on_fit_epoch_end(self, result, epoch):
        """
        Logs metrics at the end of each training epoch with provided result and epoch number.

        Args:
            result (dict): Dictionary containing the results of the current epoch, with keys representing metric names
                and values as their corresponding measurements.
            epoch (int): The current epoch number.

        Returns:
            None

        Examples:
            ```python
            logger = CometLogger(opt, hyp)
            result = {"loss": 0.26, "accuracy": 92.5}
            epoch = 5
            logger.on_fit_epoch_end(result, epoch)
            ```

        Notes:
            This method is part of the integration with Comet.ml for logging metrics at the end of each epoch in order to
            monitor and visualize the training progress.
        """
        self.log_metrics(result, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        """
        Logs and saves the model periodically if conditions met, excluding final epoch unless best fitness is achieved.

        Args:
            last (Path): Path to the most recent model checkpoint.
            epoch (int): Current epoch number.
            final_epoch (bool): Flag to indicate if it's the final epoch of training.
            best_fitness (float): Best fitness score achieved so far.
            fi (float): Fitness score of the current checkpoint.

        Returns:
            None

        Notes:
            Model will save if the current epoch is not the final epoch and the save period condition is met. If it is the final
            epoch, the model will only be saved if it achieves the best fitness score.

        Example:
            ```python
            comet_logger.on_model_save(last_checkpoint_path, current_epoch, is_final_epoch, best_fitness_score, current_fitness_score)
            ```
        """
        if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
            self.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_params_update(self, params):
        """
        Updates and logs model parameters.

        Args:
            params (dict): Dictionary containing model parameters to be updated and logged.

        Returns:
            None
        """
        self.log_parameters(params)

    def finish_run(self):
        """
        Terminates the current experiment and performs necessary cleanup operations.

        Returns:
            None.
        """
        self.experiment.end()
