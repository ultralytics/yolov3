# Ultralytics YOLOv3 🚀, AGPL-3.0 license

# WARNING ⚠️ wandb is deprecated and will be removed in future release.
# See supported integrations at https://github.com/ultralytics/yolov5#integrations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from utils.general import LOGGER, colorstr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
RANK = int(os.getenv("RANK", -1))
DEPRECATION_WARNING = (
    f"{colorstr('wandb')}: WARNING ⚠️ wandb is deprecated and will be removed in a future release. "
    f'See supported integrations at https://github.com/ultralytics/yolov5#integrations.'
)

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
    LOGGER.warning(DEPRECATION_WARNING)
except (ImportError, AssertionError):
    wandb = None


class WandbLogger:
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.

    This logger sends information to W&B at wandb.ai. By default, this information includes hyperparameters, system
    configuration and metrics, model metrics, and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets, models and predictions can also be logged.

    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, run_id=None, job_type="Training"):
        """
        Initialize the WandbLogger instance.

        Args:
            opt (namespace): Commandline arguments for this run.
            run_id (str | None): Optional. Run ID of W&B run to be resumed. Defaults to None.
            job_type (str): To set the job type for this run. Should be one of 'Training', 'Evaluation', etc. Defaults to 'Training'.

        Returns:
            None

        Notes:
            - The WandbLogger instance is responsible for logging training runs, datasets, models, and predictions to Weights & Biases (W&B).
            - If W&B is successfully imported, a warning about its deprecation is logged. W&B support will be removed in a future release.
            - For more details on supported integrations, see https://github.com/ultralytics/yolov5#integrations.

        Example:
            ```python
            from your_library import WandbLogger

            # Assuming `args` is a parsed Namespace containing relevant attributes
            wandb_logger = WandbLogger(opt=args)
            ```
        """
        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, wandb.run if wandb else None
        self.val_artifact, self.train_artifact = None, None
        self.train_artifact_path, self.val_artifact_path = None, None
        self.result_artifact = None
        self.val_table, self.result_table = None, None
        self.max_imgs_to_log = 16
        self.data_dict = None
        if self.wandb:
            self.wandb_run = wandb.run or wandb.init(
                config=opt,
                resume="allow",
                project="YOLOv3" if opt.project == "runs/train" else Path(opt.project).stem,
                entity=opt.entity,
                name=opt.name if opt.name != "exp" else None,
                job_type=job_type,
                id=run_id,
                allow_val_change=True,
            )

        if self.wandb_run and self.job_type == "Training":
            if isinstance(opt.data, dict):
                # This means another dataset manager has already processed the dataset info (e.g. ClearML)
                # and they will have stored the already processed dict in opt.data
                self.data_dict = opt.data
            self.setup_training(opt)

    def setup_training(self, opt):
        """
        Sets up the training process for YOLO models, including model and dataset downloading, and logging
        configurations.

        Args:
            opt (namespace): Command line arguments for this run, including options for resuming and logging intervals.

        Returns:
            None

        Notes:
            - Downloading artifacts is attempted if `opt.resume` starts with `WANDB_ARTIFACT_PREFIX`.
            - Updates `data_dict` with information of previous runs if resumed, and paths of downloaded dataset artifacts.
            - Initializes logging dictionary and bounding box interval configuration.
        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval
        if isinstance(opt.resume, str):
            model_dir, _ = self.download_model_artifact(opt)
            if model_dir:
                self.weights = Path(model_dir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp, opt.imgsz = (
                    str(self.weights),
                    config.save_period,
                    config.batch_size,
                    config.bbox_interval,
                    config.epochs,
                    config.hyp,
                    config.imgsz,
                )

        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = opt.epochs + 1  # disable bbox_interval

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as a W&B artifact.

        Args:
            path (Path): Path of the directory containing the checkpoints.
            opt (namespace): Command-line arguments for the current run.
            epoch (int): Current epoch number.
            fitness_score (float): Fitness score for the current epoch.
            best_model (bool): Flag indicating if the current checkpoint is the best yet.

        Returns:
            None

        Notes:
            W&B integration is deprecated and will be removed in a future release. Refer to the supported integrations
            at https://github.com/ultralytics/yolov5#integrations.
        """
        model_artifact = wandb.Artifact(
            f"run_{wandb.run.id}_model",
            type="model",
            metadata={
                "original_url": str(path),
                "epochs_trained": epoch + 1,
                "save period": opt.save_period,
                "project": opt.project,
                "total_epochs": opt.epochs,
                "fitness_score": fitness_score,
            },
        )
        model_artifact.add_file(str(path / "last.pt"), name="last.pt")
        wandb.log_artifact(
            model_artifact,
            aliases=[
                "latest",
                "last",
                f"epoch {str(self.current_epoch)}",
                "best" if best_model else "",
            ],
        )
        LOGGER.info(f"Saving model artifact on epoch {epoch + 1}")

    def val_one_image(self, pred, predn, path, names, im):
        """
        Evaluates the model's prediction for a single image, updating metrics based on the comparison with ground truth.

        Args:
          pred (torch.Tensor): The model's prediction tensor for the image.
          predn (torch.Tensor): The normalized prediction tensor.
          path (str): Path to the image file.
          names (List[str]): List of class names.
          im (numpy.ndarray): The image array.

        Returns:
          None

        Notes:
          - wandb is deprecated and will be removed in a future release. See supported integrations at:
            https://github.com/ultralytics/yolov5#integrations.
        """
        pass

    def log(self, log_dict):
        """
        Log training metrics and media to Weights & Biases (W&B).

        Args:
            log_dict (Dict): Dictionary containing metrics and media to log for the current step.

        Returns:
            None

        Notes:
            This method logs various metrics such as loss, accuracy, and custom-defined metrics from the `log_dict` to W&B
            for comprehensive tracking and visualization. Ensure that `wandb` is properly initialized and `wandb_run` is active
            before calling this method. For more information on supported integrations, refer to
            https://github.com/ultralytics/yolov5#integrations.

        Example:
            ```python
            log_data = {
                "epoch": 1,
                "train/loss": 0.123,
                "val/accuracy": 0.987,
            }
            logger.log(log_data)
            ```
        """
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self):
        """
        Commit the accumulated logs, model artifacts, and tables to Weights & Biases (W&B) and reset the logging
        dictionary.

        Args:
            None

        Returns:
            None

        Notes:
            - This function should be called at the end of each training epoch to ensure that all metrics and log data
              collected during the epoch are sent to the W&B server.
            - Ensure that the W&B logging instance is properly initialized before invoking this method.

        Example:
            ```python
            logger = WandbLogger(opt)
            # ... training loop ...
            logger.end_epoch()
            ```
        """
        if self.wandb_run:
            with all_logging_disabled():
                try:
                    wandb.log(self.log_dict)
                except BaseException as e:
                    LOGGER.info(
                        f"An error occurred in wandb. The training will proceed without interruption. More info\n{e}"
                    )
                    self.wandb_run.finish()
                    self.wandb_run = None
                self.log_dict = {}

    def finish_run(self):
        """
        Log metrics if any and finish the current W&B run.

        Returns:
            None: This function does not return any value.
        """
        if self.wandb_run:
            if self.log_dict:
                with all_logging_disabled():
                    wandb.log(self.log_dict)
            wandb.run.finish()
            LOGGER.warning(DEPRECATION_WARNING)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager to temporarily disable all logging.

    Args:
        highest_level (int): The maximum logging level to disable. Default is `logging.CRITICAL`. Modify only if a custom level
                             higher than CRITICAL is defined.

    Returns:
        None

    Examples:
        Temporarily disable logging within a code block:

        ```python
        with all_logging_disabled():
            # your code here
        ```

    Source:
        For more information, see the [Gist source](https://gist.github.com/simon-weber/7853144).
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)
