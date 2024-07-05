# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Logging utils."""

import os
import warnings
from pathlib import Path

import pkg_resources as pkg
import torch

from utils.general import LOGGER, colorstr, cv2
from utils.loggers.clearml.clearml_utils import ClearmlLogger
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ("csv", "tb", "wandb", "clearml", "comet")  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv("RANK", -1))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:

    def SummaryWriter(*args):
        """Imports TensorBoard's SummaryWriter for logging, with a fallback returning None if TensorBoard is not
        installed.
        """
        return None  # None = SummaryWriter(str)


try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version("0.12.2") and RANK in {0, -1}:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

try:
    import clearml

    assert hasattr(clearml, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    clearml = None

try:
    if RANK in {0, -1}:
        import comet_ml

        assert hasattr(comet_ml, "__version__")  # verify package import not local dir
        from utils.loggers.comet import CometLogger

    else:
        comet_ml = None
except (ImportError, AssertionError):
    comet_ml = None


class Loggers:
    # YOLOv3 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        """
        Initializes logging for YOLOv3 with specified directories, weights, options, hyperparameters, and loggers.

        Args:
            save_dir (Path | str, optional): Directory to save logs and outputs. Defaults to None.
            weights (str, optional): Path to model weights. Defaults to None.
            opt (argparse.Namespace, optional): Parsed command-line options. Defaults to None.
            hyp (dict, optional): Dictionary of hyperparameters. Defaults to None.
            logger (logging.Logger, optional): Logger for console output. Defaults to None.
            include (tuple, optional): Loggers to include from 'csv', 'tb', 'wandb', 'clearml', 'comet'.
                Defaults to all loggers, i.e., ('csv', 'tb', 'wandb', 'clearml', 'comet').

        Returns:
            None

        Examples:
            ```python
            from ultralytics import Loggers
            loggers = Loggers(save_dir='/path/to/logs', weights='yolov3.pt', opt=options, hyp=hyp)
            ```
        Notes:
            - Ensure that required packages for the included loggers (e.g., TensorBoard, W&B, ClearML, Comet) are installed.
            - When using Comet, run 'pip install comet_ml' to enable logging and visualization on the Comet platform.
        """
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # plot results
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # train loss
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",  # metrics
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # val loss
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params
        self.best_keys = ["best/epoch", "best/precision", "best/recall", "best/mAP_0.5", "best/mAP_0.5:0.95"]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Messages
        if not comet_ml:
            prefix = colorstr("Comet: ")
            s = f"{prefix}run 'pip install comet_ml' to automatically track and visualize YOLOv3 🚀 runs in Comet"
            self.logger.info(s)
        # TensorBoard
        s = self.save_dir
        if "tb" in self.include and not self.opt.evolve:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and "wandb" in self.include:
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt)
        else:
            self.wandb = None

        # ClearML
        if clearml and "clearml" in self.include:
            try:
                self.clearml = ClearmlLogger(self.opt, self.hyp)
            except Exception:
                self.clearml = None
                prefix = colorstr("ClearML: ")
                LOGGER.warning(
                    f"{prefix}WARNING ⚠️ ClearML is installed but not configured, skipping ClearML logging."
                    f" See https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration#readme"
                )

        else:
            self.clearml = None

        # Comet
        if comet_ml and "comet" in self.include:
            if isinstance(self.opt.resume, str) and self.opt.resume.startswith("comet://"):
                run_id = self.opt.resume.split("/")[-1]
                self.comet_logger = CometLogger(self.opt, self.hyp, run_id=run_id)

            else:
                self.comet_logger = CometLogger(self.opt, self.hyp)

        else:
            self.comet_logger = None

    @property
    def remote_dataset(self):
        """
        Fetches a dataset dictionary from ClearML, W&B, or Comet ML based on the logger instantiated.

        Returns:
            dict | None: A dataset dictionary if available, otherwise `None`.

        Examples:
            ```python
            loggers = Loggers(opt=opt)
            dataset = loggers.remote_dataset
            if dataset:
                print("Fetched dataset from remote logger successfully.")
            else:
                print("No remote dataset available.")
            ```

        Notes:
            - Requires the appropriate remote logger (ClearML, W&B, or Comet ML) to be properly configured.
            - Refer to [ClearML Documentation](https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration#readme) for setup information.
        """
        data_dict = None
        if self.clearml:
            data_dict = self.clearml.data_dict
        if self.wandb:
            data_dict = self.wandb.data_dict
        if self.comet_logger:
            data_dict = self.comet_logger.data_dict

        return data_dict

    def on_train_start(self):
        """
        Calls the `on_train_start` method on the Comet logger if it is available.

        Note:
            This method is typically invoked at the beginning of the training process to ensure that the
            Comet logger is properly initialized and logging can commence accordingly.

        Args:
            None

        Returns:
            None

        Examples:
            ```python
            loggers = Loggers(save_dir='path/to/save', weights='weights.pt', opt=opt, hyp=hyp, logger=LOGGER)
            loggers.on_train_start()
            ```

        Notes:
            - Ensure that the Comet logger is included in the logging configuration.
            - If Comet logger is not set up, this function will not perform any action.
        """
        if self.comet_logger:
            self.comet_logger.on_train_start()

    def on_pretrain_routine_start(self):
        """
        Initiates the pretraining routine for the logging system if the Comet logger is available.

        Args:
            None

        Returns:
            None: This function does not return anything.

        Notes:
            This method is designed to trigger specific logging actions or setups required before the training routine
            begins, specifically utilizing the Comet logger if it has been configured and included. This method ensures
            that all necessary pre-training logging processes are initiated properly.
        """
        if self.comet_logger:
            self.comet_logger.on_pretrain_routine_start()

    def on_pretrain_routine_end(self, labels, names):
        """
        Logs the end of the pretraining routine, generates label plots, and updates WandB/Comet with the generated
        images.

        Args:
            labels (list[int]): List of label indices.
            names (list[str]): List of class names corresponding to the label indices.

        Returns:
            None

        Notes:
            - If `plots` is enabled, this method will generate and save label plots in the specified save directory.
            - If a WandB logger is initialized, this method will log the generated label images to WandB.
            - If a Comet logger is initialized, it will handle additional actions for Comet logging.
        """
        if self.plots:
            plot_labels(labels, names, self.save_dir)
            paths = self.save_dir.glob("*labels*.jpg")  # training labels
            if self.wandb:
                self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})
            # if self.clearml:
            #    pass  # ClearML saves these images automatically using hooks
            if self.comet_logger:
                self.comet_logger.on_pretrain_routine_end(paths)

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        """
        Logs details of the training batch at the end of each batch, including to TensorBoard, Weights & Biases, and
        ClearML if enabled.

        Args:
            model (torch.nn.Module): The neural network model being trained.
            ni (int): Number of integrated batches since the start of training.
            imgs (torch.Tensor): Batch of input images.
            targets (torch.Tensor): Ground truth targets for the batch.
            paths (List[str]): List of image file paths corresponding to the batch.
            vals (List[float]): List of validation metric values (e.g., loss, precision) for the batch.

        Returns:
            None

        Notes:
            - This function can generate and save plots of the first few training batches.
            - TensorBoard, Weights & Biases, and ClearML logging is conditional based on user configuration.

        Example:
            To use `on_train_batch_end`, ensure TensorBoard and the respective loggers are properly set:

            ```python
            from utils.loggers import Loggers

            loggers = Loggers(save_dir=Path('/path/to/save'), opt=opt)
            loggers.on_train_batch_end(model, ni, imgs, targets, paths, vals)
            ```

            Ensure `opt` contains `imgsz`, `sync_bn`, and other necessary options.
        """
        log_dict = dict(zip(self.keys[:3], vals))
        # Callback runs on train batch end
        # ni: number integrated batches (since train start)
        if self.plots:
            if ni < 3:
                f = self.save_dir / f"train_batch{ni}.jpg"  # filename
                plot_images(imgs, targets, paths, f)
                if ni == 0 and self.tb and not self.opt.sync_bn:
                    log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz, self.opt.imgsz))
            if ni == 10 and (self.wandb or self.clearml):
                files = sorted(self.save_dir.glob("train*.jpg"))
                if self.wandb:
                    self.wandb.log({"Mosaics": [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})
                if self.clearml:
                    self.clearml.log_debug_samples(files, title="Mosaics")

        if self.comet_logger:
            self.comet_logger.on_train_batch_end(log_dict, step=ni)

    def on_train_epoch_end(self, epoch):
        """
        Callback that updates the current epoch in Weights & Biases (W&B) and Comet ML loggers at the end of each
        training epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None

        Notes:
            This function increments the current epoch in the W&B logger and triggers the `on_train_epoch_end` event in
            the Comet ML logger, if these loggers are available.

        Examples:
            ```python
            loggers = Loggers()
            loggers.on_train_epoch_end(epoch=5)
            ```
        """
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

        if self.comet_logger:
            self.comet_logger.on_train_epoch_end(epoch)

    def on_val_start(self):
        """
        Callback that notifies the comet logger at the start of each validation phase.

        Returns:
            None
        """
        if self.comet_logger:
            self.comet_logger.on_val_start()

    def on_val_image_end(self, pred, predn, path, names, im):
        """
        Callback that logs a single validation image and its predictions to WandB or ClearML at the end of validation.

        Args:
            pred (torch.Tensor): The predicted bounding boxes of the validation image.
            predn (torch.Tensor): The normalized predicted bounding boxes of the validation image.
            path (Path | str): Path to the validation image.
            names (List[str]): List of class names corresponding to the predictions.
            im (np.ndarray): The original validation image in numpy array format.

        Returns:
            None
        """
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)
        if self.clearml:
            self.clearml.log_image_with_boxes(path, pred, names, im)

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        """
        Logs a single validation batch for Comet ML analytics.

        Args:
            batch_i (int): The index of the current validation batch.
            im (torch.Tensor): The input images for the current batch.
            targets (torch.Tensor): The target annotations for the current batch.
            paths (list[str]): The file paths of the images in the current batch.
            shapes (list[tuple]): The shapes of the original images.
            out (torch.Tensor): The model's output predictions for the current batch.

        Returns:
            None

        Notes:
            This function will invoke the corresponding method in the initialized CometLogger if it's available, facilitating
            the tracking and analytics of validation batches.
        """
        if self.comet_logger:
            self.comet_logger.on_val_batch_end(batch_i, im, targets, paths, shapes, out)

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        """
        Logs validation results and images at the end of the validation phase for visual analytics.

        Args:
            nt (int): Number of true targets.
            tp (tensor): True positive counts for each class.
            fp (tensor): False positive counts for each class.
            p (tensor): Precision for each class.
            r (tensor): Recall for each class.
            f1 (tensor): F1 score for each class.
            ap (tensor): Average precision for each class.
            ap50 (tensor): AP at IoU threshold 50 for each class.
            ap_class (tensor): Classes for which AP is calculated.
            confusion_matrix: Confusion matrix recording the performance.

        Returns:
            None

        Examples:
            ```python
            loggers = Loggers()
            loggers.on_val_end(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)
            ```

        Note:
            This method is instrumental in tracking and visualizing training progress using Comet, WandB, and ClearML loggers,
            aiding in model performance diagnostics.
        """
        if self.wandb or self.clearml:
            files = sorted(self.save_dir.glob("val*.jpg"))
        if self.wandb:
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})
        if self.clearml:
            self.clearml.log_debug_samples(files, title="Validation")

        if self.comet_logger:
            self.comet_logger.on_val_end(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        """
        Logs epoch-end results, updating CSV files and logging metrics to various visual analytics platforms.

        Args:
            vals (list[float]): List of floating-point values representing metrics to log for the current epoch.
            epoch (int): The current epoch number.
            best_fitness (float): The best fitness score evaluated.
            fi (float): Fitness score for the current epoch.

        Returns:
            None

        Notes:
            This function updates the 'results.csv' file with current epoch metrics if CSV logging is enabled. It also logs
            metrics to TensorBoard, ClearML, Weights & Biases (W&B), and Comet if these loggers are initialized. For W&B, the function
            also updates the summary with the best results so far when the current epoch's fitness score matches the best fitness score.

        Example:
            ```python
            loggers = Loggers(save_dir='logs', weights='yolov3.pt', opt=options, hyp=hyp_params)
            vals = [0.1, 0.2, 0.3, 0.85, 0.75, 0.65, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05]
            loggers.on_fit_epoch_end(vals, epoch=5, best_fitness=0.85, fi=0.85)
            ```
        """
        x = dict(zip(self.keys, vals))
        if self.csv:
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = "" if file.exists() else (("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
        elif self.clearml:  # log to ClearML if TensorBoard not used
            for k, v in x.items():
                title, series = k.split("/")
                self.clearml.task.get_logger().report_scalar(title, series, v, epoch)

        if self.wandb:
            if best_fitness == fi:
                best_results = [epoch] + vals[3:7]
                for i, name in enumerate(self.best_keys):
                    self.wandb.wandb_run.summary[name] = best_results[i]  # log best results in the summary
            self.wandb.log(x)
            self.wandb.end_epoch()

        if self.clearml:
            self.clearml.current_epoch_logged_images = set()  # reset epoch image limit
            self.clearml.current_epoch += 1

        if self.comet_logger:
            self.comet_logger.on_fit_epoch_end(x, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        """
        Logs the current model state to WandB and ClearML after validating the save period and whether it is the final
        epoch.

        Args:
            last (Path): The path of the last saved model checkpoint.
            epoch (int): The current epoch number.
            final_epoch (bool): Whether the current epoch is the final training epoch.
            best_fitness (float): The best observed fitness score across training epochs.
            fi (float): The fitness score for the current epoch.

        Returns:
            None

        Notes:
            - The model is logged to WandB/ClearML only if the epoch matches the save period and it is not the final epoch.
            - For WandB, the model checkpoint that results in the best fitness score is marked as the best model.
        """
        if (epoch + 1) % self.opt.save_period == 0 and not final_epoch and self.opt.save_period != -1:
            if self.wandb:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)
            if self.clearml:
                self.clearml.task.update_output_model(
                    model_path=str(last), model_name="Latest Model", auto_delete_file=False
                )

        if self.comet_logger:
            self.comet_logger.on_model_save(last, epoch, final_epoch, best_fitness, fi)

    def on_train_end(self, last, best, epoch, results):
        """
        Callback to execute at the end of training, saving results and relevant metrics to the specified save directory
        and logging them to various integrated loggers.

        Args:
            last (Path): Path to the most recent model saved.
            best (Path): Path to the best model saved.
            epoch (int): The final epoch number.
            results (list of float): List of results metrics [P, R, mAP@.5, mAP@.5:.95].

        Returns:
            None

        Notes:
            - This method will save result plots as files in the configured save directory.
            - Results and final metrics are logged to any integrated loggers (e.g., TensorBoard, WandB, ClearML, Comet).
            - WandB logs additional metadata and artifacts if enabled.

        Examples:
            ```python
            loggers = Loggers(save_dir=Path('/path/to/save_dir'), ...)
            loggers.on_train_end(last_model_path, best_model_path, epoch, [0.9, 0.8, 0.85, 0.7])
            ```

        Refer to https://docs.ultralytics.com/ for more details on integrating with different logging platforms.
        """
        if self.plots:
            plot_results(file=self.save_dir / "results.csv")  # save results.png
        files = ["results.png", "confusion_matrix.png", *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R"))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb and not self.clearml:  # These images are already captured by ClearML by now, we don't want doubles
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats="HWC")

        if self.wandb:
            self.wandb.log(dict(zip(self.keys[3:10], results)))
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            if not self.opt.evolve:
                wandb.log_artifact(
                    str(best if best.exists() else last),
                    type="model",
                    name=f"run_{self.wandb.wandb_run.id}_model",
                    aliases=["latest", "best", "stripped"],
                )
            self.wandb.finish_run()

        if self.clearml and not self.opt.evolve:
            self.clearml.task.update_output_model(
                model_path=str(best if best.exists() else last), name="Best Model", auto_delete_file=False
            )

        if self.comet_logger:
            final_results = dict(zip(self.keys[3:10], results))
            self.comet_logger.on_train_end(files, self.save_dir, last, best, epoch, final_results)

    def on_params_update(self, params: dict):
        """
        Updates experiment hyperparameters or configs in WandB and Comet logger with provided params dictionary.

        Args:
            params (dict): Dictionary containing hyperparameters or configurations to be updated in the loggers.

        Returns:
            None

        Example:
            ```python
            loggers.on_params_update({'learning_rate': 0.001, 'batch_size': 16})
            ```
        Notes:
            - Only updates if WandB or Comet logger is enabled.
            - Parameters that are updated should be valid and consistent with the original setup configurations where applicable.
        """
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)
        if self.comet_logger:
            self.comet_logger.on_params_update(params)


class GenericLogger:
    """
    YOLOv3 General purpose logger for non-task specific logging
    Usage: from utils.loggers import GenericLogger; logger = GenericLogger(...)
    Arguments
        opt:             Run arguments
        console_logger:  Console logger
        include:         loggers to include
    """

    def __init__(self, opt, console_logger, include=("tb", "wandb")):
        """
        Initializes a generic logger for YOLOv3, with options for TensorBoard and wandb logging.

        Args:
          opt (argparse.Namespace): Configuration options for the logging.
          console_logger (logging.Logger): Logger for console output.
          include (tuple[str] | tuple): Loggers to include, defaults to ('tb', 'wandb').

        Returns:
          None

        Notes:
          - Ensures specified log directories are set up for TensorBoard logging.
          - Configures wandb if installed and specified in include list.
        """
        self.save_dir = Path(opt.save_dir)
        self.include = include
        self.console_logger = console_logger
        self.csv = self.save_dir / "results.csv"  # CSV logger
        if "tb" in self.include:
            prefix = colorstr("TensorBoard: ")
            self.console_logger.info(
                f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/"
            )
            self.tb = SummaryWriter(str(self.save_dir))

        if wandb and "wandb" in self.include:
            self.wandb = wandb.init(
                project=web_project_name(str(opt.project)), name=None if opt.name == "exp" else opt.name, config=opt
            )
        else:
            self.wandb = None

    def log_metrics(self, metrics, epoch):
        """
        Logs provided metrics to the appropriate logging platforms, including CSV, TensorBoard, and WandB.

        Args:
            metrics (dict): Dictionary containing metric names as keys and their respective values as values.
            epoch (int): The current epoch number.

        Returns:
            None

        Examples:
            ```python
            logger = GenericLogger(opt, console_logger)
            metrics = {"train/loss": 0.5, "val/accuracy": 0.85}
            epoch = 1
            logger.log_metrics(metrics, epoch)
            ```
        Notes:
            - Ensure the metrics dictionary is well-formed with metric names as keys.
            - If using WandB, ensure it is properly initialized before logging metrics.
        """
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
            with open(self.csv, "a") as f:
                f.write(s + ("%23.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            self.wandb.log(metrics, step=epoch)

    def log_images(self, files, name="Images", epoch=0):
        """
        Logs images to TensorBoard and Weights & Biases (wandb) for visualization and monitoring.

        Args:
          files (str | list[str]): Path(s) to the image file(s) to be logged.
          name (str, optional): Name of the image(s) to be used in Tensorboard and WandB logs. Defaults to "Images".
          epoch (int, optional): The epoch number to be associated with the images. Defaults to 0.

        Returns:
          None

        Notes:
          This function ensures all specified image files exist before attempting to log them. Images are logged in 'HWC'
          format for TensorBoard.

        Examples:
          ```python
          logger = GenericLogger(opt, console_logger)
          logger.log_images(['path/to/image1.jpg', 'path/to/image2.jpg'], name='Sample Images', epoch=5)
          ```
        """
        files = [Path(f) for f in (files if isinstance(files, (tuple, list)) else [files])]  # to Path
        files = [f for f in files if f.exists()]  # filter by exists

        if self.tb:
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats="HWC")

        if self.wandb:
            self.wandb.log({name: [wandb.Image(str(f), caption=f.name) for f in files]}, step=epoch)

    def log_graph(self, model, imgsz=(640, 640)):
        """
        Logs the computational graph of the given model to TensorBoard and other integrated loggers.

        Args:
            model (torch.nn.Module): The neural network model to log.
            imgsz (tuple[int, int]): The input image size for the model (default is (640, 640)).

        Returns:
            None

        Examples:
            ```python
            from utils.loggers import GenericLogger
            import torch
            import torchvision.models as models

            model = models.resnet18()
            logger = GenericLogger(opt, console_logger)
            logger.log_graph(model, (224, 224))
            ```
        """
        if self.tb:
            log_tensorboard_graph(self.tb, model, imgsz)

    def log_model(self, model_path, epoch=0, metadata=None):
        """
        Logs model to all configured loggers, given a model path, epoch, and optional metadata.

        Args:
            model_path (str | Path): Path to the model file to be logged.
            epoch (int): Epoch number to associate with the logged model. Defaults to 0.
            metadata (dict, optional): A dictionary of additional metadata to log with the model. Defaults to None.

        Returns:
            None

        Notes:
            This function logs the model to WandB if it is included in the loggers. Ensure that WandB is properly configured
            before calling this function.

        Example:
            ```python
            logger = GenericLogger(opt, console_logger, include=["tb", "wandb"])
            logger.log_model("/path/to/model.pth", epoch=10, metadata={"lr": 0.001, "batch_size": 16})
            ```
        """
        if metadata is None:
            metadata = {}
        if self.wandb:
            art = wandb.Artifact(name=f"run_{wandb.run.id}_model", type="model", metadata=metadata)
            art.add_file(str(model_path))
            wandb.log_artifact(art)

    def update_params(self, params):
        """
        Updates the experiment hyperparameters or configurations in WandB with the provided params dictionary.

        Args:
            params (dict): A dictionary containing the parameters to update.

        Returns:
            None

        Examples:
            ```python
            logger = GenericLogger(opt, console_logger)
            new_params = {"learning_rate": 0.002, "batch_size": 32}
            logger.update_params(new_params)
            ```

        Notes:
            Ensure that the WandB logger is initialized before calling this method.
        """
        if self.wandb:
            wandb.run.config.update(params, allow_val_change=True)


def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    """
    log_tensorboard_graph(tb, model, imgsz=(640, 640)): Logs a model graph to TensorBoard using an all-zero input image
    of shape `(1, 3, imgsz, imgsz)`.

    Args:
        tb (SummaryWriter): TensorBoard SummaryWriter object used for logging.
        model (torch.nn.Module): PyTorch model to be logged.
        imgsz (int | tuple[int, int], optional): Size of the input image, by default (640, 640). Can be an integer for
            square images or a tuple for rectangular images.

    Returns:
        None

    Raises:
        Exception: Raises a general exception if any error occurs during the logging process.

    Notes:
        It's crucial that the input image is initialized with zeros as `torch.zeros((1, 3, *imgsz))`, as using `torch.empty`
        might cause issues with the model tracing in PyTorch JIT.

    Example:
        ```python
        from torch.utils.tensorboard import SummaryWriter
        from ultralytics import YOLOv3

        model = YOLOv3()
        writer = SummaryWriter('runs/experiment1')
        log_tensorboard_graph(writer, model, imgsz=(640, 640))
        ```
    """
    try:
        p = next(model.parameters())  # for device, type
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # expand
        im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)  # input image (WARNING: must be zeros, not empty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress jit trace warning
            tb.add_graph(torch.jit.trace(de_parallel(model), im, strict=False), [])
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ TensorBoard graph visualization failure {e}")


def web_project_name(project):
    """
    Converts a local project name to a web-friendly format by adding a suffix based on its type (classification or
    segmentation).

    Args:
        project (str): The local project name.

    Returns:
        str: Converted web-friendly project name with an appropriate suffix.

    Examples:
        ```python
        from ultralytics import web_project_name

        project = "runs/train-experiment-cls"
        web_friendly_project = web_project_name(project)
        print(web_friendly_project)  # Outputs: runs/train-experiment-cls-Classify
        ```
    """
    if not project.startswith("runs/train"):
        return project
    suffix = "-Classify" if project.endswith("-cls") else "-Segment" if project.endswith("-seg") else ""
    return f"YOLOv3{suffix}"
