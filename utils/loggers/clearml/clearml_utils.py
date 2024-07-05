# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Main Logger class for ClearML experiment tracking."""

import glob
import re
from pathlib import Path

import numpy as np
import yaml
from ultralytics.utils.plotting import Annotator, colors

try:
    import clearml
    from clearml import Dataset, Task

    assert hasattr(clearml, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    clearml = None


def construct_dataset(clearml_info_string):
    """
    Load and parse a ClearML dataset definition from a YAML file.

    Args:
        clearml_info_string (str): A string containing the ClearML dataset identifier, prefixed by "clearml://".

    Returns:
        dict: A dictionary containing dataset paths and metadata with the following keys:
            - 'train' (str | None): Path to the training dataset.
            - 'test' (str | None): Path to the test dataset.
            - 'val' (str | None): Path to the validation dataset.
            - 'nc' (int): Number of classes in the dataset.
            - 'names' (list): List of class names.

    Raises:
        ValueError: If more than one YAML file is found in the ClearML dataset root directory.
        ValueError: If no YAML file is found in the ClearML dataset root directory.
        AssertionError: If the YAML file does not contain all required keys: 'train', 'test', 'val', 'nc', 'names'.

    Example:
        ```python
        clearml_info_string = "clearml://dataset_id"
        data_dict = construct_dataset(clearml_info_string)
        ```
    """
    dataset_id = clearml_info_string.replace("clearml://", "")
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_root_path = Path(dataset.get_local_copy())

    # We'll search for the yaml file definition in the dataset
    yaml_filenames = list(glob.glob(str(dataset_root_path / "*.yaml")) + glob.glob(str(dataset_root_path / "*.yml")))
    if len(yaml_filenames) > 1:
        raise ValueError(
            "More than one yaml file was found in the dataset root, cannot determine which one contains "
            "the dataset definition this way."
        )
    elif not yaml_filenames:
        raise ValueError(
            "No yaml definition found in dataset root path, check that there is a correct yaml file "
            "inside the dataset root path."
        )
    with open(yaml_filenames[0]) as f:
        dataset_definition = yaml.safe_load(f)

    assert set(
        dataset_definition.keys()
    ).issuperset(
        {"train", "test", "val", "nc", "names"}
    ), "The right keys were not found in the yaml file, make sure it at least has the following keys: ('train', 'test', 'val', 'nc', 'names')"

    data_dict = {
        "train": (
            str((dataset_root_path / dataset_definition["train"]).resolve()) if dataset_definition["train"] else None
        )
    }
    data_dict["test"] = (
        str((dataset_root_path / dataset_definition["test"]).resolve()) if dataset_definition["test"] else None
    )
    data_dict["val"] = (
        str((dataset_root_path / dataset_definition["val"]).resolve()) if dataset_definition["val"] else None
    )
    data_dict["nc"] = dataset_definition["nc"]
    data_dict["names"] = dataset_definition["names"]

    return data_dict


class ClearmlLogger:
    """
    Log training runs, datasets, models, and predictions to ClearML.

    This logger sends information to ClearML at app.clear.ml or to your own hosted server. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics, code information and basic data metrics
    and analyses.

    By providing additional command line arguments to train.py, datasets, models and predictions can also be logged.
    """

    def __init__(self, opt, hyp):
        """
        Initialize the ClearMLLogger class.

        Args:
            opt (namespace): Command line arguments for this run, including ClearML-specific options.
            hyp (dict): Hyperparameters for this run, typically specified in a configuration file.

        Returns:
            None

        Notes:
            - Initializes a ClearML Task to capture the experiment.
            - Optionally uploads the dataset version to ClearML Data if `opt.upload_dataset` is True.
            - Sets up the logger to track and limit the number of images logged per epoch to a configurable maximum.
            - Configures ClearML auto-connection settings and Docker image for ease of remote execution.
            - Connects provided hyperparameters and arguments to the ClearML Task.
            - If ClearML dataset logging is enabled, loads and prepares the dataset for logging.
        """
        self.current_epoch = 0
        # Keep tracked of amount of logged images to enforce a limit
        self.current_epoch_logged_images = set()
        # Maximum number of images to log to clearML per epoch
        self.max_imgs_to_log_per_epoch = 16
        # Get the interval of epochs when bounding box images should be logged
        self.bbox_interval = opt.bbox_interval
        self.clearml = clearml
        self.task = None
        self.data_dict = None
        if self.clearml:
            self.task = Task.init(
                project_name=opt.project if opt.project != "runs/train" else "YOLOv3",
                task_name=opt.name if opt.name != "exp" else "Training",
                tags=["YOLOv3"],
                output_uri=True,
                reuse_last_task_id=opt.exist_ok,
                auto_connect_frameworks={"pytorch": False},
                # We disconnect pytorch auto-detection, because we added manual model save points in the code
            )
            # ClearML's hooks will already grab all general parameters
            # Only the hyperparameters coming from the yaml config file
            # will have to be added manually!
            self.task.connect(hyp, name="Hyperparameters")
            self.task.connect(opt, name="Args")

            # Make sure the code is easily remotely runnable by setting the docker image to use by the remote agent
            self.task.set_base_docker(
                "ultralytics/yolov5:latest",
                docker_arguments='--ipc=host -e="CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1"',
                docker_setup_bash_script="pip install clearml",
            )

            # Get ClearML Dataset Version if requested
            if opt.data.startswith("clearml://"):
                # data_dict should have the following keys:
                # names, nc (number of classes), test, train, val (all three relative paths to ../datasets)
                self.data_dict = construct_dataset(opt.data)
                # Set data to data_dict because wandb will crash without this information and opt is the best way
                # to give it to them
                opt.data = self.data_dict

    def log_debug_samples(self, files, title="Debug Samples"):
        """
        Log files (images) as debug samples in the ClearML task.

        Args:
            files (List[Path]): A list of file paths in PosixPath format.
            title (str): A title grouping images with similar context. Default is "Debug Samples".

        Returns:
            None

        Notes:
            This function assumes that each file name follows a convention including '_batch' followed by digits,
            which determine the iteration number for ClearML logging. If '_batch' is not found in the file name, the
            iteration is logged as 0 by default.
        """
        for f in files:
            if f.exists():
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                self.task.get_logger().report_image(
                    title=title, series=f.name.replace(it.group(), ""), local_path=str(f), iteration=iteration
                )

    def log_image_with_boxes(self, image_path, boxes, class_names, image, conf_threshold=0.25):
        """
        Log a single image with drawn bounding boxes to ClearML as a debug sample.

        Args:
            image_path (Path): Path to the original image file.
            boxes (list): List of scaled predictions in the format [xmin, ymin, xmax, ymax, confidence, class].
            class_names (dict): Mapping of class integer to class name.
            image (torch.Tensor): Tensor containing the actual image data.
            conf_threshold (float, optional): Confidence threshold for displaying bounding boxes. Defaults to 0.25.

        Returns:
            None

        Notes:
            The function logs up to a maximum number of images per epoch, enforcing this limit with `self.max_imgs_to_log_per_epoch`.
            Bounding boxes are only logged for epochs that meet the interval criterion set by `self.bbox_interval`.
        """
        if (
            len(self.current_epoch_logged_images) < self.max_imgs_to_log_per_epoch
            and self.current_epoch >= 0
            and (self.current_epoch % self.bbox_interval == 0 and image_path not in self.current_epoch_logged_images)
        ):
            im = np.ascontiguousarray(np.moveaxis(image.mul(255).clamp(0, 255).byte().cpu().numpy(), 0, 2))
            annotator = Annotator(im=im, pil=True)
            for i, (conf, class_nr, box) in enumerate(zip(boxes[:, 4], boxes[:, 5], boxes[:, :4])):
                color = colors(i)

                class_name = class_names[int(class_nr)]
                confidence_percentage = round(float(conf) * 100, 2)
                label = f"{class_name}: {confidence_percentage}%"

                if conf > conf_threshold:
                    annotator.rectangle(box.cpu().numpy(), outline=color)
                    annotator.box_label(box.cpu().numpy(), label=label, color=color)

            annotated_image = annotator.result()
            self.task.get_logger().report_image(
                title="Bounding Boxes", series=image_path.name, iteration=self.current_epoch, image=annotated_image
            )
            self.current_epoch_logged_images.add(image_path)
