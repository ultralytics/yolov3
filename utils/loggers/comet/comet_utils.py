# Ultralytics YOLOv3 🚀, AGPL-3.0 license

import logging
import os
from urllib.parse import urlparse

try:
    import comet_ml
except ImportError:
    comet_ml = None

import yaml

logger = logging.getLogger(__name__)

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
COMET_DEFAULT_CHECKPOINT_FILENAME = os.getenv("COMET_DEFAULT_CHECKPOINT_FILENAME", "last.pt")


def download_model_checkpoint(opt, experiment):
    """
    Downloads the model checkpoint from a Comet ML experiment and updates the `opt.weights` parameter with the local
    file path of the downloaded checkpoint.

    Args:
        opt (Namespace): Options containing experiment configurations, including the `weights` attribute that will be updated with the local file path of the downloaded checkpoint.
        experiment (Experiment): A Comet ML experiment object from which the model checkpoints are to be downloaded.

    Returns:
        None

    Notes:
        If the specified checkpoint filename is not found within the Comet ML experiment, an error message will be logged, and the function will return without updating `opt.weights`. The default checkpoint filename can be customized using the `COMET_DEFAULT_CHECKPOINT_FILENAME` environment variable.

    Examples:
        ```python
        from namespace import Namespace
        from comet_ml import Experiment

        opt = Namespace(project='project_name', weights='initial_weights_path')
        experiment = Experiment(api_key="YOUR_COMET_API_KEY", project_name="project_name")

        download_model_checkpoint(opt, experiment)

        print(f"Updated weights path: {opt.weights}")
        ```
    """
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    model_name = COMET_MODEL_NAME
    model_asset_list = experiment.get_model_asset_list(model_name)

    if len(model_asset_list) == 0:
        logger.error(f"COMET ERROR: No checkpoints found for model name : {model_name}")
        return

    model_asset_list = sorted(
        model_asset_list,
        key=lambda x: x["step"],
        reverse=True,
    )
    logged_checkpoint_map = {asset["fileName"]: asset["assetId"] for asset in model_asset_list}

    resource_url = urlparse(opt.weights)
    checkpoint_filename = resource_url.query

    if checkpoint_filename:
        asset_id = logged_checkpoint_map.get(checkpoint_filename)
    else:
        asset_id = logged_checkpoint_map.get(COMET_DEFAULT_CHECKPOINT_FILENAME)
        checkpoint_filename = COMET_DEFAULT_CHECKPOINT_FILENAME

    if asset_id is None:
        logger.error(f"COMET ERROR: Checkpoint {checkpoint_filename} not found in the given Experiment")
        return

    try:
        logger.info(f"COMET INFO: Downloading checkpoint {checkpoint_filename}")
        asset_filename = checkpoint_filename

        model_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
        model_download_path = f"{model_dir}/{asset_filename}"
        with open(model_download_path, "wb") as f:
            f.write(model_binary)

        opt.weights = model_download_path

    except Exception as e:
        logger.warning("COMET WARNING: Unable to download checkpoint from Comet")
        logger.exception(e)


def set_opt_parameters(opt, experiment):
    """
    Update the opts Namespace with parameters from Comet's ExistingExperiment when resuming a run.

    Args:
        opt (argparse.Namespace): Namespace of command line options containing various attributes
                                  that influence the configuration and behavior of the training process.
        experiment (comet_ml.APIExperiment): Comet API Experiment object that represents the Comet ML
                                             experiment whose assets are used to update the `opt` parameters.

    Returns:
        None: The function updates the provided `opt` object in place and does not return a value.

    Example:
        ```python
        from argparse import Namespace
        from comet_ml import APIExperiment

        opt = Namespace(project='my_project', resume='path/to/checkpoint', hyp={})
        experiment = APIExperiment(api_key='your_api_key', previous_experiment='your_previous_experiment_key')

        set_opt_parameters(opt, experiment)
        ```

    Note:
        This function assumes that an 'opt.yaml' file exists among the assets of the specified Comet experiment,
        and that it contains parameters relevant to the `opt` Namespace. Additionally, it creates a `hyp.yaml` file
        in the save directory to ensure compatibility with the training script's checks.
    """
    asset_list = experiment.get_asset_list()
    resume_string = opt.resume

    for asset in asset_list:
        if asset["fileName"] == "opt.yaml":
            asset_id = asset["assetId"]
            asset_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
            opt_dict = yaml.safe_load(asset_binary)
            for key, value in opt_dict.items():
                setattr(opt, key, value)
            opt.resume = resume_string

    # Save hyperparameters to YAML file
    # Necessary to pass checks in training script
    save_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(save_dir, exist_ok=True)

    hyp_yaml_path = f"{save_dir}/hyp.yaml"
    with open(hyp_yaml_path, "w") as f:
        yaml.dump(opt.hyp, f)
    opt.hyp = hyp_yaml_path


def check_comet_weights(opt):
    """
    Downloads model weights from Comet ML and updates the weights path in the options to the saved weights location.

    Args:
        opt (argparse.Namespace): Command line arguments containing the weights path which may include a Comet ML URL.

    Returns:
        bool | None: Returns True if weights are successfully downloaded and updated; otherwise, returns None.

    Notes:
        - Requires the `comet_ml` package to be installed.
        - The Comet URL prefix must be recognized and correctly parsed to fetch the weights.
        - The function relies on environment variables `COMET_MODEL_NAME` and `COMET_DEFAULT_CHECKPOINT_FILENAME` for model identification.
    """
    if comet_ml is None:
        return

    if isinstance(opt.weights, str) and opt.weights.startswith(COMET_PREFIX):
        api = comet_ml.API()
        resource = urlparse(opt.weights)
        experiment_path = f"{resource.netloc}{resource.path}"
        experiment = api.get(experiment_path)
        download_model_checkpoint(opt, experiment)
        return True

    return None


def check_comet_resume(opt):
    """
    Restores run parameters and model checkpoint from a Comet ML experiment.

    Args:
        opt (argparse.Namespace): Namespace object containing command-line arguments passed to the YOLOv3 training script.

    Returns:
        bool | None: Returns True if the run is restored successfully, otherwise returns None.

    Notes:
        - Requires `comet_ml` package. If the `comet_ml` package is not installed, the function will not execute.
        - Comet ML API is used to retrieve experiment details and associated assets for restoring the run state.
    """
    if comet_ml is None:
        return

    if isinstance(opt.resume, str) and opt.resume.startswith(COMET_PREFIX):
        api = comet_ml.API()
        resource = urlparse(opt.resume)
        experiment_path = f"{resource.netloc}{resource.path}"
        experiment = api.get(experiment_path)
        set_opt_parameters(opt, experiment)
        download_model_checkpoint(opt, experiment)

        return True

    return None
