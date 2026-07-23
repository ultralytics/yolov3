# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
PyTorch Hub models https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/.

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # official model
    model = torch.hub.load('ultralytics/yolov3:master', 'yolov3')  # from branch
    model = torch.hub.load('ultralytics/yolov3', 'custom', 'yolov3.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov3.pt', source='local')  # local repo
"""

from ultralytics.utils.patches import torch_load


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates or loads a YOLOv3 model with specified configurations and optional pretrained weights.

    Args:
        name (str): Model name such as 'yolov3' or a path to a model checkpoint file, e.g., 'path/to/best.pt'.
        pretrained (bool): Whether to load pretrained weights into the model. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): Whether to apply the YOLOv3 .autoshape() wrapper to the model for handling multiple input
                          types. Default is True.
        verbose (bool): If True, print all information to the screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters ('cpu', 'cuda', etc.). If None, defaults
            to the best available device.

    Returns:
        torch.nn.Module: YOLOv3 model loaded with or without pretrained weights.

    Raises:
        Exception: If model loading fails, re-raised with a message pointing to the PyTorch Hub troubleshooting guide:
            https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading

    Examples:
        ```python
        import torch
        model = _create('yolov3')
        ```
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import DetectionModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(ROOT / "requirements.txt", exclude=("opencv-python", "tensorboard", "thop"))
    name = Path(name)
    path = name.with_suffix(".pt") if name.suffix == "" and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = next(iter((Path(__file__).parent / "models").rglob(f"{path.stem}.yaml")))  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch_load(attempt_download(path), map_location=device)  # load
                csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt["model"].names) == classes:
                    model.names = ckpt["model"].names  # set class names attribute
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model.to(device)

    except Exception as e:
        help_url = "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading"
        s = f"{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help."
        raise RuntimeError(s) from e


def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None):
    """Loads a custom or local YOLOv3 model from a specified path, with options for autoshaping and device assignment.

    Args:
        path (str): Path to the model file. Supports both local and URL paths.
        autoshape (bool): If True, applies the YOLOv3 `.autoshape()` wrapper to allow for various input formats. Default
            is True.
        _verbose (bool): If True, outputs detailed information. Otherwise, limits verbosity. Default is True.
        device (str | torch.device | None): Device to load the model on. Default is None, which uses the available GPU
            if possible.

    Returns:
        (torch.nn.Module): The loaded YOLOv3 model, either with or without autoshaping applied.

    Raises:
        Exception: If the model loading fails due to invalid path or incompatible model state, with helpful suggestions
            including a reference to the troubleshooting page:
            https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov3', 'custom', 'path/to/best.pt')
        model = torch.hub.load('ultralytics/yolov3', 'custom', 'path/to/best.pt', autoshape=False, device='cpu')
        ```
    """
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov3(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiates the YOLOv3 model with optional pretrained weights, configurable input channels, classes,
    autoshaping, and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels. Defaults to 3.
        classes (int): Number of detection classes. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv3 .autoshape() wrapper to the model for various input formats like
            file/URI/PIL/cv2/np and adds non-maximum suppression (NMS). Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): Device to use for model computations (e.g., 'cpu', 'cuda'). If None, the
            best available device is automatically selected. Defaults to None.

    Returns:
        torch.nn.Module: The instantiated YOLOv3 model.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # using official model
        model = torch.hub.load('ultralytics/yolov3:master', 'yolov3')  # from specific branch
        model = torch.hub.load('ultralytics/yolov3', 'custom', 'yolov3.pt')  # using custom/local model
        model = torch.hub.load('.', 'custom', 'yolov3.pt', source='local')  # from local repository
        ```
    """
    return _create("yolov3", pretrained, channels, classes, autoshape, _verbose, device)


def yolov3_spp(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiates the YOLOv3-SPP model with optional pretrained weights, configurable input channels, classes,
    autoshaping, and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels. Defaults to 3.
        classes (int): Number of detection classes. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv3 .autoshape() wrapper to the model for various input formats and
            adds non-maximum suppression (NMS). Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): Device to use for model computations (e.g., 'cpu', 'cuda'). If None, the
            best available device is automatically selected. Defaults to None.

    Returns:
        torch.nn.Module: The instantiated YOLOv3-SPP model.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov3', 'yolov3_spp')  # using official model
        ```
    """
    return _create("yolov3-spp", pretrained, channels, classes, autoshape, _verbose, device)


def yolov3_tiny(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiates the YOLOv3-tiny model with optional pretrained weights, configurable input channels, classes,
    autoshaping, and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels. Defaults to 3.
        classes (int): Number of detection classes. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv3 .autoshape() wrapper to the model for various input formats and
            adds non-maximum suppression (NMS). Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): Device to use for model computations (e.g., 'cpu', 'cuda'). If None, the
            best available device is automatically selected. Defaults to None.

    Returns:
        torch.nn.Module: The instantiated YOLOv3-tiny model.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov3', 'yolov3_tiny')  # using official model
        ```
    """
    return _create("yolov3-tiny", pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov3-tiny", help="model name")
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [
        "data/images/zidane.jpg",  # filename
        Path("data/images/zidane.jpg"),  # Path
        "https://ultralytics.com/images/zidane.jpg",  # URI
        cv2.imread("data/images/bus.jpg")[:, :, ::-1],  # OpenCV
        Image.open("data/images/bus.jpg"),  # PIL
        np.zeros((320, 640, 3)),
    ]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()
