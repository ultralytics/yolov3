# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5.

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # official model
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # from branch
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # local repo
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """
    Creates or loads a YOLOv3 model with specified configurations and optional pretrained weights.

    Args:
        name (str): Model name such as 'yolov5s' or a path to a model checkpoint file, e.g., 'path/to/best.pt'.
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

    Example:
        ```python
        import torch
        model = _create('yolov5s')
        ```

    Raises:
        Exception: If an error occurs while loading the model, returns an error message with a helpful URL:
                   "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading".
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
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
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv3 ClassificationModel is not yet AutoShape compatible. "
                            "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224)."
                        )
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv3 SegmentationModel is not yet AutoShape compatible. "
                            "You will not be able to run inference with this model."
                        )
                    else:
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = list((Path(__file__).parent / "models").rglob(f"{path.stem}.yaml"))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
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
        raise Exception(s) from e


def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None):
    """
    Loads a custom or local YOLOv3 model from a specified path, with options for autoshaping and device assignment.

    Args:
        path (str): Path to the model file. Supports both local and URL paths.
        autoshape (bool): If True, applies the YOLOv3 `.autoshape()` wrapper to allow for various input formats. Default is True.
        _verbose (bool): If True, outputs detailed information. Otherwise, limits verbosity. Default is True.
        device (str | torch.device | None): Device to load the model on. Default is None, which uses the available GPU if
            possible.

    Returns:
        (torch.nn.Module): The loaded YOLOv3 model, either with or without autoshaping applied.

    Raises:
        Exception: If the model loading fails due to invalid path or incompatible model state, with helpful suggestions
            including a reference to the troubleshooting page:
            https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/best.pt')
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/best.pt', autoshape=False, device='cpu')
        ```
    """
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates a YOLOv5n model with optional pretrained weights, configurable input channels, number of classes,
    autoshaping, and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels. Defaults to 3.
        classes (int): Number of detection classes. Defaults to 80.
        autoshape (bool): If True, applies YOLOv5 .autoshape() wrapper to the model for various input formats like file/URI/PIL/cv2/np
            and adds non-maximum suppression (NMS). Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): Device to use for model computations (e.g., 'cpu', 'cuda'). If None, the best
            available device is automatically selected. Defaults to None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5n model.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # using official model
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5n')  # from specific branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n.pt')  # using custom/local model
        model = torch.hub.load('.', 'custom', 'yolov5n.pt', source='local')  # from local repository
        ```

    Note:
        PyTorch Hub models can be explored at https://pytorch.org/hub/ultralytics_yolov5. This allows easy model loading and usage.
    """
    return _create("yolov5n", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Load the YOLOv5s model with customizable options for pretrained weights, input channels, number of classes,
    autoshape functionality, and device selection.

    Args:
        pretrained (bool, optional): If True, loads model with pretrained weights. Default is True.
        channels (int, optional): Specifies the number of input channels. Default is 3.
        classes (int, optional): Defines the number of model classes. Default is 80.
        autoshape (bool, optional): Applies YOLOv5 .autoshape() wrapper to the model for enhanced usability. Default is True.
        _verbose (bool, optional): If True, prints detailed information during model loading. Default is True.
        device (str | torch.device | None, optional): Specifies the device to load the model on. Accepts 'cpu', 'cuda', or
            torch.device. Default is None, which automatically selects the best available option.

    Returns:
        torch.nn.Module: The initialized YOLOv5s model loaded with the specified options.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        ```

    For more information, refer to [PyTorch Hub models](https://pytorch.org/hub/ultralytics_yolov5).
    """
    return _create("yolov5s", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5m model with options for pretrained weights, input channels, number of classes, autoshape
    functionality, and device selection.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels for the model. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): If True, applies the YOLOv5 .autoshape() wrapper for handling multiple input types and NMS.
            Default is True.
        _verbose (bool, optional): If True, prints detailed information during model loading. Default is True.
        device (str | torch.device | None, optional): Device for model computations (e.g., 'cpu', 'cuda'). Automatically
            selects the best available device if None. Default is None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5m model.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        ```
    """
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Load the YOLOv5l model with customizable options for pretrained weights, input channels, number of classes,
    autoshape functionality, and device selection.

    Args:
        pretrained (bool, optional): If True, load model with pretrained weights. Default is True.
        channels (int, optional): Specifies the number of input channels. Default is 3.
        classes (int, optional): Defines the number of model classes. Default is 80.
        autoshape (bool, optional): Applies the YOLOv5 .autoshape() wrapper to the model for enhanced usability. Default is
            True.
        _verbose (bool, optional): If True, prints detailed information during model loading. Default is True.
        device (str | torch.device | None, optional): Specifies the device to load the model on. Accepts 'cpu', 'cuda', or
            torch.device. Default is None, which automatically selects the best available option.

    Returns:
        torch.nn.Module: The initialized YOLOv5l model loaded with the specified options.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        ```

    For more information, refer to [PyTorch Hub models](https://pytorch.org/hub/ultralytics_yolov5).
    """
    return _create("yolov5l", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Load the YOLOv5x model with options for pretrained weights, number of input channels, classes, autoshaping, and
    device selection.

    Args:
        pretrained (bool, optional): If True, loads the model with pretrained weights. Defaults to True.
        channels (int, optional): Number of input channels. Defaults to 3.
        classes (int, optional): Number of detection classes. Defaults to 80.
        autoshape (bool, optional): If True, applies the YOLOv5 .autoshape() wrapper, enabling various input formats and
            non-maximum suppression (NMS). Defaults to True.
        _verbose (bool, optional): If True, prints detailed information during model loading. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model parameters (e.g., 'cpu', 'cuda'). Defaults to
            None, selecting the best available device automatically.

    Returns:
        torch.nn.Module: The YOLOv5x model loaded with the specified configuration.

    Examples:
        ```python
        import torch

        # Load YOLOv5x model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

        # Load YOLOv5x model with custom device
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device='cuda:0')
        ```

    For more details, refer to [PyTorch Hub models](https://pytorch.org/hub/ultralytics_yolov5).
    """
    return _create("yolov5x", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5n6 model with options for pretrained weights, input channels, classes, autoshaping, verbosity, and
    device assignment.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): If True, applies the YOLOv3 .autoshape() wrapper to the model. Default is True.
        _verbose (bool, optional): If True, prints all information to the screen. Default is True.
        device (str | torch.device | None, optional): Device to use for model parameters, e.g., 'cpu', '0', or torch.device.
            Default is None.

    Returns:
        torch.nn.Module: YOLOv5n6 model loaded on the specified device and configured as per the provided options.

    Notes:
        For more information on PyTorch Hub models, refer to: https://pytorch.org/hub/ultralytics_yolov5

    Example:
        ```python
        model = yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device='cuda')
        ```
    """
    return _create("yolov5n6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5s6 model with options for weights, channels, classes, autoshaping, and device selection.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights into the model. Defaults to True.
        channels (int, optional): Number of input channels. Defaults to 3.
        classes (int, optional): Number of model classes. Defaults to 80.
        autoshape (bool, optional): Apply YOLOv5 .autoshape() wrapper to the model. Defaults to True.
        _verbose (bool, optional): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model parameters, e.g., 'cpu', 'cuda:0'.
            If None, it will select the appropriate device automatically. Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5s6 model, ready for inference or further training.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True, channels=3, classes=80)
        model.eval()  # Set the model to evaluation mode
        ```

    For more details, see the official documentation at:
    https://github.com/ultralytics/yolov5
    """
    return _create("yolov5s6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads YOLOv5m6 model with options for pretrained weights, input channels, number of classes, autoshaping, and device
    selection.

    Args:
        pretrained (bool): Whether to load pretrained weights into the model. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): Whether to apply YOLOv5 .autoshape() wrapper to the model. Default is True.
        _verbose (bool): Whether to print all information to the screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters, e.g., 'cpu', 'cuda', 'mps', or torch device.
            Default is None.

    Returns:
        YOLOv5m6 model (torch.nn.Module): The instantiated YOLOv5m6 model with specified options.

    Example:
        ```python
        import torch

        # Load YOLOv5m6 model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')

        # Load custom YOLOv5m6 model from a local path with specific options
        model = torch.hub.load('.', 'yolov5m6', pretrained=False, channels=1, classes=10, device='cuda')
        ```

    Notes:
        For more detailed documentation, visit https://github.com/ultralytics/yolov5
    """
    return _create("yolov5m6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5l6 model with options for pretrained weights, input channels, the number of classes, autoshaping,
    and device selection.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): If True, applies the YOLOv5 .autoshape() wrapper to the model for automatic shape
            inference. Default is True.
        _verbose (bool, optional): If True, prints all information to the screen. Default is True.
        device (str | torch.device | None, optional): Device to use for the model parameters, e.g., 'cpu', 'cuda', or
            a specific GPU like 'cuda:0'. Default is None, which means the best available device will be selected
            automatically.

    Returns:
        yolov5.models.yolo.DetectionModel: YOLOv5l6 model initialized with defined custom configurations.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')  # Load YOLOv5l6 model
        ```

    Note:
        For more details, visit the [Ultralytics YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5).
    """
    return _create("yolov5l6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5x6 model, allowing customization for pretrained weights, input channels, and model classes.

    Args:
        pretrained (bool): If True, loads the model with pretrained weights. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of output classes for the model. Default is 80.
        autoshape (bool): If True, applies the .autoshape() wrapper for inference on diverse input formats. Default is True.
        _verbose (bool): If True, prints detailed information during model loading. Default is True.
        device (str | torch.device | None): Specifies the device to load the model on ('cpu', 'cuda', etc.). Default is None,
            which uses the best available device.

    Returns:
        torch.nn.Module: The YOLOv5x6 model with the specified configurations.

    Example:
        ```python
        from ultralytics import yolov5x6

        # Load the model with default settings
        model = yolov5x6()

        # Load the model with custom configurations
        model = yolov5x6(pretrained=False, channels=1, classes=10, autoshape=False, device='cuda')
        ```

    Notes:
        For more information, refer to the YOLOv5 repository: https://github.com/ultralytics/yolov5
    """
    return _create("yolov5x6", pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s", help="model name")
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
