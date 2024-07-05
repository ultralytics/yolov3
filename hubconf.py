# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5

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
    Creates or loads a YOLOv3 model, with options for pretrained weights and model customization.

    Args:
        name (str): Model name (e.g., 'yolov5s') or path to the model checkpoint (e.g., 'path/to/best.pt').
        pretrained (bool, optional): If True, loads pretrained weights into the model. Defaults to True.
        channels (int, optional): Number of input channels the model expects. Defaults to 3.
        classes (int, optional): Number of classes for model outputs. Defaults to 80.
        autoshape (bool, optional): If True, applies YOLOv3 .autoshape() wrapper to the model, allowing various input formats.
            Defaults to True.
        verbose (bool, optional): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model parameters (e.g., 'cpu', 'cuda'). If None,
            automatically selects the device. Defaults to None.

    Returns:
        YOLOv3: An instance of the YOLOv3 model configured as specified.

    Note:
        For more details on how to use and load models with PyTorch Hub, see the tutorial at:
        https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load an official model
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # Load from a specific branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # Load a custom/local model
        model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # Load from a local repository
        ```
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
        path (str): The file path to the custom YOLOv3 model. Typically ends in '.pt'.
        autoshape (bool): Whether to apply YOLOv3's .autoshape() wrapper for convenient inference with
            file/URI/PIL/cv2/np inputs followed by non-max suppression (NMS). Defaults to True.
        _verbose (bool): If True, prints all relevant information to the console. Defaults to True.
        device (str | torch.device | None): The device on which to load the model (e.g., 'cpu', 'cuda:0').
            If None, defaults to the best available device.

    Returns:
        torch.nn.Module: The loaded YOLOv3 model, ready for inference.

    Usage:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/model.pt')
        model = torch.hub.load('.', 'custom', path='path/to/local_model.pt', source='local')
        ```

    See Also:
        - PyTorch Hub models documentation: https://pytorch.org/hub/ultralytics_yolov5
        - Ultralytics GitHub repository: https://github.com/ultralytics/ultralytics
    """
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates a YOLOv5n model with options for pretrained weights, class/channel count, autoshaping, and device
    selection.

    Args:
        pretrained (bool, optional): If True, loads the pretrained weights into the model. Defaults to True.
        channels (int, optional): Number of input channels for the model. Defaults to 3.
        classes (int, optional): Number of output classes for the model. Defaults to 80.
        autoshape (bool, optional): If True, applies the YOLOv3 .autoshape() wrapper to the model. Defaults to True.
        _verbose (bool, optional): If True, prints detailed information during the model creation. Defaults to True.
        device (str | torch.device | None, optional): Device on which to load the model, e.g., 'cpu', 'cuda', or a torch.device object. Defaults to None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5n model with specified configurations.

    Examples:
        ```python
        import torch

        # Load the YOLOv5n model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

        # Load YOLOv5n model on a specific device, disabling autoshape
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cuda', autoshape=False)
        ```

    Notes:
        - For more information on usage, visit the official PyTorch Hub models page: https://pytorch.org/hub/ultralytics_yolov5
        - Custom local models can be loaded by specifying the path using the `custom` function.
    """
    return _create("yolov5n", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5s model with options for pretrained weights, input channel and class customization, autoshaping, and
    device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): The number of input channels. Defaults to 3.
        classes (int): The number of model classes. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper for easier inference with various input types. Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): The device on which to load the model parameters, e.g., 'cpu', 'cuda', or specific device like 'cuda:0'. Defaults to None.

    Returns:
        torch.nn.Module: The loaded YOLOv5s model.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)  # Load model without pretrained weights
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=90)  # Load model with custom number of classes
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')  # Load model on specific device
        ```

    Notes:
        - For more details on model loading, visit https://pytorch.org/hub/ultralytics_yolov5.
    """
    return _create("yolov5s", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads YOLOv5m model with options for pretrained weights, input channel and class numbers, autoshaping, device
    assignment, and verbosity.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights. Defaults to True.
        channels (int, optional): Number of input channels. Defaults to 3.
        classes (int, optional): Number of model classes. Defaults to 80.
        autoshape (bool, optional): If True, applies YOLOv5 .autoshape() wrapper for various input formats and NMS.
            Defaults to True.
        _verbose (bool, optional): If True, prints detailed information during model loading. Defaults to True.
        device (str | torch.device | None, optional): Specifies device for model parameters (e.g., 'cpu', 'cuda:0').
            Defaults to None.

    Returns:
        torch.nn.Module: Loaded YOLOv5m model instance.

    Usage:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # official YOLOv5m model
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/custom_model.pt')  # custom YOLOv5 model
        ```
    """
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads YOLOv5l model, with options for pretrained weights, channel/class customization, autoshaping, and device
    choice.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels. Defaults to 3.
        classes (int): Number of model classes. Defaults to 80.
        autoshape (bool): If True, applies YOLOv3 `.autoshape()` wrapper to model. Defaults to True.
        _verbose (bool): If True, prints all information to the screen. Defaults to True.
        device (str | torch.device | None): Device to use for model parameters. Defaults to None, which selects the
            default device.

    Returns:
        torch.nn.Module: The YOLOv5l model configured with the specified parameters.

    Examples:
        ```python
        import torch

        # Load YOLOv5l model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

        # Load YOLOv5l model with custom device and channel settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, channels=1, device='cuda')
        ```

    Notes:
        For more information, visit the PyTorch Hub models documentation:
        https://pytorch.org/hub/ultralytics_yolov5
    """
    return _create("yolov5l", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads YOLOv5x model with customization options for pretrained weights, input channels, number of classes,
    autoshaping, and device selection.

    Args:
        pretrained (bool): Whether to load pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels. Defaults to 3.
        classes (int): Number of model classes. Defaults to 80.
        autoshape (bool): Whether to apply the YOLOv5 .autoshape() wrapper to the model for various input formats.
            Defaults to True.
        _verbose (bool): Whether to print detailed information to the screen. Defaults to True.
        device (str | torch.device | None): The device to load the model onto. Can be a string (e.g., 'cuda', 'cpu'),
            a torch.device object, or None. If None, the default device will be selected. Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5x model loaded with the specified configurations.

    Note:
        For additional guidance and examples, refer to the PyTorch Hub documentation:
        https://pytorch.org/hub/ultralytics_yolov5

    Example:
        ```python
        import torch

        # Load YOLOv5x model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

        # Load YOLOv5x model with custom settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=False, channels=1, classes=10, autoshape=False, device='cpu')
        ```
    """
    return _create("yolov5x", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5n6 model with options for pretrained weights, input channels, number of classes, autoshaping, and
    device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights. Default is True.
        channels (int): Number of input channels for the model. Default is 3.
        classes (int): Number of object classes for the model. Default is 80.
        autoshape (bool): If True, applies the YOLOv3 `.autoshape()` wrapper to the model. Default is True.
        _verbose (bool): If True, prints detailed information during model loading. Default is True.
        device (str | torch.device | None): Device designation for model parameters. Default is None.

    Returns:
        torch.nn.Module: The YOLOv5n6 model.

    Usage:
        ```python
        from ultralytics import yolov5n6

        model = yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device='cuda')
        ```

    Refer to the PyTorch Hub documentation for more information: https://pytorch.org/hub/ultralytics_yolov5
    """
    return _create("yolov5n6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5s6 model with customization options for pretrained weights, input channels, class count,
    autoshaping, and device selection.

    Args:
        pretrained (bool, optional): If True, loads the model with pretrained weights. Defaults to True.
        channels (int, optional): Number of input channels. Defaults to 3.
        classes (int, optional): Number of model classes. Defaults to 80.
        autoshape (bool, optional): If True, applies YOLOv3 .autoshape() wrapper for various input types and NMS. Defaults to True.
        _verbose (bool, optional): If True, prints detailed information during the model creation process. Defaults to True.
        device (str | torch.device | None, optional): Specifies the device for model parameters, e.g., 'cpu', 'cuda', or torch.device object. Defaults to None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5s6 model.

    Example:
        ```python
        import torch

        # Load YOLOv5s6 model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')

        # Load YOLOv5s6 model with custom settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=False, channels=1, classes=10, autoshape=False, device='cuda')
        ```

    See Also:
        PyTorch Hub models: https://pytorch.org/hub/ultralytics_yolov5
        Ultralytics YOLOv5 GitHub repository: https://github.com/ultralytics/yolov5
    """
    return _create("yolov5s6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5m6 model with customization options for weights, channels, classes, autoshaping, and device
    selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): The number of input channels for the model. Defaults to 3.
        classes (int): The number of classes for the model to predict. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper to the model for file/URI/PIL/cv2/np inputs
            and NMS. Defaults to True.
        _verbose (bool): If True, prints detailed information during model loading. Defaults to True.
        device (str | torch.device | None): The device on which to load the model. If None, the default device is used.
            Defaults to None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5m6 model with specified options.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
        # Load the YOLOv5m6 model with pretrained weights
        ```

    For more details, see https://github.com/ultralytics/yolov5
    """
    return _create("yolov5m6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5l6 model with options for pretrained weights, input channels, number of classes, autoshaping, and
    device assignment.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights. Defaults to True.
        channels (int, optional): Number of input channels. Typically, 3 for RGB images. Defaults to 3.
        classes (int, optional): Number of output classes for the model. Defaults to 80.
        autoshape (bool, optional): If True, applies the YOLOv5 .autoshape() wrapper for convenient inference handling.
                                     Defaults to True.
        _verbose (bool, optional): If True, prints all information to the screen. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model parameters. Can be a string (e.g., 'cpu',
                                                      'cuda'), a torch.device object, or None for automatic selection.
                                                      Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5l6 model loaded onto the specified device.

    Example:
        ```python
        import torch

        # Load YOLOv5l6 model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
        # Custom/local model loading
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/yolov5l6.pt')
        ```
    """
    return _create("yolov5l6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Loads the YOLOv5x6 model, providing options for pretrained weights, input channels, number of classes, autoshaping,
    and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model (default is True).
        channels (int): Number of input channels for the model. Typically 3 for RGB images (default is 3).
        classes (int): Number of classes the model should output (default is 80).
        autoshape (bool): If True, applies the YOLOv3 .autoshape() wrapper to the model, allowing for input types
                          like file paths, URIs, PIL images, OpenCV images, and NumPy arrays alongside NMS
                          (default is True).
        _verbose (bool): If True, prints detailed information to the screen during model loading (default is True).
        device (str | torch.device | None): Specifies the device on which to load the model parameters.
                                            If None, the best available device (CUDA or CPU) will be selected (default is None).

    Returns:
        torch.nn.Module: The loaded YOLOv5x6 model.

    Example:
        ```python
        import torch
        from ultralytics import yolov5x6

        # Load a pretrained YOLOv5x6 model
        model = yolov5x6(pretrained=True, channels=3, classes=80)

        # Perform inference
        results = model("path/to/image.jpg")
        results.print()
        ```

    Notes:
        For more information about the model architecture and usage, visit the YOLOv5 GitHub repository:
        https://github.com/ultralytics/yolov5.
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
