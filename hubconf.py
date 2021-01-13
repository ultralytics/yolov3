"""File for accessing YOLOv3 via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True, channels=3, classes=80)
"""

from pathlib import Path

import torch

from models.yolo import Model
from utils.general import set_logging
from utils.google_utils import attempt_download

dependencies = ['torch', 'yaml']
set_logging()


def create(name, pretrained, channels, classes, autoshape):
    """Creates a specified YOLOv3 model

    Arguments:
        name (str): name of model, i.e. 'yolov3_spp'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    config = Path(__file__).parent / 'models' / f'{name}.yaml'  # model.yaml path
    try:
        model = Model(config, channels, classes)
        if pretrained:
            fname = f'{name}.pt'  # checkpoint filename
            attempt_download(fname)  # download if not found locally
            ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
            model.load_state_dict(state_dict, strict=False)  # load
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names  # set class names attribute
            if autoshape:
                model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache maybe be out of date, try force_reload=True. See %s for help.' % help_url
        raise Exception(s) from e


def yolov3(pretrained=False, channels=3, classes=80, autoshape=True):
    """YOLOv3 model from https://github.com/ultralytics/yolov3

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov3', pretrained, channels, classes, autoshape)


def yolov3_spp(pretrained=False, channels=3, classes=80, autoshape=True):
    """YOLOv3-SPP model from https://github.com/ultralytics/yolov3

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov3-spp', pretrained, channels, classes, autoshape)


def yolov3_tiny(pretrained=False, channels=3, classes=80, autoshape=True):
    """YOLOv3-tiny model from https://github.com/ultralytics/yolov3

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov3-tiny', pretrained, channels, classes, autoshape)


def custom(path_or_model='path/to/model.pt', autoshape=True):
    """YOLOv3-custom model from https://github.com/ultralytics/yolov3

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    return hub_model.autoshape() if autoshape else hub_model


if __name__ == '__main__':
    model = create(name='yolov3', pretrained=True, channels=3, classes=80, autoshape=True)  # pretrained example
    # model = custom(path_or_model='path/to/model.pt')  # custom example

    # Verify inference
    from PIL import Image

    imgs = [Image.open(x) for x in Path('data/images').glob('*.jpg')]
    results = model(imgs)
    results.show()
    results.print()
