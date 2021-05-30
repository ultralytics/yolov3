"""YOLOv3 PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov3/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov3', 'yolov3_tiny')
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates a specified YOLOv3 model

    Arguments:
        name (str): name of model, i.e. 'yolov3'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv3 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv3 pytorch model
    """
    from pathlib import Path

    from models.yolo import Model, attempt_load
    from utils.general import check_requirements, set_logging
    from utils.google_utils import attempt_download
    from utils.torch_utils import select_device

    check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('tensorboard', 'pycocotools', 'thop'))
    set_logging(verbose=verbose)

    fname = Path(name).with_suffix('.pt')  # checkpoint filename
    try:
        if pretrained and channels == 3 and classes == 80:
            model = attempt_load(fname, map_location=torch.device('cpu'))  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(fname), map_location=torch.device('cpu'))  # load
                msd = model.state_dict()  # model state_dict
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        device = select_device('0' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, verbose=True, device=None):
    # YOLOv3 custom or local model
    return _create(path, autoshape=autoshape, verbose=verbose, device=device)


def yolov3(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv3 model https://github.com/ultralytics/yolov3
    return _create('yolov3', pretrained, channels, classes, autoshape, verbose, device)

def yolov3_spp(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv3-SPP model https://github.com/ultralytics/yolov3
    return _create('yolov3-spp', pretrained, channels, classes, autoshape, verbose, device)

def yolov3_tiny(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv3-tiny model https://github.com/ultralytics/yolov3
    return _create('yolov3-tiny', pretrained, channels, classes, autoshape, verbose, device)


if __name__ == '__main__':
    model = _create(name='yolov3', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)  # pretrained
    # model = custom(path='path/to/model.pt')  # custom

    # Verify inference
    import cv2
    import numpy as np
    from PIL import Image

    imgs = ['data/images/zidane.jpg',  # filename
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg',  # URI
            cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
            Image.open('data/images/bus.jpg'),  # PIL
            np.zeros((320, 640, 3))]  # numpy

    results = model(imgs)  # batched inference
    results.print()
    results.save()
