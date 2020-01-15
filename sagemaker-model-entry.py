import numpy as np
from PIL import Image

import cv2
import json
import torch

from utils.datasets import *
from utils.utils import *
from models import *

JSON_CONTENT_TYPE   = 'application/json'
JPEG_CONTENT_TYPE   = 'image/jpeg'
DEFAULT_WEIGHT_LOC  = 'weights/best.pt'
DEFAULT_CFG_LOC     = 'cfg/yolov3-single-cfg.cfg'
DEFAULT_DAT_LOC     = 'cfg/code.data'
DEFAULT_IMG_SIZE    = 416
DEFAULT_DEVICE      = ''
DEFAULT_CONF_THRESH = 0.3
DEFAULT_IOU_THRESH  = 0.5

def model_fn(model_dir):
    img_size = DEFAULT_IMG_SIZE
    weights  = DEFAULT_WEIGHT_LOC
    device   = DEFAULT_DEVICE

    # Initialize
    device = torch_utils.select_device(device)
    model = Darknet(DEFAULT_CFG_LOC, img_size)

    # Load the weights.
    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(os.path.join(model_dir, weights), map_location=device)['model'])

    model.to(device).eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    raw_img = Image.open(io.BytesIO(request_body))
    raw_img = np.asarray(raw_img)
    img = letterbox(raw_img)[0]

    img = img[:, :, :].transpose(2, 0, 1) # W, H, C -> C, W, H
    img = img[0:3] # Remove alpha channel.
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return {'img': img, 'raw_img': raw_img, 'size': img.size}

def predict_fn(input_object, model):
    img = input_object['img']
    
    device = torch_utils.select_device(DEFAULT_DEVICE)
    img = torch.from_numpy(img).to(device)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Get a prediction.
    pred = model(img)[0]
    return {'pred': pred, 'img': img, 'raw_img': input_object['raw_img']}

def output_fn(prediction, content_type=JSON_CONTENT_TYPE):
    pred     = prediction['pred']
    img      = prediction['img']
    raw_img  = prediction['raw_img']

    # TODO: Handle time_ms in future..
    results = {
        'containers': []
    }

    pred = non_max_suppression(pred, DEFAULT_CONF_THRESH, DEFAULT_IOU_THRESH)

    # Process detections
    for _, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_img.shape).round()
            for *xyxy, conf, _ in det:
                results['containers'].append({
                    'box': {
                        'x': int(xyxy[0]),
                        'y': int(xyxy[1]),
                        'width': int(xyxy[2]),
                        'height': int(xyxy[3])
                    },
                    'confidence': conf.tolist()
                })

    results['containers'] = sorted(results['containers'], key = lambda res: res['confidence'], reverse=True)
    return json.dumps(results)