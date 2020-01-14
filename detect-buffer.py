import argparse
import json
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

# Predict bounding boxes from a raw img buffer.
def detect_buffer(buffer):
    t_start = time.time()

    img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    source, weights, half = opt.source, opt.weights, opt.half

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    dataset = LoadImages(source, img_size=img_size, half=half)

    # Final output structure.
    results = {
        'time_ms': '',
        'containers': []
    }

    # Run inference
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for _, det in enumerate(pred):  # detections per image
            _, _, im0 = path, '', im0s

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, _, _ in det:
                    # Add new bounding box to result.
                    box_coords = [x.tolist() for x in xyxy]
                    results['containers'].append({
                        'box': {
                            'x': int(box_coords[0]),
                            'y': int(box_coords[1]),
                            'width': int(box_coords[2]),
                            'height': int(box_coords[3])
                        },
                        'confidence': conf.tolist()
                    })

    results['containers'] = sorted(results['containers'], key = lambda res: res['confidence'], reverse=True)

    results['time_ms'] = int((time.time() - t0) * 1000)
    print(json.dumps(results))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect_buffer([])
