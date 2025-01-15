# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate a trained YOLOv3 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """
    Saves detection results in a text format, including labels and optionally confidence scores.

    Args:
        predn (torch.Tensor): A tensor containing normalized prediction results in the format (x1, y1, x2, y2, conf, cls).
        save_conf (bool): A flag indicating whether to save confidence scores.
        shape (tuple[int, int]): Original image shape in the format (height, width).
        file (str | Path): Path to the file where the results will be saved.

    Returns:
        None

    Example:
        ```python
        from pathlib import Path
        import torch

        predn = torch.tensor([
            [10, 20, 100, 200, 0.9, 1],
            [30, 40, 150, 250, 0.8, 0],
        ])
        save_conf = True
        shape = (416, 416)
        file = Path("results.txt")

        save_one_txt(predn, save_conf, shape, file)
        ```

    Notes:
        - The function normalizes bounding box coordinates before saving.
        - Each line in the output file will contain class, x-center, y-center, width, height and optionally confidence score.
        - The format is compatible with YOLO training dataset format.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    """
    Save detection results in JSON format containing image_id, category_id, bbox, and score per detection.

    Args:
        predn (torch.Tensor): Normalized prediction tensor of shape (N, 6) where N is the number of detections.
            Each detection should contain (x1, y1, x2, y2, confidence, class).
        jdict (list): List to store the JSON serializable detections.
        path (Path): Path object representing the image file path.
        class_map (dict[int, int]): Dictionary mapping class indices to their respective category IDs.

    Returns:
        None

    Example:
        ```python
        predn = torch.tensor([[50, 30, 200, 150, 0.9, 0], [30, 20, 180, 150, 0.8, 1]])
        jdict = []
        path = Path('images/000001.jpg')
        class_map = {0: 1, 1: 2}
        save_one_json(predn, jdict, path, class_map)
        ```

    Notes:
        - The image_id is extracted from the image file path.
        - Bounding boxes are converted from xyxy format to xywh format.
        - The JSON output format is compatible with COCO dataset evaluation.
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Computes correct prediction matrix for detections against ground truth labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of detections with shape (N, 6), where each detection contains [x1, y1, x2, y2,
            confidence, class].
        labels (np.ndarray): Array of ground truth labels with shape (M, 5), where each label contains [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to use for evaluation.

    Returns:
        np.ndarray: Boolean array of shape (N, len(iouv)), indicating correct predictions at each IoU threshold.

    Notes:
        - This function compares detections and ground truth labels to establish matches based on IoU and class.
        - It supports multiple IoU thresholds to evaluate prediction accuracy flexibly.

    Example:
        ```python
        detections = np.array([[50, 50, 150, 150, 0.8, 0],
                               [30, 30, 120, 120, 0.7, 1]])
        labels = np.array([[0, 50, 50, 150, 150],
                           [1, 30, 30, 120, 120]])
        iouv = np.array([0.5, 0.6, 0.7])

        correct = process_batch(detections, labels, iouv)
        ```
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
):
    """
    Validates a trained YOLO model on a dataset and saves detection results in specified formats.

    Args:
        data (str | dict): Path to the dataset configuration file (.yaml) or a dictionary containing the dataset paths.
        weights (str | list, optional): Path to the trained model weights file(s). Default is None.
        batch_size (int, optional): Batch size for inference. Default is 32.
        imgsz (int, optional): Input image size for inference in pixels. Default is 640.
        conf_thres (float, optional): Confidence threshold for object detection. Default is 0.001.
        iou_thres (float, optional): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.6.
        max_det (int, optional): Maximum number of detections per image. Default is 300.
        task (str, optional): Task type, can be 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        device (str, optional): Device for computation, e.g., '0' for GPU or 'cpu' for CPU. Default is "".
        workers (int, optional): Number of dataloader workers. Default is 8.
        single_cls (bool, optional): Whether to treat the dataset as a single-class dataset. Default is False.
        augment (bool, optional): Whether to apply augmented inference. Default is False.
        verbose (bool, optional): Whether to output verbose information. Default is False.
        save_txt (bool, optional): Whether to save detection results in text format (*.txt). Default is False.
        save_hybrid (bool, optional): Whether to save hybrid results (labels+predictions) in text format (*.txt). Default is False.
        save_conf (bool, optional): Whether to save confidence scores in text format labels. Default is False.
        save_json (bool, optional): Whether to save detection results in COCO JSON format. Default is False.
        project (str | Path, optional): Directory path to save validation results. Default is ROOT / 'runs/val'.
        name (str, optional): Directory name to save validation results. Default is 'exp'.
        exist_ok (bool, optional): Whether to overwrite existing project/name directory. Default is False.
        half (bool, optional): Whether to use half-precision (FP16) for inference. Default is True.
        dnn (bool, optional): Whether to use OpenCV DNN for ONNX inference. Default is False.
        model (torch.nn.Module, optional): Existing model instance. Default is None.
        dataloader (torch.utils.data.DataLoader, optional): Existing dataloader instance. Default is None.
        save_dir (Path, optional): Path to directory to save results. Default is Path("").
        plots (bool, optional): Whether to generate plots for visual results. Default is True.
        callbacks (Callbacks, optional): Callbacks instance for event handling. Default is Callbacks().
        compute_loss (Callable, optional): Loss function for computing training loss. Default is None.

    Returns:
        (tuple): A tuple containing:
            - metrics (torch.Tensor): Dictionary containing metrics such as precision, recall, mAP, F1 score, etc.
            - times (dict): Dictionary containing times for different parts of the pipeline (e.g., preprocessing, inference, NMS).
            - samples (torch.Tensor): Torch tensor containing validation samples.

    Example:
        ```python
        metrics, times, samples = run(
            data='data/coco.yaml',
            weights='yolov5s.pt',
            batch_size=32,
            imgsz=640,
            conf_thres=0.001,
            iou_thres=0.6,
            max_det=300,
            task='val',
            device='cpu'
        )
        ```
    """
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """
    Parses and returns command-line options for dataset paths, model parameters, and inference settings.

    Args:
        --data (str): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        --weights (list[str]): Paths to one or more model files. Default is 'yolov3-tiny.pt'.
        --batch-size (int): Number of images per batch during inference. Default is 32.
        --imgsz (int): Inference size (pixels). Default is 640.
        --conf-thres (float): Confidence threshold for object detection. Default is 0.001.
        --iou-thres (float): IoU threshold for non-max suppression (NMS). Default is 0.6.
        --max-det (int): Maximum number of detections per image. Default is 300.
        --task (str): Task to perform: 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        --device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu' for using CPU. Default is "".
        --workers (int): Maximum number of dataloader workers (per RANK in DDP mode). Default is 8.
        --single-cls (bool): Treat the dataset as a single-class dataset. Default is False.
        --augment (bool): Apply test-time augmentation during inference. Default is False.
        --verbose (bool): Print mAP by class. Default is False.
        --save-txt (bool): Save detection results in '.txt' format. Default is False.
        --save-hybrid (bool): Save hybrid results containing both label and prediction in '.txt' format. Default is False.
        --save-conf (bool): Save confidence scores in the '--save-txt' labels. Default is False.
        --save-json (bool): Save detection results in COCO JSON format. Default is False.
        --project (str): Project directory to save results. Default is 'runs/val'.
        --name (str): Name of the experiment to save results. Default is 'exp'.
        --exist-ok (bool): Whether to overwrite existing project/name without incrementing. Default is False.
        --half (bool): Use FP16 half-precision during inference. Default is False.
        --dnn (bool): Use OpenCV DNN backend for ONNX inference. Default is False.

    Returns:
        opt (argparse.Namespace): Parsed command-line options.

    Notes:
        - The function uses `argparse` to handle command-line options.
        - It also modifies some options based on specific conditions, such as appending additional flags for saving
        in JSON format and checking for the `coco.yaml` dataset.

    Example:
        Use the following command to run validation with custom settings:
        ```python
        $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov3-tiny.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes model tasks including training, validation, and speed or study benchmarks based on specified options.

    Args:
        opt (argparse.Namespace): Parsed command-line options for dataset paths, model parameters, and inference settings.

    Returns:
        None

    Note:
        This function orchestrates different tasks based on the user input provided through command-line arguments. It supports tasks
        like `train`, `val`, `test`, `speed`, and `study`. Depending on the task, it validates the model on a dataset, performs speed
        benchmarks, or runs mAP benchmarks.

    Examples:
        To validate a trained YOLOv3 model:

        ```bash
        $ python val.py --weights yolov3.pt --data coco.yaml --img 640 --task val
        ```

        For running speed benchmarks:

        ```bash
        $ python val.py --task speed --data coco.yaml --weights yolov3.pt --batch-size 1
        ```

    Links:
        For more information, visit the official repository: https://github.com/ultralytics/ultralytics
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
