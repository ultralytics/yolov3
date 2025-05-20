# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv3 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
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
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Run YOLOv3 detection inference on various input sources such as images, videos, streams, and YouTube URLs.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL (default: 'yolov5s.pt').
        source (str | Path): Source of input data such as a file, directory, URL, glob pattern, or device identifier
            (default: 'data/images').
        data (str | Path): Path to the dataset YAML file (default: 'data/coco128.yaml').
        imgsz (tuple[int, int]): Inference size as a tuple (height, width) (default: (640, 640)).
        conf_thres (float): Confidence threshold for detection (default: 0.25).
        iou_thres (float): Intersection Over Union (IOU) threshold for Non-Max Suppression (NMS) (default: 0.45).
        max_det (int): Maximum number of detections per image (default: 1000).
        device (str): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu' (default: '').
        view_img (bool): Whether to display results during inference (default: False).
        save_txt (bool): Whether to save detection results to text files (default: False).
        save_conf (bool): Whether to save detection confidences in the text labels (default: False).
        save_crop (bool): Whether to save cropped detection boxes (default: False).
        nosave (bool): Whether to prevent saving images or videos with detections (default: False).
        classes (list[int] | None): List of class indices to filter, e.g., [0, 2, 3] (default: None).
        agnostic_nms (bool): Whether to perform class-agnostic NMS (default: False).
        augment (bool): Whether to apply augmented inference (default: False).
        visualize (bool): Whether to visualize feature maps (default: False).
        update (bool): Whether to update all models (default: False).
        project (str | Path): Path to the project directory where results will be saved (default: 'runs/detect').
        name (str): Name for the specific run within the project directory (default: 'exp').
        exist_ok (bool): Whether to allow existing project/name directory without incrementing run index (default: False).
        line_thickness (int): Thickness of bounding box lines in pixels (default: 3).
        hide_labels (bool): Whether to hide labels in the results (default: False).
        hide_conf (bool): Whether to hide confidences in the results (default: False).
        half (bool): Whether to use half-precision (FP16) for inference (default: False).
        dnn (bool): Whether to use OpenCV DNN for ONNX inference (default: False).
        vid_stride (int): Stride for video frame rate (default: 1).

    Returns:
        None

    Notes:
        This function supports a variety of input sources such as image files, video files, directories, URL patterns,
        webcam streams, and YouTube links. It also supports multiple model formats including PyTorch, ONNX, OpenVINO,
        TensorRT, CoreML, TensorFlow, PaddlePaddle, and others. The results can be visualized in real-time or saved to
        specified directories. Use command-line arguments to modify the behavior of the function.

    Examples:
        ```python
        # Run YOLOv3 inference on an image
        run(weights='yolov5s.pt', source='data/images/bus.jpg')

        # Run YOLOv3 inference on a video
        run(weights='yolov5s.pt', source='data/videos/video.mp4', view_img=True)

        # Run YOLOv3 inference on a webcam
        run(weights='yolov5s.pt', source='0', view_img=True)
        ```
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """
    Parses and returns command-line options for running YOLOv3 model detection.

    Args:
        --weights (list[str]): Model path or Triton URL. Default: ROOT / "yolov3-tiny.pt".
        --source (str): Input data source like file/dir/URL/glob/screen/0(webcam). Default: ROOT / "data/images".
        --data (str): Optional path to dataset.yaml. Default: ROOT / "data/coco128.yaml".
        --imgsz (list[int]): Inference size as height, width. Accepts multiple values. Default: [640].
        --conf-thres (float): Confidence threshold for predictions. Default: 0.25.
        --iou-thres (float): IoU threshold for Non-Maximum Suppression (NMS). Default: 0.45.
        --max-det (int): Maximum number of detections per image. Default: 1000.
        --device (str): CUDA device identifier, e.g. "0" or "0,1,2,3" or "cpu". Default: "" (auto-select).
        --view-img (bool): Display results. Default: False.
        --save-txt (bool): Save results to *.txt files. Default: False.
        --save-conf (bool): Save confidence scores in text labels. Default: False.
        --save-crop (bool): Save cropped prediction boxes. Default: False.
        --nosave (bool): Do not save images/videos. Default: False.
        --classes (list[int] | None): Filter results by class, e.g. [0, 2, 3]. Default: None.
        --agnostic-nms (bool): Perform class-agnostic NMS. Default: False.
        --augment (bool): Apply augmented inference. Default: False.
        --visualize (bool): Visualize feature maps. Default: False.
        --update (bool): Update all models. Default: False.
        --project (str): Directory to save results; results saved to "project/name". Default: ROOT / "runs/detect".
        --name (str): Name of the specific run; results saved to "project/name". Default: "exp".
        --exist-ok (bool): Allow results to be saved in an existing directory without incrementing. Default: False.
        --line-thickness (int): Bounding box line thickness in pixels. Default: 3.
        --hide-labels (bool): Hide labels on detections. Default: False.
        --hide-conf (bool): Hide confidence scores on labels. Default: False.
        --half (bool): Use FP16 half-precision inference. Default: False.
        --dnn (bool): Use OpenCV DNN backend for ONNX inference. Default: False.
        --vid-stride (int): Frame-rate stride for video input. Default: 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments for YOLOv3 inference configurations.

    Example:
        ```python
        options = parse_opt()
        run(**vars(options))
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default=ROOT / "yolov3-tiny.pt", help="model path or triton URL"
    )
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Entry point for running the YOLO model; checks requirements and calls `run` with parsed options.

    Args:
        opt (argparse.Namespace): Parsed command-line options, which include:
            - weights (str | list of str): Path to the model weights or Triton server URL.
            - source (str): Input source, can be a file, directory, URL, glob, screen, or webcam index.
            - data (str): Path to the dataset configuration file (.yaml).
            - imgsz (tuple of int): Inference image size as (height, width).
            - conf_thres (float): Confidence threshold for detections.
            - iou_thres (float): Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).
            - max_det (int): Maximum number of detections per image.
            - device (str): Device to run inference on; options are CUDA device id(s) or 'cpu'
            - view_img (bool): Flag to display inference results.
            - save_txt (bool): Save detection results in .txt format.
            - save_conf (bool): Save detection confidences in .txt labels.
            - save_crop (bool): Save cropped bounding box predictions.
            - nosave (bool): Do not save images/videos with detections.
            - classes (list of int): Filter results by class, e.g., --class 0 2 3.
            - agnostic_nms (bool): Use class-agnostic NMS.
            - augment (bool): Enable augmented inference.
            - visualize (bool): Visualize feature maps.
            - update (bool): Update the model during inference.
            - project (str): Directory to save results.
            - name (str): Name for the results directory.
            - exist_ok (bool): Allow existing project/name directories without incrementing.
            - line_thickness (int): Thickness of bounding box lines.
            - hide_labels (bool): Hide class labels on bounding boxes.
            - hide_conf (bool): Hide confidence scores on bounding boxes.
            - half (bool): Use FP16 half-precision inference.
            - dnn (bool): Use OpenCV DNN backend for ONNX inference.
            - vid_stride (int): Video frame-rate stride.

    Returns:
        None

    Example:
        ```python
        if __name__ == "__main__":
            opt = parse_opt()
            main(opt)
        ```

    Notes:
        Run this function as the entry point for using YOLO for object detection on a variety of input sources such as
        images, videos, directories, webcams, streams, etc. This function ensures all requirements are checked and
        subsequently initiates the detection process by calling the `run` function with appropriate options.
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
