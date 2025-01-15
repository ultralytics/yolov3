# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv3 benchmarks on all supported export formats.

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU
    $ pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com  # TensorRT

Usage:
    $ python benchmarks.py --weights yolov5s.pt --img 640
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import export
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from segment.val import run as val_seg
from utils import notebook_init
from utils.general import LOGGER, check_yaml, file_size, print_args
from utils.torch_utils import select_device
from val import run as val_det


def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=640,  # inference size (pixels)
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    test=False,  # test exports only
    pt_only=False,  # test PyTorch only
    hard_fail=False,  # throw error on benchmark failure
):
    """
    Run YOLOv3 benchmarks on multiple export formats and validate performance metrics.

    Args:
        weights (str | Path): Path to the weights file. Defaults to 'yolov5s.pt'.
        imgsz (int): Inference image size in pixels. Defaults to 640.
        batch_size (int): Batch size for inference. Defaults to 1.
        data (str | Path): Path to the dataset configuration file (dataset.yaml). Defaults to 'data/coco128.yaml'.
        device (str): Device to be used for inference, e.g., '0' or '0,1,2,3' for GPU or 'cpu' for CPU. Defaults to ''.
        half (bool): Use FP16 half-precision for inference. Defaults to False.
        test (bool): Test exports only without running benchmarks. Defaults to False.
        pt_only (bool): Run benchmarks only for PyTorch format. Defaults to False.
        hard_fail (bool): Raise an error if any benchmark test fails. Defaults to False.

    Returns:
        None

    Notes:
        This function iterates over multiple export formats, performs the export, and then validates the model's performance
        using appropriate validation functions for detection and segmentation models. The results are logged, and optionally,
        benchmarks can be configured to raise errors on failures using the `hard_fail` argument.

    Examples:
        ```python
        # Run benchmarks on the default 'yolov5s.pt' model with an image size of 640 pixels
        run()

        # Run benchmarks on a specific model with GPU and half-precision enabled
        run(weights='custom_model.pt', device='0', half=True)

        # Test only PyTorch export
        run(pt_only=True)
        ```
    """
    y, t = [], time.time()
    device = select_device(device)
    model_type = type(attempt_load(weights, fuse=False))  # DetectionModel, SegmentationModel, etc.
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, CPU, GPU)
        try:
            assert i not in (9, 10), "inference not supported"  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if f == "-":
                w = weights  # PyTorch format
            else:
                w = export.run(
                    weights=weights, imgsz=[imgsz], include=[f], batch_size=batch_size, device=device, half=half
                )[-1]  # all others
            assert suffix in str(w), "export failed"

            # Validate
            if model_type == SegmentationModel:
                result = val_seg(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][7]  # (box(p, r, map50, map), mask(p, r, map50, map), *loss(box, obj, cls))
            else:  # DetectionModel:
                result = val_det(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][3]  # (p, r, map50, map, *loss(box, obj, cls))
            speed = result[2][1]  # times (preprocess, inference, postprocess)
            y.append([name, round(file_size(w), 1), round(metric, 4), round(speed, 2)])  # MB, mAP, t_inference
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f"Benchmark --hard-fail for {name}: {e}"
            LOGGER.warning(f"WARNING ⚠️ Benchmark failure for {name}: {e}")
            y.append([name, None, None, None])  # mAP, t_inference
        if pt_only and i == 0:
            break  # break after PyTorch

    # Print results
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  # print system info
    c = ["Format", "Size (MB)", "mAP50-95", "Inference time (ms)"] if map else ["Format", "Export", "", ""]
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f"\nBenchmarks complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py if map else py.iloc[:, :2]))
    if hard_fail and isinstance(hard_fail, str):
        metrics = py["mAP50-95"].array  # values to compare to floor
        floor = eval(hard_fail)  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f"HARD FAIL: mAP50-95 < floor {floor}"
    return py


def test(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=640,  # inference size (pixels)
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    test=False,  # test exports only
    pt_only=False,  # test PyTorch only
    hard_fail=False,  # throw error on benchmark failure
):
    """
    Run YOLOv3 export tests for various formats and log the results, including export success status.

    Args:
        weights (str | Path): Path to the weights file. Defaults to ROOT / "yolov5s.pt".
        imgsz (int): Inference size in pixels. Defaults to 640.
        batch_size (int): Number of images per batch. Defaults to 1.
        data (str | Path): Path to the dataset yaml file. Defaults to ROOT / "data/coco128.yaml".
        device (str): Device for inference. Accepts cuda device (e.g., "0" or "0,1,2,3") or "cpu". Defaults to "".
        half (bool): Use FP16 half-precision inference. Defaults to False.
        test (bool): Run export tests only, no inference. Defaults to False.
        pt_only (bool): Run tests on PyTorch format only. Defaults to False.
        hard_fail (bool): Raise an error on benchmark failure. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the export formats and their success status.

    Examples:
        ```python
        from ultralytics import test

        results = test(
            weights="path/to/yolov5s.pt",
            imgsz=640,
            batch_size=1,
            data="path/to/coco128.yaml",
            device="0",
            half=False,
            test=True,
            pt_only=False,
            hard_fail=True,
        )
        print(results)
        ```

    Notes:
        Ensure all required packages are installed as specified in the Ultralytics YOLOv3 documentation:
        https://github.com/ultralytics/ultralytics
    """
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, gpu-capable)
        try:
            w = (
                weights
                if f == "-"
                else export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]
            )  # weights
            assert suffix in str(w), "export failed"
            y.append([name, True])
        except Exception:
            y.append([name, False])  # mAP, t_inference

    # Print results
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  # print system info
    py = pd.DataFrame(y, columns=["Format", "Export"])
    LOGGER.info(f"\nExports complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py))
    return py


def parse_opt():
    """
    Parses command line arguments for YOLOv3 inference and export configurations.

    Args:
        --weights (str): Path to the weights file. Default is 'ROOT / "yolov3-tiny.pt"'.
        --imgsz | --img | --img-size (int): Inference image size in pixels. Default is 640.
        --batch-size (int): Batch size for inference. Default is 1.
        --data (str): Path to the dataset configuration file (dataset.yaml). Default is 'ROOT / "data/coco128.yaml"'.
        --device (str): CUDA device identifier, e.g., '0' for single GPU, '0,1,2,3' for multiple GPUs, or 'cpu' for CPU
            inference. Default is "".
        --half (bool): If set, use FP16 half-precision inference. Default is False.
        --test (bool): If set, test only exports without running inference. Default is False.
        --pt-only (bool): If set, test only the PyTorch model without exporting to other formats. Default is False.
        --hard-fail (str | bool): If set, raise an exception on benchmark failure. Can also be a string representing the
            minimum metric floor for success. Default is False.

    Returns:
        argparse.Namespace: The parsed arguments as a namespace object.

    Example:
        To run inference on the YOLOv3-tiny model with a different image size:

        ```python
        $ python benchmarks.py --weights yolov3-tiny.pt --imgsz 512 --device 0
        ```

    Notes:
        The `--hard-fail` argument can be a boolean or a string. If a string is provided, it should be an expression that
        represents the minimum acceptable metric value, such as '0.29' for mAP (mean Average Precision).

    Links:
        https://github.com/ultralytics/ultralytics
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov3-tiny.pt", help="weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--test", action="store_true", help="test exports only")
    parser.add_argument("--pt-only", action="store_true", help="test PyTorch only")
    parser.add_argument("--hard-fail", nargs="?", const=True, default=False, help="Exception on error or < min metric")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes the export and benchmarking pipeline for YOLOv3 models, testing multiple export formats and validating
    performance metrics.

    Args:
        opt (argparse.Namespace): Parsed command line arguments, including options for weights, image size, batch size,
            dataset path, device, half-precision inference, test mode, PyTorch-only testing, and hard fail conditions.

    Returns:
        pd.DataFrame: A DataFrame containing benchmarking results with columns:
            - Format: Name of the export format
            - Size (MB): File size of the exported model
            - mAP50-95: Mean Average Precision for the model
            - Inference time (ms): Time taken for inference

    Notes:
        The function runs the main pipeline by exporting the YOLOv3 model to various formats and running benchmarks to
        evaluate performance. If `opt.test` is set to True, it only tests the export process and logs the results.

    Example:
        Running the function from command line with required arguments:

        ```python
        $ python benchmarks.py --weights yolov5s.pt --img 640
        ```

    For more details, visit the Ultralytics YOLOv3 repository on [GitHub](https://github.com/ultralytics/ultralytics).
    """
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
