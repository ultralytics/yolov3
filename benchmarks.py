# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv3 benchmarks on all supported export formats.

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov3-tiny.pt
TorchScript                 | `torchscript`                 | yolov3-tiny.torchscript
ONNX                        | `onnx`                        | yolov3-tiny.onnx
OpenVINO                    | `openvino`                    | yolov3-tiny_openvino_model/
TensorRT                    | `engine`                      | yolov3-tiny.engine
CoreML                      | `coreml`                      | yolov3-tiny.mlmodel
PaddlePaddle                | `paddle`                      | yolov3-tiny_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnxruntime openvino-dev  # CPU
    $ pip install -r requirements.txt coremltools onnx onnxruntime-gpu openvino-dev  # GPU
    $ pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com  # TensorRT

Usage:
    $ python benchmarks.py --weights yolov3-tiny.pt --img 640
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
from utils import notebook_init
from utils.general import LOGGER, check_yaml, file_size, print_args
from utils.torch_utils import select_device
from val import run as val_det


def run(
    weights=ROOT / "yolov3-tiny.pt",  # weights path
    imgsz=640,  # inference size (pixels)
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    test=False,  # test exports only
    pt_only=False,  # test PyTorch only
    hard_fail=False,  # throw error on benchmark failure
):
    """Run YOLOv3 benchmarks on multiple export formats and validate performance metrics.

    Args:
        weights (str | Path): Path to the weights file. Defaults to 'yolov3-tiny.pt'.
        imgsz (int): Inference image size in pixels. Defaults to 640.
        batch_size (int): Batch size for inference. Defaults to 1.
        data (str | Path): Path to the dataset configuration file (dataset.yaml). Defaults to 'data/coco128.yaml'.
        device (str): Device to be used for inference, e.g., '0' or '0,1,2,3' for GPU or 'cpu' for CPU. Defaults to ''.
        half (bool): Use FP16 half-precision for inference. Defaults to False.
        test (bool): Test exports only without running benchmarks. Defaults to False.
        pt_only (bool): Run benchmarks only for PyTorch format. Defaults to False.
        hard_fail (bool): Raise an error if any benchmark test fails. Defaults to False.

    Returns:
        (pandas.DataFrame): Per-format results with columns ["Format", "Size (MB)", "mAP50-95", "Inference time (ms)"].

    Notes:
        Iterates over the supported export formats, exports each, then validates it with the detection validator and
        logs the results. When `hard_fail` is a numeric string, mAP below that floor raises an assertion.
    """
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, CPU, GPU)
        try:
            assert i not in (6, 7, 8, 9, 10), "TensorFlow export not supported by this repository"
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
    notebook_init()  # print system info
    c = ["Format", "Size (MB)", "mAP50-95", "Inference time (ms)"]
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f"\nBenchmarks complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py))
    if hard_fail and isinstance(hard_fail, str):
        metrics = py["mAP50-95"].array  # values to compare to floor
        floor = float(hard_fail)  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv3-tiny
        assert all(x > floor for x in metrics if pd.notna(x)), f"HARD FAIL: mAP50-95 < floor {floor}"
    return py


def test(
    weights=ROOT / "yolov3-tiny.pt",  # weights path
    imgsz=640,  # inference size (pixels)
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    test=False,  # test exports only
    pt_only=False,  # test PyTorch only
    hard_fail=False,  # throw error on benchmark failure
):
    """Run YOLOv3 export tests for various formats and log the results, including export success status.

    Args:
        weights (str | Path): Path to the weights file. Defaults to ROOT / "yolov3-tiny.pt".
        imgsz (int): Inference size in pixels. Defaults to 640.
        batch_size (int): Number of images per batch. Defaults to 1.
        data (str | Path): Path to the dataset yaml file. Defaults to ROOT / "data/coco128.yaml".
        device (str): Device for inference. Accepts cuda device (e.g., "0" or "0,1,2,3") or "cpu". Defaults to "".
        half (bool): Use FP16 half-precision inference. Defaults to False.
        test (bool): Run export tests only, no inference. Defaults to False.
        pt_only (bool): Run tests on PyTorch format only. Defaults to False.
        hard_fail (bool): Raise an error on benchmark failure. Defaults to False.

    Returns:
        (pandas.DataFrame): Table with columns ["Format", "Export"] indicating whether each format exported
            successfully.
    """
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, CPU, GPU)
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
    notebook_init()  # print system info
    py = pd.DataFrame(y, columns=["Format", "Export"])
    LOGGER.info(f"\nExports complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py))
    return py


def parse_opt():
    """Parses command line arguments for YOLOv3 inference and export configurations.

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
        (argparse.Namespace): The parsed arguments as a namespace object.

    Examples:
        Run benchmarks on the YOLOv3-tiny model with a custom image size:
        ```bash
        python benchmarks.py --weights yolov3-tiny.pt --imgsz 512 --device 0
        ```

    Notes:
        `--hard-fail` may be a boolean or a string. As a string it is an expression for the minimum acceptable metric,
        such as '0.29' for mAP (mean Average Precision).
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
    """Executes the export and benchmarking pipeline for YOLOv3 models, testing multiple export formats and validating
    performance metrics.

    Args:
        opt (argparse.Namespace): Parsed command line arguments, including options for weights, image size, batch size,
            dataset path, device, half-precision inference, test mode, PyTorch-only testing, and hard fail conditions.

    Returns:
        (pandas.DataFrame): Benchmark results with columns ["Format", "Size (MB)", "mAP50-95", "Inference time (ms)"],
            or export-status results when `opt.test` is True.

    Notes:
        When `opt.test` is True, only the export process is tested; otherwise full benchmarks (export plus validation)
        are run.
    """
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
