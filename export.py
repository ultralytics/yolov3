# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLOv3 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

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
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
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

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_version,
    check_yaml,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
    yaml_save,
)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == "Darwin"  # macOS environment


class iOSModel(torch.nn.Module):
    """Exports a PyTorch model to an iOS-compatible format with normalized input dimensions and class configurations."""

    def __init__(self, model, im):
        """
        Initializes an iOSModel with normalized input dimensions and number of classes from a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model from which to initialize the iOS model. This should include attributes
                like `nc` (number of classes) which will be used to configure the iOS model.
            im (torch.Tensor): A Tensor representing a sample input image. The shape of this tensor should be
                (batch_size, channels, height, width). This is used to extract dimensions for input normalization.

        Returns:
            None

        Notes:
            - This class is specifically designed for use in exporting a PyTorch model for deployment on iOS platforms, optimizing
              input dimensions and class configurations to suit mobile requirements.
            - Normalization factor is derived from the input image dimensions, which impacts the model's performance during
              inference on iOS devices.
            - Ensure the sample input image `im` provided has correct dimensions and shape for accurate model configuration.
        """
        super().__init__()
        b, c, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = model.nc  # number of classes
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)
            # np = model(im)[0].shape[1]  # number of points
            # self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h]).expand(np, 4)  # explicit (faster, larger)

    def forward(self, x):
        """
        Performs a forward pass, returning scaled confidences and normalized coordinates given an input tensor.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images, with dimensions [batch_size, channels,
                height, width].

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three elements:
                - xywh (torch.Tensor): Tensor of shape [batch_size, num_detections, 4] containing normalized x, y, width,
                  and height coordinates.
                - conf (torch.Tensor): Tensor of shape [batch_size, num_detections, 1] containing confidence scores for
                  each detection.
                - cls (torch.Tensor): Tensor of shape [batch_size, num_detections, num_classes] containing class
                  probabilities.

        Notes:
            The dimensions of `x` should match the input dimensions used during the model's initialization to ensure
            proper scaling and normalization.

        Examples:
            ```python
            model = iOSModel(trained_model, input_image_tensor)
            detection_results = model.forward(input_tensor)
            xywh, conf, cls = detection_results
            ```

        Further reading on exporting models to different formats:
        https://github.com/ultralytics/ultralytics

        See Also:
            `export.py` for exporting a YOLOv3 PyTorch model to various formats.
            https://github.com/zldrobit for TensorFlow export scripts.
        """
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


def export_formats():
    """
    Lists supported YOLOv3 model export formats including file suffixes and CPU/GPU compatibility.

    Returns:
        list: A list of lists where each sublist contains information about a specific export format. Each sublist includes
            the following elements:
            - str: The name of the format.
            - str: The command-line argument for including this format.
            - str: The file suffix used for this format.
            - bool: Indicates if the format is compatible with CPU.
            - bool: Indicates if the format is compatible with GPU.

    Examples:
        ```python
        formats = export_formats()
        for format in formats:
            print(f"Format: {format[0]}, Suffix: {format[2]}, CPU Compatible: {format[3]}, GPU Compatible: {format[4]}")
        ```
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    """
    Profiles and logs the export process of YOLOv3 models, capturing success or failure details.

    Args:
        inner_func (Callable): The function that performs the actual export process and returns the model file path
            and the exported model.

    Returns:
        Callable: A wrapped function that profiles and logs the export process, handling successes and failures.

    Examples:
        ```python
        @try_export
        def export_onnx(py_model_path: str, output_path: str):
            # Export logic here
            return output_path, model
        ```

    Notes:
        Applying this decorator to an export function will log the export results, including export success or failure,
        along with associated time and file size details.
    """
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Profiles and logs the export process of YOLOv3 models, capturing success or failure details."""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func


@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr("TorchScript:")):
    """
    Export a YOLOv3 model to TorchScript format, with optional optimization for mobile deployment.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): A tensor representing the input image for the model, typically with shape (N, 3, H, W).
        file (pathlib.Path): The file path where the TorchScript model will be saved.
        optimize (bool): A boolean flag indicating whether to optimize the model for mobile devices.
        prefix (str): A prefix for logging messages. Defaults to `colorstr("TorchScript:")`.

    Returns:
        (pathlib.Path | None, torch.nn.Module | None): Tuple containing the path to the saved TorchScript model and the
        model itself. Returns `(None, None)` if the export fails.

    Raises:
        Exception: If there is an error during export, it logs the error and returns `(None, None)`.

    Notes:
        The function uses `torch.jit.trace` to trace the model with the input image tensor (`im`). Required metadata such as
        input shape, stride, and class names are stored in an extra file included in the TorchScript model.

    Examples:
        ```python
        from pathlib import Path
        import torch

        model = ...  # Assume model is loaded or created
        im = torch.randn(1, 3, 640, 640)  # A sample input tensor
        file = Path("model.torchscript")
        optimize = True

        export_torchscript(model, im, file, optimize)
        ```

    For more information, visit: https://ultralytics.com/.
    """
    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = file.with_suffix(".torchscript")

    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {"config.txt": json.dumps(d)}  # torch._C.ExtraFilesMap()
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
    """
    Export a YOLOv3 model to ONNX format with dynamic shape and simplification options.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): A sample input tensor for tracing the model.
        file (pathlib.Path): The file path where the ONNX model will be saved.
        opset (int): The ONNX opset version to use for the export.
        dynamic (bool): If `True`, enables dynamic shape support.
        simplify (bool): If `True`, simplifies the ONNX model using onnx-simplifier.
        prefix (str): A prefix for logging messages.

    Returns:
        tuple[pathlib.Path, None]: The path to the saved ONNX model, None as the second tuple element (kept for consistency).

    Example:
        ```python
        from pathlib import Path
        import torch

        model = ...  # Assume model is loaded or created
        im = torch.randn(1, 3, 640, 640)  # A sample input tensor
        file = Path("model.onnx")
        opset = 12
        dynamic = True
        simplify = True

        export_onnx(model, im, file, opset, dynamic, simplify)
        ```

    Notes:
        Ensure `onnx`, `onnx-simplifier`, and suitable runtime packages are installed.
        This function uses `torch.onnx.export` to create the ONNX model, followed by optional simplification using
        `onnx-simplifier`. If `dynamic` is enabled, dynamic axes mappings are added to support variable input shapes.
        Relevant YOLO model metadata like `stride` and `names` are included as part of the ONNX model's metadata.

    For more details on exporting and running inferences, visit:
    - https://github.com/ultralytics/ultralytics
    - https://github.com/zldrobit for TensorFlow export scripts.
    """
    check_requirements("onnx>=1.12.0")
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    if dynamic:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
            dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {"stride": int(max(model.stride)), "names": model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1"))
            import onnxsim

            LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")
    return f, model_onnx


@try_export
def export_openvino(file, metadata, half, int8, data, prefix=colorstr("OpenVINO:")):
    """
    Export a YOLOv3 model to OpenVINO format with optional INT8 quantization and inference metadata.

    Args:
        file (Path): Path to the output file.
        metadata (dict): Inference metadata to include in the exported model.
        half (bool): Indicates if FP16 precision should be used.
        int8 (bool): Indicates if INT8 quantization should be applied.
        data (str): Path to the dataset file (.yaml) for post-training quantization.

    Returns:
        tuple[Path | None, openvino.runtime.Model | None]: Tuple containing the path to the exported model and the OpenVINO
            model object, or None if the export failed.

    Notes:
        - Requires the `openvino-dev>=2023.0` and optional `nncf>=2.4.0` package for INT8 quantization.
        - Refer to OpenVINO documentation for further details: https://docs.openvino.ai/latest/index.html.

    Examples:
        ```python
        model_file = Path('/path/to/model.onnx')
        metadata = {'names': ['class1', 'class2'], 'stride': 32}
        export_openvino(model_file, metadata, half=True, int8=False, data='/path/to/dataset.yaml')
        ```
    """
    check_requirements("openvino-dev>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo  # noqa

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
    f = str(file).replace(file.suffix, f"_openvino_model{os.sep}")
    f_onnx = file.with_suffix(".onnx")
    f_ov = str(Path(f) / file.with_suffix(".xml").name)
    if int8:
        check_requirements("nncf>=2.4.0")  # requires at least version 2.4.0 to use the post-training quantization
        import nncf
        import numpy as np
        from openvino.runtime import Core

        from utils.dataloaders import create_dataloader

        core = Core()
        onnx_model = core.read_model(f_onnx)  # export

        def prepare_input_tensor(image: np.ndarray):
            """Prepares the input tensor by normalizing pixel values and converting the datatype to float32."""
            input_tensor = image.astype(np.float32)  # uint8 to fp16/32
            input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

            if input_tensor.ndim == 3:
                input_tensor = np.expand_dims(input_tensor, 0)
            return input_tensor

        def gen_dataloader(yaml_path, task="train", imgsz=640, workers=4):
            """Generates a PyTorch dataloader for the specified task using dataset configurations from a YAML file."""
            data_yaml = check_yaml(yaml_path)
            data = check_dataset(data_yaml)
            dataloader = create_dataloader(
                data[task], imgsz=imgsz, batch_size=1, stride=32, pad=0.5, single_cls=False, rect=False, workers=workers
            )[0]
            return dataloader

        # noqa: F811

        def transform_fn(data_item):
            """
            Quantization transform function.

            Extracts and preprocess input data from dataloader item for quantization.

            Parameters:
               data_item: Tuple with data item produced by DataLoader during iteration
            Returns:
                input_tensor: Input data for quantization
            """
            img = data_item[0].numpy()
            input_tensor = prepare_input_tensor(img)
            return input_tensor

        ds = gen_dataloader(data)
        quantization_dataset = nncf.Dataset(ds, transform_fn)
        ov_model = nncf.quantize(onnx_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)
    else:
        ov_model = mo.convert_model(f_onnx, model_name=file.stem, framework="onnx", compress_to_fp16=half)  # export

    ov.serialize(ov_model, f_ov)  # save
    yaml_save(Path(f) / file.with_suffix(".yaml").name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr("PaddlePaddle:")):
    """
    Export a YOLOv3 model to PaddlePaddle format using X2Paddle, saving to a specified directory and including model
    metadata.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): A sample input tensor used for tracing the model.
        file (pathlib.Path): Destination file path for the exported model, with `.pt` suffix.
        metadata (dict): Additional metadata to be saved in YAML format alongside the exported model.
        prefix (str, optional): Log message prefix. Defaults to a colored "PaddlePaddle:" string.

    Returns:
        tuple: A tuple containing the directory path (str) where the PaddlePaddle model is saved, and `None`.

    Requirements:
        - paddlepaddle: Install via `pip install paddlepaddle`.
        - x2paddle: Install via `pip install x2paddle`.

    Notes:
        The function first checks for required packages `paddlepaddle` and `x2paddle`. It then uses X2Paddle to trace
        the model and export it to a PaddlePaddle format, saving the resulting files in the specified directory with
        included metadata in a YAML file.

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.yolo import DetectionModel

        model = DetectionModel()  # Example model initialization
        im = torch.rand(1, 3, 640, 640)  # Example input tensor
        file = Path("path/to/save/model.pt")
        metadata = {"nc": 80, "names": ["class1", "class2", ...]}  # Example metadata

        export_paddle(model, im, file, metadata)
        ```
    """
    check_requirements(("paddlepaddle", "x2paddle"))
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
    f = str(file).replace(".pt", f"_paddle_model{os.sep}")

    pytorch2paddle(module=model, save_dir=f, jit_type="trace", input_examples=[im])  # export
    yaml_save(Path(f) / file.with_suffix(".yaml").name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_coreml(model, im, file, int8, half, nms, prefix=colorstr("CoreML:")):
    """
    Export a YOLOv3 model to CoreML format with optional quantization and Non-Maximum Suppression (NMS).

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Input tensor used for tracing the model. Shape should be (batch_size, channels, height, width).
        file (pathlib.Path): Destination file path where the CoreML model will be saved.
        int8 (bool): Whether to use INT8 quantization. If True, quantizes the model to 8-bit integers.
        half (bool): Whether to use FP16 quantization. If True, converts the model to 16-bit floating point numbers.
        nms (bool): Whether to include Non-Maximum Suppression in the CoreML model.
        prefix (str): Prefix string for logging purposes. Default is colorstr("CoreML:").

    Returns:
        str: Path to the saved CoreML model (.mlmodel).

    Raises:
        Exception: If there is an error during export, logs the error and stops the process.

    Notes:
        - This function requires `coremltools` to be installed.
        - If `nms` is enabled, the model is wrapped with `iOSModel` to include NMS.
        - Quantization only works on macOS.

    Example:
        ```python
        from ultralytics.utils import export_coreml
        from pathlib import Path
        import torch

        model = ...  # Assume model is loaded or created
        im = torch.randn(1, 3, 640, 640)  # A sample input tensor
        file = Path("model.mlmodel")
        export_coreml(model, im, file, int8=False, half=True, nms=True)
        ```
    """
    check_requirements("coremltools")
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    f = file.with_suffix(".mlmodel")

    if nms:
        model = iOSModel(model, im)
    ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
    ct_model = ct.convert(ts, inputs=[ct.ImageType("image", shape=im.shape, scale=1 / 255, bias=[0, 0, 0])])
    bits, mode = (8, "kmeans_lut") if int8 else (16, "linear") if half else (32, None)
    if bits < 32:
        if MACOS:  # quantization only supported on macOS
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        else:
            print(f"{prefix} quantization only supported on macOS, skipping...")
    ct_model.save(f)
    return f, ct_model


@try_export
def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr("TensorRT:")):
    """
    Export a YOLOv3 model to TensorRT engine format, optimizing it for GPU inference.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Sample input tensor used for tracing the model.
        file (Path): File path where the exported TensorRT engine will be saved.
        half (bool): Whether to use FP16 precision. Requires a supported GPU.
        dynamic (bool): Whether to use dynamic input shapes.
        simplify (bool): Whether to simplify the model during the ONNX export.
        workspace (int): The maximum workspace size in GB. Default is 4.
        verbose (bool): Whether to print detailed export logs.
        prefix (str): Prefix string for log messages. Default is "TensorRT:".

    Returns:
        tuple[Path, None]: The output file path (Path) and None.

    Raises:
        AssertionError: If the model is running on CPU instead of GPU.
        RuntimeError: If the ONNX file failed to load.

    Notes:
        Requires TensorRT installation to execute. Nvidia TensorRT: https://developer.nvidia.com/tensorrt

    Example:
        ```python
        from pathlib import Path
        import torch

        # Initialize model and dummy input
        model = YOLOv3(...)  # or another correct initialization
        im = torch.randn(1, 3, 640, 640)

        # Export the model
        export_engine(model, im, Path("yolov3.engine"), half=True, dynamic=True, simplify=True)
        ```
    """
    assert im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == "Linux":
            check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
        import tensorrt as trt

    if trt.__version__[0] == "7":  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
        model.model[-1].anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, "8.0.0", hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
    onnx = file.with_suffix(".onnx")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    assert onnx.exists(), f"failed to export ONNX file: {onnx}"
    f = file.with_suffix(".engine")  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f"failed to load ONNX file: {onnx}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, "wb") as t:
        t.write(engine.serialize())
    return f, None


@try_export
def export_saved_model(
    model,
    im,
    file,
    dynamic,
    tf_nms=False,
    agnostic_nms=False,
    topk_per_class=100,
    topk_all=100,
    iou_thres=0.45,
    conf_thres=0.25,
    keras=False,
    prefix=colorstr("TensorFlow SavedModel:"),
):
    """
    Exports a YOLOv3 model to TensorFlow SavedModel format, including optional settings for Non-Max Suppression (NMS).

    Args:
        model (torch.nn.Module): The YOLOv3 PyTorch model to be exported.
        im (torch.Tensor): Tensor of sample input data used for tracing the model.
        file (pathlib.Path): File path where the exported TensorFlow SavedModel will be saved.
        dynamic (bool): If `True`, supports dynamic input shapes.
        tf_nms (bool, optional): If `True`, includes TensorFlow NMS in the exported model. Defaults to `False`.
        agnostic_nms (bool, optional): If `True`, uses class-agnostic NMS. Defaults to `False`.
        topk_per_class (int, optional): Number of top-K predictions to keep per class after NMS. Defaults to `100`.
        topk_all (int, optional): Number of top-K predictions to keep overall after NMS. Defaults to `100`.
        iou_thres (float, optional): Intersection over Union (IoU) threshold for NMS. Defaults to `0.45`.
        conf_thres (float, optional): Confidence threshold for NMS. Defaults to `0.25`.
        keras (bool, optional): If `True`, saves the model in Keras format. Defaults to `False`.
        prefix (str, optional): Prefix for logging messages. Defaults to `colorstr("TensorFlow SavedModel:")`.

    Returns:
        (str, None): Path to the saved TensorFlow model as a string and `None` (kept for interface consistency).

    Raises:
        ImportError: If the required TensorFlow libraries are not installed.

    Examples:
        ```python
        from pathlib import Path
        from models.common import DetectMultiBackend
        import torch

        model = DetectMultiBackend(weights='yolov5s.pt')
        im = torch.zeros(1, 3, 640, 640)  # Sample input tensor
        file = Path("output/saved_model")

        export_saved_model(model, im, file, dynamic=True)
        ```

    Notes:
        - Ensure that required TensorFlow libraries are installed (e.g., `pip install tensorflow`).
        - For more information, visit https://github.com/ultralytics/yolov5.
    """
    # YOLOv3 TensorFlow SavedModel export
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}")
        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from models.tf import TFModel

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    f = str(file).replace(".pt", "_saved_model")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format="tf")
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(
            tfm,
            f,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
            if check_version(tf.__version__, "2.6")
            else tf.saved_model.SaveOptions(),
        )
    return f, keras_model


@try_export
def export_pb(keras_model, file, prefix=colorstr("TensorFlow GraphDef:")):
    """
    Export a Keras model to TensorFlow GraphDef (*.pb) format, which is compatible with YOLOv3.

    Args:
        keras_model (tf.keras.Model): The trained Keras model to be exported.
        file (pathlib.Path): The target file path for saving the exported model.
        prefix (str, optional): Prefix string for logging. Defaults to colorstr("TensorFlow GraphDef:").

    Returns:
        tuple[pathlib.Path, None]: The file path where the model is saved and None.

    Example:
        ```python
        from tensorflow.keras.models import load_model
        from pathlib import Path
        export_pb(load_model('model.h5'), Path('model.pb'))
        ```

    See Also:
        For more details on TensorFlow GraphDef, visit
        https://github.com/leimao/Frozen_Graph_TensorFlow.

    Notes:
        Ensure TensorFlow is properly installed in your environment as it is required for this function to execute.
        TensorFlow's version should be compatible with the version used to train your model to avoid any compatibility
        issues.
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    f = file.with_suffix(".pb")

    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
    return f, None


@try_export
def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")):
    """
    Export a YOLOv3 PyTorch model to TensorFlow Lite (TFLite) format.

    Args:
        keras_model (tf.keras.Model): The Keras model obtained after converting the PyTorch model.
        im (torch.Tensor): Sample input tensor to determine model input size.
        file (pathlib.Path): Desired file path for saving the exported TFLite model.
        int8 (bool): Flag to enable INT8 quantization for the TFLite model.
        data (str): Path to dataset YAML file for representative data generation used in quantization.
        nms (bool): Flag to include Non-Maximum Suppression (NMS) in the exported TFLite model.
        agnostic_nms (bool): Flag to apply class-agnostic NMS during inference.
        prefix (str, optional): Prefix for logging messages. Defaults to colorstr("TensorFlow Lite:").

    Returns:
        (str | None): File path of the saved TensorFlow Lite model file or None if export fails.

    Notes:
        - Ensure TensorFlow is installed to perform the export.
        - INT8 quantization requires a representative dataset to provide accurate calibration for the model.
        - Including Non-Max Suppression (NMS) modifies the exported model to handle post-processing.

    Example:
        ```python
        import torch
        from pathlib import Path
        from models.experimental import attempt_load

        # Load and prepare model
        model = attempt_load('yolov5s.pt', map_location='cpu')
        im = torch.zeros(1, 3, 640, 640)  # Dummy input tensor

        # Export model
        export_tflite(model, im, Path('yolov5s'), int8=False, data=None, nms=True, agnostic_nms=False)
        ```

    For more details, refer to:
        TensorFlow Lite Developer Guide: https://www.tensorflow.org/lite/guide
        Model Conversion Reference: https://github.com/leimao/Frozen_Graph_TensorFlow
    """
    import tensorflow as tf

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    f = str(file).replace(".pt", "-fp16.tflite")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from models.tf import representative_dataset_gen

        dataset = LoadImages(check_dataset(check_yaml(data))["train"], img_size=imgsz, auto=False)
        converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        f = str(file).replace(".pt", "-int8.tflite")
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()
    open(f, "wb").write(tflite_model)
    return f, None


@try_export
def export_edgetpu(file, prefix=colorstr("Edge TPU:")):
    """
    Export a YOLOv5 model to TensorFlow Edge TPU format with INT8 quantization.

    Args:
        file (Path): The file path for the PyTorch model to be exported, with a `.pt` suffix.
        prefix (str): A prefix to be used for logging output. Defaults to "Edge TPU:"

    Returns:
        Tuple[Path | None, None]: A tuple containing the file path of the exported model with the `-int8_edgetpu.tflite`
         suffix and `None`, if successful. If unsuccessful, returns `(None, None)`.

    Raises:
        AssertionError: If the export is not executed on a Linux system.
        subprocess.CalledProcessError: If there are issues with subprocess execution, particularly around Edge TPU compiler
         installation or model conversion.

    Notes:
        This function is designed to work exclusively on Linux systems and requires the Edge TPU compiler to be installed.
        If the compiler is not found, the function attempts to install it.

    Example:
        ```python
        from pathlib import Path
        from ultralytics import export_edgetpu

        model_file = Path('yolov5s.pt')
        exported_model, _ = export_edgetpu(model_file)
        print(f"Model exported to {exported_model}")
        ```

    For additional details, visit the Edge TPU compiler documentation:
    https://coral.ai/docs/edgetpu/compiler/
    """
    cmd = "edgetpu_compiler --version"
    help_url = "https://coral.ai/docs/edgetpu/compiler/"
    assert platform.system() == "Linux", f"export only supported on Linux. See {help_url}"
    if subprocess.run(f"{cmd} > /dev/null 2>&1", shell=True).returncode != 0:
        LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
        sudo = subprocess.run("sudo --version >/dev/null", shell=True).returncode == 0  # sudo installed on system
        for c in (
            "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
            'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
            "sudo apt-get update",
            "sudo apt-get install edgetpu-compiler",
        ):
            subprocess.run(c if sudo else c.replace("sudo ", ""), shell=True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

    LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
    f = str(file).replace(".pt", "-int8_edgetpu.tflite")  # Edge TPU model
    f_tfl = str(file).replace(".pt", "-int8.tflite")  # TFLite model

    subprocess.run(
        [
            "edgetpu_compiler",
            "-s",
            "-d",
            "-k",
            "10",
            "--out_dir",
            str(file.parent),
            f_tfl,
        ],
        check=True,
    )
    return f, None


@try_export
def export_tfjs(file, int8, prefix=colorstr("TensorFlow.js:")):
    """
    Export a YOLOv3 model to TensorFlow.js format, with an optional quantization to uint8.

    Args:
        file (Path): The path to the model file to be exported.
        int8 (bool): Boolean flag to determine if the model should be quantized to uint8.
        prefix (str): String prefix for logging, by default "TensorFlow.js".

    Returns:
        (tuple[str, None]): The directory path where the TensorFlow.js model files are saved and `None` placeholder to match
            the expected return type from 'try_export' decorator.

    Raises:
        ImportError: If the required 'tensorflowjs' package is not installed.

    Example:
        ```python
        from pathlib import Path
        export_tfjs(file=Path("yolov5s.pt"), int8=False)
        ```

    Note:
        Ensure that you have TensorFlow.js installed in your environment. Install the package via:
        ```bash
        pip install tensorflowjs
        ```

        For more details on using the converted model:
        Refer to the official TensorFlow.js documentation: https://www.tensorflow.org/js.

    Usage:
        The converted model can be used directly in JavaScript environments using the TensorFlow.js library.

        For usage in web applications:
            - Clone the example repository:
                ```bash
                cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
                ```
            - Install dependencies:
                ```bash
                npm install
                ```
            - Create a symbolic link to the exported web model:
                ```bash
                ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
                ```
            - Start the example application:
                ```bash
                npm start
                ```
    """
    check_requirements("tensorflowjs")
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
    f = str(file).replace(".pt", "_web_model")  # js dir
    f_pb = file.with_suffix(".pb")  # *.pb path
    f_json = f"{f}/model.json"  # *.json path

    args = [
        "tensorflowjs_converter",
        "--input_format=tf_frozen_model",
        "--quantize_uint8" if int8 else "",
        "--output_node_names=Identity,Identity_1,Identity_2,Identity_3",
        str(f_pb),
        f,
    ]
    subprocess.run([arg for arg in args if arg], check=True)

    json = Path(f_json).read_text()
    with open(f_json, "w") as j:  # sort JSON Identity_* in ascending order
        subst = re.sub(
            r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}}}',
            r'{"outputs": {"Identity": {"name": "Identity"}, '
            r'"Identity_1": {"name": "Identity_1"}, '
            r'"Identity_2": {"name": "Identity_2"}, '
            r'"Identity_3": {"name": "Identity_3"}}}',
            json,
        )
        j.write(subst)
    return f, None


def add_tflite_metadata(file, metadata, num_outputs):
    """
    Adds metadata to a TensorFlow Lite model to enhance its usability with `tflite_support`.

    Args:
        file (str): Path to the TensorFlow Lite model file.
        metadata (dict): Dictionary of metadata to add, including descriptions of inputs, outputs, and other relevant info.
        num_outputs (int): Number of output tensors in the model.

    Returns:
        None

    Example:
        ```python
        metadata = {
            "input": {"description": "Input image tensor"},
            "output": [{"name": "scores", "description": "Detection scores"}],
        }
        add_tflite_metadata("/path/to/model.tflite", metadata, num_outputs=1)
        ```

    Note:
        Requires the `tflite_support` library for adding metadata to the TensorFlow Lite model.
        Installation: `pip install tflite-support`

        ```python
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

        tmp_file = Path("/tmp/meta.txt")
        with open(tmp_file, "w") as meta_f:
            meta_f.write(str(metadata))

        model_meta = _metadata_fb.ModelMetadataT()
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
        subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        ```

        This function is a helper to add metadata to a TFLite model, making it easier to interpret and process for tasks like
        object detection or classification. It leverages `tflite_support` to load and attach the metadata directly to the
        model file.
    """
    with contextlib.suppress(ImportError):
        # check_requirements('tflite_support')
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

        tmp_file = Path("/tmp/meta.txt")
        with open(tmp_file, "w") as meta_f:
            meta_f.write(str(metadata))

        model_meta = _metadata_fb.ModelMetadataT()
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
        subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()


def pipeline_coreml(model, im, file, names, y, prefix=colorstr("CoreML Pipeline:")):
    """
    Processes and exports a YOLOv3 model into the CoreML model format, applying metadata and non-maximum suppression
    (NMS).

    Args:
        model (coremltools.models.MLModel): The pre-trained YOLOv3 CoreML model to be used for the pipeline.
        im (torch.Tensor): Input image tensor in BCHW (Batch, Channel, Height, Width) format with a shape (B, 3, H, W).
        file (pathlib.Path): Destination file path where the CoreML model will be saved.
        names (dict): A dictionary that maps class indices to class names.
        y (torch.Tensor): Output detection tensor from the YOLO model, containing predictions.
        prefix (str): Prefix for logging messages, default is "CoreML Pipeline:".

    Returns:
        pathlib.Path | None: The path to the saved CoreML model if successful, otherwise None.

    Example:
        ```python
        from pathlib import Path
        import torch
        from coremltools.models import MLModel

        # Load example CoreML model
        model = MLModel('path/to/pretrained/model.mlmodel')

        # Create example input tensor: B, C, H, W format
        im = torch.randn(1, 3, 640, 640)

        # Define where the CoreML model will be saved
        file = Path('path/to/save/model.mlmodel')

        # Define example class names
        names = {0: 'class0', 1: 'class1'}

        # Dummy YOLO model output prediction having similar dimensions to y
        y = torch.randn(1, 25200, 85)

        # Execute CoreML pipeline
        pipeline_coreml(model, im, file, names, y)
        ```

    Notes:
        - The function adds NMS to the CoreML model, supporting dynamic thresholds for IoU and confidence.
        - Metadata fields are updated to include class names, thresholds, and additional information.
        - The pipeline exports the final enhanced model into the specified file path in CoreML (`.mlmodel`) format.
        - Ensure that `coremltools` is installed and properly configured in your environment.
        - This function is designed to work primarily on macOS systems as CoreML is macOS-specific.

    References:
    - `coremltools`: https://github.com/apple/coremltools
    - YOLOv3: https://github.com/ultralytics/yolov5
    """
    import coremltools as ct
    from PIL import Image

    print(f"{prefix} starting pipeline with coremltools {ct.__version__}...")
    batch_size, ch, h, w = list(im.shape)  # BCHW
    t = time.time()

    # YOLOv3 Output shapes
    spec = model.get_spec()
    out0, out1 = iter(spec.description.output)
    if platform.system() == "Darwin":
        img = Image.new("RGB", (w, h))  # img(192 width, 320 height)
        # img = torch.zeros((*opt.img_size, 3)).numpy()  # img size(320,192,3) iDetection
        out = model.predict({"image": img})
        out0_shape, out1_shape = out[out0.name].shape, out[out1.name].shape
    else:  # linux and windows can not run model.predict(), get sizes from pytorch output y
        s = tuple(y[0].shape)
        out0_shape, out1_shape = (s[1], s[2] - 5), (s[1], 4)  # (3780, 80), (3780, 4)

    # Checks
    nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
    na, nc = out0_shape
    # na, nc = out0.type.multiArrayType.shape  # number anchors, classes
    assert len(names) == nc, f"{len(names)} names found for nc={nc}"  # check

    # Define output shapes (missing)
    out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
    out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
    # spec.neuralNetwork.preprocessing[0].featureName = '0'

    # Flexible input shapes
    # from coremltools.models.neural_network import flexible_shape_utils
    # s = [] # shapes
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (height, width)
    # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
    # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # shape ranges
    # r.add_height_range((192, 640))
    # r.add_width_range((192, 640))
    # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

    # Print
    print(spec.description)

    # Model from spec
    model = ct.models.MLModel(spec)

    # 3. Create NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    nms_spec.description.output[0].name = "confidence"
    nms_spec.description.output[1].name = "coordinates"

    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out0.name  # 1x507x80
    nms.coordinatesInputFeatureName = out1.name  # 1x507x4
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.pickTop.perClass = True
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec)

    # 4. Pipeline models together
    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, ny, nx)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=["confidence", "coordinates"],
    )
    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # Correct datatypes
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # Update metadata
    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.versionString = "https://github.com/ultralytics/yolov5"
    pipeline.spec.description.metadata.shortDescription = "https://github.com/ultralytics/yolov5"
    pipeline.spec.description.metadata.author = "glenn.jocher@ultralytics.com"
    pipeline.spec.description.metadata.license = "https://github.com/ultralytics/yolov5/blob/master/LICENSE"
    pipeline.spec.description.metadata.userDefined.update(
        {
            "classes": ",".join(names.values()),
            "iou_threshold": str(nms.iouThreshold),
            "confidence_threshold": str(nms.confidenceThreshold),
        }
    )

    # Save the model
    f = file.with_suffix(".mlmodel")  # filename
    model = ct.models.MLModel(pipeline.spec)
    model.input_description["image"] = "Input image"
    model.input_description["iouThreshold"] = f"(optional) IOU Threshold override (default: {nms.iouThreshold})"
    model.input_description["confidenceThreshold"] = (
        f"(optional) Confidence Threshold override (default: {nms.confidenceThreshold})"
    )
    model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
    model.save(f)  # pipelined
    print(f"{prefix} pipeline success ({time.time() - t:.2f}s), saved as {f} ({file_size(f):.1f} MB)")


@smart_inference_mode()
def run(
    data=ROOT / "data/coco128.yaml",  # 'dataset.yaml path'
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv3 Detect() inplace=True
    keras=False,  # use Keras
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/TF INT8 quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # TF: add NMS to model
    agnostic_nms=False,  # TF: add agnostic NMS to model
    topk_per_class=100,  # TF.js NMS: topk per class to keep
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
):
    """
    Export a PyTorch model to various formats like ONNX, CoreML, and TensorRT.

    Args:
        data (str | Path): Path to dataset configuration file.
        weights (str | Path): Path to model weights file in PyTorch format.
        imgsz (tuple[int, int]): Tuple specifying image height and width for input dimensions.
        batch_size (int): Batch size for model inference.
        device (str): Device to use for inference (e.g., '0', '0,1,2,3', 'cpu').
        include (tuple[str]): Formats to include for model export (e.g., 'torchscript', 'onnx', etc.).
        half (bool): Whether to export model with FP16 precision.
        inplace (bool): Set YOLOv3 Detect module inplace option to True.
        keras (bool): Save Keras model when exporting TensorFlow SavedModel format.
        optimize (bool): Optimize the TorchScript model for mobile inference.
        int8 (bool): Apply INT8 quantization for CoreML/TF models.
        dynamic (bool): Enable dynamic axes for ONNX/TF/TensorRT models.
        simplify (bool): Simplify the ONNX model after export.
        opset (int): ONNX opset version.
        verbose (bool): Enable verbose logging for TensorRT engine export.
        workspace (int): Workspace size in GB for TensorRT engine.
        nms (bool): Enable Non-Maximum Suppression (NMS) in TensorFlow models.
        agnostic_nms (bool): Enable class-agnostic NMS in TensorFlow models.
        topk_per_class (int): Top-K per class to keep in TensorFlow JSON model.
        topk_all (int): Top-K for all classes to keep in TensorFlow JSON model.
        iou_thres (float): IOU threshold for TensorFlow JSON model.
        conf_thres (float): Confidence threshold for TensorFlow JSON model.

    Returns:
        None

    Notes:
        - Requires various packages installed for different export formats, e.g., `onnx`, `coremltools`, etc.
        - Some formats have additional dependencies (e.g., TensorFlow, TensorRT, etc.)

    Examples:
        ```python
        run(
            data='data/coco128.yaml',
            weights='yolov5s.pt',
            imgsz=(640, 640),
            batch_size=1,
            device='cpu',
            include=('torchscript', 'onnx'),
            half=False,
            dynamic=True,
            opset=12
        )
        ```
    """
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != "cpu" or coreml, "--half only compatible with GPU export, i.e. use --device 0"
        assert not dynamic, "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == "cpu", "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [""] * len(fmts)  # exported filenames
    warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
    if jit:  # TorchScript
        f[0], _ = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required before ONNX
        f[1], _ = export_engine(model, im, file, half, dynamic, simplify, workspace, verbose)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)
    if xml:  # OpenVINO
        f[3], _ = export_openvino(file, metadata, half, int8, data)
    if coreml:  # CoreML
        f[4], ct_model = export_coreml(model, im, file, int8, half, nms)
        if nms:
            pipeline_coreml(ct_model, im, file, model.names, y)
    if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
        assert not tflite or not tfjs, "TFLite and TF.js models must be exported separately, please pass only one type."
        assert not isinstance(model, ClassificationModel), "ClassificationModel export to TF formats not yet supported."
        f[5], s_model = export_saved_model(
            model.cpu(),
            im,
            file,
            dynamic,
            tf_nms=nms or agnostic_nms or tfjs,
            agnostic_nms=agnostic_nms or tfjs,
            topk_per_class=topk_per_class,
            topk_all=topk_all,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            keras=keras,
        )
        if pb or tfjs:  # pb prerequisite to tfjs
            f[6], _ = export_pb(s_model, file)
        if tflite or edgetpu:
            f[7], _ = export_tflite(s_model, im, file, int8 or edgetpu, data=data, nms=nms, agnostic_nms=agnostic_nms)
            if edgetpu:
                f[8], _ = export_edgetpu(file)
            add_tflite_metadata(f[8] or f[7], metadata, num_outputs=len(s_model.outputs))
        if tfjs:
            f[9], _ = export_tfjs(file, int8)
    if paddle:  # PaddlePaddle
        f[10], _ = export_paddle(model, im, file, metadata)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        cls, det, seg = (isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel))  # type
        det &= not seg  # segmentation models inherit from SegmentationModel(DetectionModel)
        dir = Path("segment" if seg else "classify" if cls else "")
        h = "--half" if half else ""  # --half FP16 inference arg
        s = (
            "# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference"
            if cls
            else "# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference"
            if seg
            else ""
        )
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
            f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return list of exported files/dirs


def parse_opt(known=False):
    """
    Parse command-line arguments for model export configuration.

    Args:
        known (bool): If True, parse only known arguments and ignore others. Default is False.

    Returns:
        argparse.Namespace: Namespace object containing export configuration parameters.

    Example:
        ```python
        from ultralytics.export import parse_opt

        options = parse_opt(known=True)
        print(options)
        ```

    Notes:
        This function leverages `argparse` to handle command-line arguments for various model export configurations, allowing
        users to specify export formats, model parameters, and optimization settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov3-tiny.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 640], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv3 Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF/OpenVINO INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument("--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep")
    parser.add_argument("--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Run(**vars(opt))."""
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
