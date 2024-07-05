# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
Export a YOLOv3 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

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
    def __init__(self, model, im):
        """
        Initializes an iOSModel with normalized input dimensions and the number of classes from a given torch model.

        Args:
            model (torch.nn.Module): The PyTorch model to be converted, which contains structural and functional details
                necessary for inference.
            im (torch.Tensor): Sample input tensor used to determine dimensions (batch, channel, height, width). This helps
                in normalizing input dimensions and adjusting model parameters accordingly.

        Returns:
            None

        Notes:
            The iOSModel class facilitates the conversion of a trained PyTorch model to formats compatible with iOS devices,
            focusing on maintaining input dimension consistency and class information for inference.
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
        Performs forward pass, returning scaled confidences and normalized coordinates given the input tensor `x`.

        Args:
          x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
          torch.Tensor: A tensor containing the scaled confidences and normalized coordinates.

        Notes:
          The output tensor consists of three parts split along the last dimension:
          - xywh: Coordinates in the format (x, y, width, height).
          - conf: Confidence scores for each bounding box.
          - cls: Class predictions for each bounding box, scaled accordingly.
        """
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


def export_formats():
    """
    Lists supported YOLOv3 model export formats including file suffixes and CPU/GPU compatibility.

    Returns:
        list of list: A list containing sublists that each describe a supported model export format. Each sublist includes:
            - str: The name of the format (e.g., 'PyTorch').
            - str: The identifier used to specify that format for the export script.
            - str: The file suffix or directory name that corresponds to the exported model.
            - bool: Whether the format is compatible with CPU inference.
            - bool: Whether the format is compatible with GPU inference.

    Example:
    """python
        formats = export_formats()
        for fmt in formats:
            print(f"Format: {fmt[0]}, Identifier: {fmt[1]}, Suffix: {fmt[2]}, CPU: {fmt[3]}, GPU: {fmt[4]}")
        """
    
    Notes:
        To use these formats in practice via the `export.py` script, see the detailed usage examples provided in the 
        Ultralytics YOLOv3 repository: https://github.com/ultralytics/ultralytics.
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
    Profiles and logs the export process of YOLOv3 models, handling success and failure cases.

    Args:
        inner_func (Callable): The function responsible for exporting the model, typically a format-specific function.

    Returns:
        Callable: A wrapped function that executes `inner_func`, providing logging and profiling for the export process.

    Notes:
        This decorator can be applied to any function involved in the export process to facilitate timing and error
        handling. It captures the performance metrics using the `Profile` context manager and logs the outcomes,
        including success or failure and relevant details such as time elapsed and file size if successful.
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
    Exports a YOLOv3 model to TorchScript format, optionally optimizing for mobile.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Input tensor with shape (batch_size, channels, height, width) used for tracing the model.
        file (pathlib.Path): File path where the exported model will be saved.
        optimize (bool): If True, apply optimizations for mobile deployment.

    Returns:
        tuple (pathlib.Path, torch.jit.ScriptModule): A tuple containing the file path of the saved TorchScript model and
        the TorchScript module itself.

    Notes:
        For more details on the export process and dependencies, refer to the official documentation at
        https://github.com/ultralytics/ultralytics.

    Example:
        ```python
        torchscript_path, torchscript_model = export_torchscript(yolo_model, input_tensor, output_path, optimize=True)
        ```
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
    Exports a YOLOv3 PyTorch model to ONNX format with options for dynamic shapes and simplification.

    Args:
      model (torch.nn.Module): The YOLOv3 model to be exported.
      im (torch.Tensor): Sample input tensor for tracing the model.
      file (pathlib.Path): Output file path for the exported ONNX model.
      opset (int): The ONNX opset version to use for export.
      dynamic (bool): If True, enables dynamic axes for the model inputs and outputs.
      simplify (bool): If True, attempts to simplify the exported ONNX model.
      prefix (str): Log prefix denoting the export format. Defaults to "ONNX:".

    Returns:
      tuple[pathlib.Path, None]: A tuple containing the file path of the exported ONNX model and None.

    Notes:
      - Ensure required packages are installed using:
        ```sh
        pip install onnx onnx-simplifier onnxruntime
        ```
      - Dynamic axes are only supported on CPU.
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
    Exports a YOLOv3 model to OpenVINO format, with optional INT8 quantization and inference metadata.

    Args:
        file (Path): Path to the original model file.
        metadata (dict): Dictionary containing model's metadata.
        half (bool): If True, export with FP16 precision.
        int8 (bool): If True, perform INT8 quantization.
        data (str): Path to the dataset configuration file for quantization.
        prefix (str): Prefix for logging messages.

    Returns:
        tuple[Path, None | openvino.runtime.Model]:
            Path to the exported OpenVINO model directory and the OpenVINO model instance if INT8 quantization was
            not applied, otherwise None.

    Raises:
        Exception: If any error occurs during the export process or quantization.

    Notes:
        Requires `openvino-dev >= 2023.0`. For INT8 quantization, requires `nncf >= 2.4.0`.
        For more details and installation, see: https://pypi.org/project/openvino-dev/

    Examples:
        ```python
        export_openvino(Path('yolov5s.onnx'), metadata={"stride": 32, "names": ["person", "car"]}, half=False,
                        int8=True, data='data/coco.yaml')
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
    Exports a YOLOv3 model to PaddlePaddle format using X2Paddle.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Input tensor representing the input image with appropriate dimensions (e.g., (1, 3, 640, 640)).
        file (Path): Path object specifying where to save the exported PaddlePaddle model.
        metadata (dict): Dictionary containing metadata related to the model, such as stride and class names.
        prefix (str, optional): String prefix for logging purposes, typically indicating the export format (default is 'PaddlePaddle:').

    Returns:
        tuple: Contains the file path to the saved PaddlePaddle model (str) and None, as no additional model object is returned.

    Notes:
        This function requires 'paddlepaddle' and 'x2paddle' packages to be installed. You can install them using:
            $ pip install paddlepaddle x2paddle

    Example:
        ```python
        from pathlib import Path
        from torch import nn, randn

        model = nn.Module()  # Replace with your YOLOv3 model
        im = randn(1, 3, 640, 640)  # Example input tensor
        file = Path("/path/to/save/model")
        metadata = {"stride": 32, "names": ["class1", "class2"]}

        export_paddle(model, im, file, metadata)
        ```

        The exported PaddlePaddle model will be saved in the specified directory with metadata saved in a YAML file.
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
    Exports a YOLOv3 model to CoreML format with optional quantization and Non-Maximum Suppression (NMS).

    Args:
        model (torch.nn.Module): The YOLOv3 PyTorch model to be exported.
        im (torch.Tensor): Example input tensor for tracing the model.
        file (Path): Path where the CoreML model will be saved.
        int8 (bool): Quantize model weights to 8-bit integer format.
        half (bool): Quantize model weights to 16-bit floating point format.
        nms (bool): Apply Non-Maximum Suppression to the model.
        prefix (str): Optional string to prepend to log messages (default is colorstr("CoreML:")).

    Returns:
        Tuple[Path, None]: The path to the saved CoreML model file and None as an output placeholder.

    Raises:
        ValueError: If an unsupported quantization configuration is provided.
        RuntimeError: If CoreML model export or quantization fails.

    Notes:
        - The `coremltools` library is required for exporting to CoreML format. Install it via `pip install coremltools`.
        - Quantization is only supported on macOS; attempting quantization on other platforms will skip this step.
        - Ensure that the `torch.jit.trace` method is used appropriately when NMS is enabled, as it wraps the input model.

    Usage Example:
        ```python
        from ultralytics import export_coreml
        from models.yolo import model_yolo    # hypothetical module and function for creating a YOLO model

        model = model_yolo(weights='path/to/yolov3/weights.pt')
        input_tensor = torch.randn(1, 3, 640, 640)  # Example input tensor for tracing

        output_path, _ = export_coreml(
            model=model,
            im=input_tensor,
            file=Path('path/to/save/model.mlmodel'),
            int8=False,
            half=True,
            nms=True
        )
        print(f"Model saved at: {output_path}")
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
    Exports a YOLOv3 model to TensorRT engine format.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path): The output file path for saving the TensorRT engine.
        half (bool): A flag to enable FP16 precision.
        dynamic (bool): A flag to enable dynamic tensor shapes.
        simplify (bool): A flag to simplify the ONNX model for TensorRT conversion.
        workspace (int): The maximum GPU workspace size in GB.
        verbose (bool): A flag to enable verbose logging.
        prefix (str): The prefix string for logging messages.

    Returns:
        Tuple[Path, None | trt.ICudaEngine]: The path to the saved TensorRT engine and the TensorRT engine object (if applicable).

    Raises:
        RuntimeError: If ONNX model conversion fails.
        AssertionError: If the script is run on a CPU or required files and libraries are missing.

    Notes:
        - Nvidia TensorRT: https://developer.nvidia.com/tensorrt

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.yolo import DetectionModel
        from export import export_engine

        # Initialize model and dummy input
        model = DetectionModel(...)
        dummy_input = torch.randn(1, 3, 640, 640).cuda()

        # Export TensorRT engine
        trt_path, trt_engine = export_engine(model, dummy_input, Path("yolov5s.engine"), half=True, dynamic=True)
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
    Exports YOLOv3 model to TensorFlow SavedModel format; includes NMS and configuration options.

    Args:
        model (torch.nn.Module): YOLOv3 model to export.
        im (torch.Tensor): Input image tensor for tracing.
        file (Path): File path to save the exported model.
        dynamic (bool): Whether the model accepts variable batch sizes.
        tf_nms (bool, optional): Enable TensorFlow NMS. Defaults to False.
        agnostic_nms (bool, optional): Enable agnostic NMS that applies across all classes. Defaults to False.
        topk_per_class (int, optional): Limit top-k detections per class. Defaults to 100.
        topk_all (int, optional): Limit top-k detections across all classes. Defaults to 100.
        iou_thres (float, optional): IoU threshold for NMS. Defaults to 0.45.
        conf_thres (float, optional): Confidence threshold for detections. Defaults to 0.25.
        keras (bool, optional): Export as a Keras model. Defaults to False.
        prefix (str, optional): Prefix for logging messages. Defaults to colorstr("TensorFlow SavedModel:").

    Returns:
        (str | None, tf.Module | None): Path to the saved model directory, TensorFlow model module or None in case of failure.

    Examples:
        ```python
        export_saved_model(
            model=my_model,
            im=my_image_tensor,
            file=Path("yolov5_saved_model"),
            dynamic=True,
            tf_nms=True,
            topk_per_class=50,
            conf_thres=0.3
        )
        ```
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
    Exports a Keras model as TensorFlow GraphDef (`*.pb`) file, compatible with YOLOv3.

    Args:
        keras_model (tf.keras.Model): The Keras model to be exported.
        file (Path): The path where the exported `.pb` file will be saved.
        prefix (str): Optional logging prefix for export status messages (default is "TensorFlow GraphDef:").

    Returns:
        tuple: A tuple containing:
            - Path: The path to the saved `.pb` file.
            - tf.Graph: The frozen TensorFlow GraphDef.

    Notes:
        This function uses TensorFlow to convert the given Keras model into a frozen graph and save it as a `.pb` file.
        It requires TensorFlow runtime and can handle models created or imported into Keras. For more details, see
        [Frozen Graph TensorFlow](https://github.com/leimao/Frozen_Graph_TensorFlow).

    Examples:
        ```python
        from tensorflow.keras.models import load_model
        from pathlib import Path

        # Load Keras model
        keras_model = load_model('path/to/keras_model.h5')

        # Define file path
        file = Path('path/to/save/yolov3.pb')

        # Export to .pb format
        export_pb(keras_model, file)
        ```

    ```python
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    f = file.with_suffix(".pb")

    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)

    return f, frozen_func.graph
    ```
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
    Exports a YOLOv3 model to TensorFlow Lite (TFLite) format with optional quantization and non-maximum suppression
    (NMS).

    Args:
        keras_model (tf.keras.Model): The Keras model to be converted to TFLite.
        im (torch.Tensor): Sample input tensor for representative dataset generation.
        file (Path): Output file path for the TFLite model.
        int8 (bool): If True, apply integer quantization using a representative dataset.
        data (str): Path to dataset configuration file (YAML format) for quantization.
        nms (bool): If True, include TensorFlow custom NMS operation.
        agnostic_nms (bool): If True, apply class-agnostic NMS.

    Returns:
        (Path, None): Path to the saved TFLite model and None.

    Notes:
        - NMS in TensorFlow Lite requires SELECT_TF_OPS, which increases the TFLite model size.
        - Integer quantization improves performance on edge devices but requires a representative dataset.

    Examples:
        ```python
        # Example of exporting YOLOv3 model to TFLite with integer quantization
        from pathlib import Path
        import torch

        # Load model and sample input
        model = torch.load('yolov5s.pt')
        im = torch.rand(1, 3, 640, 640)  # sample input

        # Export
        export_tflite(model, im, Path('yolov5s'), int8=True, data='data/coco.yaml', nms=True, agnostic_nms=False)
        ```

        - TensorFlow: https://www.tensorflow.org/lite
        - YOLOv3: https://github.com/ultralytics/yolov3
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
    Export YOLOv3 model to Edge TPU compatible format; requires Linux and Edge TPU compiler.

    Args:
        file (Path): Path to the source file of the model to be compiled.
        prefix (str, optional): Color-coded prefix for logging statements. Defaults to colorstr("Edge TPU:").

    Returns:
        (None, None): This function does not return a file or model on success or failure.

    Notes:
        - Ensure that the Edge TPU compiler is installed and accessible. The compiler is only supported on Linux platforms.

    Example:
        ```python
        from pathlib import Path
        export_edgetpu(Path("yolov5s.pt"))
        ```

        For more details, visit https://coral.ai/docs/edgetpu/compiler/
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
    Exports a YOLOv3 model to TensorFlow.js format, optionally quantizing it to uint8.

    Args:
        file (Path): The file path where the model is located.
        int8 (bool): Flag indicating whether to apply uint8 quantization.
        prefix (str): Log prefix used for the export process, default is "TensorFlow.js:".

    Returns:
        tuple[Path, None]: The path to the exported TensorFlow.js model directory and None as the second element.

    Raises:
        RuntimeError: If there is an error during the TensorFlow.js conversion process.

    Notes:
        - This function requires `tensorflowjs` to be installed.
        - The output directory will contain the `model.json` and associated weight files, which can be used with
          TensorFlow.js for in-browser inference.
        - The TensorFlow.js converter makes use of TensorFlow's frozen model format during conversion, and if quantization
          is applied, the model will be optimized for lower precision uint8 operations.

    Examples:
        ```python
        from pathlib import Path
        file = Path("yolov3.pt")
        export_tfjs(file=file, int8=True)
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
    Adds metadata to a TensorFlow Lite (TFLite) model for enhanced model understanding and processing.

    Args:
        file (str): The path to the TFLite model file.
        metadata (dict): The metadata information to be added to the model, structured according to TFLite requirements.
        num_outputs (int): The number of output tensors in the model.

    Returns:
        None

    Examples:
        ```python
        metadata = {
            "model_name": "yolov3",
            "description": "A YOLOv3 model for object detection",
            "version": "1.0",
            "author": "Ultralytics",
        }
        add_tflite_metadata("yolov3.tflite", metadata, num_outputs=3)
        ```
    Notes:
        - This function requires the `tflite_support` package. Install it using `pip install tflite_support`.
        - The `tflite_support` library provides tools for TensorFlow Lite models metadata management.
        - More information is available at https://www.tensorflow.org/lite/convert/metadata
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
    Executes the CoreML pipeline for exporting a YOLOv3 model, handling image preprocessing, prediction, and NMS.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Tensor representing the input image, with shape (Batch, Channel, Height, Width).
        file (pathlib.Path): The file path where the exported CoreML model will be saved.
        names (dict[int, str]): Dictionary mapping class indices to class names.
        y (torch.Tensor): The model output tensor used to derive output shapes in non-macOS environments.
        prefix (str): Prefix for logging messages.

    Returns:
        tuple[pathlib.Path, coremltools.models.MLModel]: The file path of the saved CoreML model and the CoreML model object.

    Raises:
        AssertionError: If the number of class names doesn't match the number of classes detected in the model.

        Notes:
        - Requires `coremltools` to be installed.
        - For macOS, the model prediction outputs are used to derive output shapes; otherwise, PyTorch outputs are used.
        - The function includes steps for building non-maximum suppression (NMS) and flexible input shapes for the CoreML model.
        - The metadata for the saved model includes links to the YOLOv5 repository and descriptive information about the model.

        Example:
        ```python
        from pathlib import Path
        import torch

        # Assuming 'model', 'im', 'names', and 'y' are defined elsewhere
        file = Path("YOLOv3.mlmodel")
        pipeline_coreml(model, im, file, names, y)
        ```
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
    Exports a PyTorch model to specified formats like ONNX, CoreML, and TensorRT.

    Args:
        data (str | Path): Path to the dataset configuration file (default: ROOT / "data/coco128.yaml").
        weights (str | Path): Path to the model weights file (default: ROOT / "yolov5s.pt").
        imgsz (tuple[int, int]): Image dimensions (height, width) for model input (default: (640, 640)).
        batch_size (int): Batch size for export (default: 1).
        device (str): Device to run the export on, e.g., 'cpu' or 'cuda:0' (default: "cpu").
        include (tuple[str, ...]): Formats to include in the export, e.g., ('onnx', 'torchscript') (default:
                                    ("torchscript", "onnx")).
        half (bool): Use FP16 half-precision for export (default: False).
        inplace (bool): Set YOLOv3 Detect() inplace=True (default: False).
        keras (bool): Use Keras for exporting TensorFlow formats (default: False).
        optimize (bool): Optimize TorchScript for mobile (default: False).
        int8 (bool): Use INT8 quantization for CoreML/TF exports (default: False).
        dynamic (bool): Use dynamic shapes for ONNX/TF/TensorRT exports (default: False).
        simplify (bool): Simplify ONNX models (default: False).
        opset (int): ONNX opset version (default: 12).
        verbose (bool): Enable verbose logging for TensorRT export (default: False).
        workspace (int): TensorRT workspace size in GB (default: 4).
        nms (bool): Add Non-Maximum Suppression (NMS) to TensorFlow models (default: False).
        agnostic_nms (bool): Add class-agnostic NMS to TensorFlow models (default: False).
        topk_per_class (int): Top-K per class for TensorFlow.js NMS (default: 100).
        topk_all (int): Top-K for all classes for TensorFlow.js NMS (default: 100).
        iou_thres (float): IoU threshold for TensorFlow.js NMS (default: 0.45).
        conf_thres (float): Confidence threshold for TensorFlow.js NMS (default: 0.25).

    Returns:
        None

    Raises:
        AssertionError: If incompatible arguments are passed or prerequisites are not met.

    Example:
        ```python
        run(data="coco128.yaml", weights="yolov5s.pt", include=("onnx", "coreml"), device="cuda:0", half=True)
        ```

    Notes:
        - Ensure that the necessary libraries are installed for the desired export formats as outlined in the class
          docstring.
        - The --optimize option is only compatible with CPU devices.
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
            f'\nExport complete ({time.time() - t:.1f}s)'
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
            f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}"
            f'\nVisualize:       https://netron.app'
        )
    return f  # return list of exported files/dirs


def parse_opt(known=False):
    """
    Parses command line arguments for model export configuration, including data paths, weights, and export options.

    Args:
        known (bool): Whether to parse only known arguments. If True, unknown arguments are ignored. If False,
                      an error is raised for unknown arguments.

    Returns:
        argparse.Namespace: An object containing all the configured arguments and their values.

    Notes:
        The function uses `argparse.ArgumentParser` to define and parse arguments. It supports various model export
        configurations such as specifying dataset paths, model weights, export formats, and optimizations.
        Example usage to parse arguments:
        ```python
        opt = parse_opt()
        ```
        Example command line usage:
        ```bash
        python export.py --weights yolov5s.pt --include torchscript onnx openvino --imgsz 640 640 --half
        ```

        Some notable arguments include:
        - `--imgsz`: Specifies the input image size.
        - `--include`: Defines the export formats to be included like torchscript, onnx, etc.
        - `--half`: Indicates whether to export the model with FP16 half-precision.

        Full list of supported arguments and their descriptions can be found within the function definition.
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
    """
    ```python Executes a YOLOv3 model export to various formats such as TorchScript, ONNX, TensorRT, CoreML, TensorFlow
    SavedModel, TensorFlow GraphDef, TensorFlow Lite, Edge TPU, TensorFlow.js, and PaddlePaddle.

    Args:
        opt (Namespace): Parsed command-line arguments specifying export options and parameters.

    Returns:
        None

    Supported Formats:
        The function exports models in the following formats based on the `--include` argument:
        - `torchscript`: TorchScript format for PyTorch models.
        - `onnx`: ONNX format for interoperability with different frameworks.
        - `openvino`: OpenVINO format for optimized inference on Intel hardware.
        - `engine`: TensorRT engine format for optimized inference on Nvidia GPUs.
        - `coreml`: CoreML format for inference on Apple devices.
        - `saved_model`: TensorFlow SavedModel format for TensorFlow serving.
        - `pb`: TensorFlow GraphDef format.
        - `tflite`: TensorFlow Lite format for mobile and embedded devices.
        - `edgetpu`: TensorFlow Lite format optimized for Edge TPU.
        - `tfjs`: TensorFlow.js format for browser-based inference.
        - `paddle`: PaddlePaddle format for inference on PaddlePaddle framework.

    Example:
        Here is how you can invoke the `main` function with custom parsed options:
        ```python
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', default='yolov5s.pt', type=str, help='path to model weights')
        parser.add_argument('--imgsz', default=[640, 640], type=int, nargs='+', help='image size')
        parser.add_argument('--batch-size', default=1, type=int, help='batch size')
        parser.add_argument('--device', default='cpu', type=str, help='device to run inference on')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision')
        parser.add_argument('--include', default=['torchscript', 'onnx'], nargs='+', help='formats to export')
        opt = parser.parse_args()

        main(opt)
        ```

    Note:
        Ensure that all required dependencies for each export format are installed. Refer to the Ultralytics YOLOv3
        repository for detailed installation instructions and requirements: https://github.com/ultralytics/ultralytics
    ```
    """
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
