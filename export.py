# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLOv3 PyTorch model to other formats.

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

Usage:
    $ python export.py --weights yolov3-tiny.pt --include torchscript onnx openvino engine coreml paddle

Inference:
    $ python detect.py --weights yolov3-tiny.pt                 # PyTorch
                                 yolov3-tiny.torchscript        # TorchScript
                                 yolov3-tiny.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov3-tiny_openvino_model     # OpenVINO
                                 yolov3-tiny.engine             # TensorRT
                                 yolov3-tiny.mlmodel            # CoreML (macOS-only)
                                 yolov3-tiny_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import platform
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
from models.yolo import Detect
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
        """Initialize an iOSModel with normalized input dimensions and number of classes from a PyTorch model.

        Args:
            model (torch.nn.Module): The source PyTorch model, expected to expose an `nc` (number of classes) attribute.
            im (torch.Tensor): Sample input image tensor of shape (batch_size, channels, height, width), used to derive
                the input normalization factor.
        """
        super().__init__()
        _b, _c, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = model.nc  # number of classes
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)
            # np = model(im)[0].shape[1]  # number of points
            # self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h]).expand(np, 4)  # explicit (faster, larger)

    def forward(self, x):
        """Run a forward pass, returning class-scaled confidences and normalized box coordinates for input tensor `x`.

        Args:
            x (torch.Tensor): Input batch of images with shape (batch_size, channels, height, width). Dimensions should
                match those used to initialize the model so normalization stays correct.

        Returns:
            (torch.Tensor): Per-class confidence scores (class probabilities multiplied by objectness).
            (torch.Tensor): Normalized box coordinates in xywh format.
        """
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


def export_formats():
    """List YOLOv3 model formats with their `--include` argument, file suffix, and CPU/GPU compatibility.

    Returns:
        (pandas.DataFrame): Table with columns ["Format", "Argument", "Suffix", "CPU", "GPU"], one row per format.

    Notes:
        TensorFlow rows are retained for `DetectMultiBackend` suffix detection of externally produced models, but
        TensorFlow export is not supported by this repository's `export.py`.
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
    """Wrap a YOLOv3 export function to time it and log success (with elapsed time and file size) or failure.

    Args:
        inner_func (Callable): Export function returning a `(file_path, model)` tuple.

    Returns:
        (Callable): Wrapped function that profiles the export and returns `(None, None)` on failure instead of raising.
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
def export_torchscript(model, im, file, optimize, prefix=colorstr("TorchScript:")):  # noqa: B008
    """Export a YOLOv3 model to TorchScript format, with optional optimization for mobile deployment.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): A tensor representing the input image for the model, typically with shape (N, 3, H, W).
        file (pathlib.Path): The file path where the TorchScript model will be saved.
        optimize (bool): A boolean flag indicating whether to optimize the model for mobile devices.
        prefix (str): A prefix for logging messages. Defaults to `colorstr("TorchScript:")`.

    Returns:
        (pathlib.Path): Path to the saved TorchScript model.
        (None): Placeholder returned for interface consistency with the `try_export` decorator.

    Notes:
        Uses `torch.jit.trace` to trace the model. Metadata (input shape, stride, class names) is stored in an extra
        `config.txt` file embedded in the TorchScript archive.
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
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):  # noqa: B008
    """Export a YOLOv3 model to ONNX format with dynamic shape and simplification options.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): A sample input tensor for tracing the model.
        file (pathlib.Path): The file path where the ONNX model will be saved.
        opset (int): The ONNX opset version to use for the export.
        dynamic (bool): If `True`, enables dynamic shape support.
        simplify (bool): If `True`, simplifies the ONNX model using onnx-simplifier.
        prefix (str): A prefix for logging messages.

    Returns:
        (pathlib.Path): Path to the saved ONNX model.
        (onnx.ModelProto): The exported (and optionally simplified) ONNX model object.

    Notes:
        Requires `onnx` and, when `simplify=True`, `onnx-simplifier`. Uses `torch.onnx.export`, then optionally
        simplifies with `onnx-simplifier`. When `dynamic` is enabled, dynamic axes are added for variable input shapes.
        Model `stride` and `names` are embedded as ONNX metadata.
    """
    check_requirements("onnx>=1.12.0")
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    output_names = ["output0"]
    if dynamic:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
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
def export_openvino(file, metadata, half, int8, data, prefix=colorstr("OpenVINO:")):  # noqa: B008
    """Export a YOLOv3 model to OpenVINO format with optional INT8 quantization and inference metadata.

    Args:
        file (Path): Path to the output file.
        metadata (dict): Inference metadata to include in the exported model.
        half (bool): Indicates if FP16 precision should be used.
        int8 (bool): Indicates if INT8 quantization should be applied.
        data (str): Path to the dataset file (.yaml) for post-training quantization.
        prefix (str): Logging prefix for OpenVINO export messages.

    Returns:
        (str): Directory path of the exported OpenVINO model.
        (None): Placeholder returned for interface consistency with the `try_export` decorator.

    Notes:
        Requires `openvino-dev>=2023.0`, plus `nncf>=2.4.0` for INT8 post-training quantization. See the OpenVINO
        documentation for details: https://docs.openvino.ai/latest/index.html.
    """
    check_requirements("openvino-dev>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.runtime as ov
    from openvino.tools import mo

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

        def transform_fn(data_item):
            """Extract and preprocess a dataloader item into an input tensor for NNCF quantization calibration.

            Args:
                data_item (tuple): Item yielded by the calibration DataLoader during iteration.

            Returns:
                (numpy.ndarray): Preprocessed input tensor ready for quantization.
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
def export_paddle(model, im, file, metadata, prefix=colorstr("PaddlePaddle:")):  # noqa: B008
    """Export a YOLOv3 model to PaddlePaddle format using X2Paddle, writing model files and a metadata YAML.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Sample input tensor used for tracing the model.
        file (pathlib.Path): Source weights path with `.pt` suffix; the output directory is derived from it.
        metadata (dict): Metadata saved in YAML format alongside the exported model.
        prefix (str): Log message prefix. Defaults to a colored "PaddlePaddle:" string.

    Returns:
        (str): Directory path where the PaddlePaddle model is saved.
        (None): Placeholder returned for interface consistency with the `try_export` decorator.

    Notes:
        Requires the `paddlepaddle` and `x2paddle` packages. X2Paddle traces the model and writes the converted files
        and a metadata YAML to the output directory.
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
def export_coreml(model, im, file, int8, half, nms, prefix=colorstr("CoreML:")):  # noqa: B008
    """Export a YOLOv3 model to CoreML format with optional quantization and Non-Maximum Suppression (NMS).

    Args:
        model (torch.nn.Module): The YOLOv3 model to be exported.
        im (torch.Tensor): Input tensor used for tracing the model. Shape should be (batch_size, channels, height,
            width).
        file (pathlib.Path): Destination file path where the CoreML model will be saved.
        int8 (bool): Whether to use INT8 quantization. If True, quantizes the model to 8-bit integers.
        half (bool): Whether to use FP16 quantization. If True, converts the model to 16-bit floating point numbers.
        nms (bool): Whether to include Non-Maximum Suppression in the CoreML model.
        prefix (str): Prefix string for logging purposes. Default is colorstr("CoreML:").

    Returns:
        (pathlib.Path): Path to the saved CoreML model (.mlmodel).
        (coremltools.models.MLModel): The exported CoreML model object.

    Notes:
        - Requires `coremltools`.
        - When `nms` is enabled, the model is wrapped with `iOSModel` to include NMS.
        - INT8/FP16 quantization is only applied on macOS.
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
def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr("TensorRT:")):  # noqa: B008
    """Export a YOLOv3 model to TensorRT engine format, optimizing it for GPU inference.

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
        (pathlib.Path): Path to the saved TensorRT engine file.
        (None): Placeholder returned for interface consistency with the `try_export` decorator.

    Raises:
        AssertionError: If the model is on CPU instead of GPU.
        RuntimeError: If the intermediate ONNX file fails to load.

    Notes:
        Requires a TensorRT installation. Nvidia TensorRT: https://developer.nvidia.com/tensorrt
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


def pipeline_coreml(model, im, file, names, y, prefix=colorstr("CoreML Pipeline:")):  # noqa: B008
    """Processes and exports a YOLOv3 model into the CoreML model format, applying metadata and non-maximum suppression
    (NMS).

    Args:
        model (coremltools.models.MLModel): The pre-trained YOLOv3 CoreML model to be used for the pipeline.
        im (torch.Tensor): Input image tensor in BCHW (Batch, Channel, Height, Width) format with a shape (B, 3, H, W).
        file (pathlib.Path): Destination file path where the CoreML model will be saved.
        names (dict): A dictionary that maps class indices to class names.
        y (torch.Tensor): Output detection tensor from the YOLO model, containing predictions.
        prefix (str): Prefix for logging messages, default is "CoreML Pipeline:".

    Notes:
        - Adds an NMS stage to the CoreML model with overridable IoU and confidence thresholds.
        - Updates model metadata with class names, thresholds, author, and license.
        - The pipelined model is saved as a `.mlmodel` file at `file`.
        - Requires `coremltools` and is primarily intended for macOS, as CoreML is Apple-specific.
    """
    import coremltools as ct
    from PIL import Image

    print(f"{prefix} starting pipeline with coremltools {ct.__version__}...")
    _batch_size, _ch, h, w = list(im.shape)  # BCHW
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
    _na, nc = out0_shape
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
    pipeline.spec.description.metadata.versionString = "https://github.com/ultralytics/yolov3"
    pipeline.spec.description.metadata.shortDescription = "https://github.com/ultralytics/yolov3"
    pipeline.spec.description.metadata.author = "glenn.jocher@ultralytics.com"
    pipeline.spec.description.metadata.license = "https://github.com/ultralytics/yolov3/blob/master/LICENSE"
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
    weights=ROOT / "yolov3-tiny.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv3 Detect() inplace=True
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/OpenVINO INT8 quantization
    dynamic=False,  # ONNX/TensorRT: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # CoreML: add NMS to model
):
    """Export a YOLOv3 PyTorch model to one or more deployment formats (TorchScript, ONNX, CoreML, TensorRT, etc.).

    Args:
        data (str | Path): Path to the dataset configuration file.
        weights (str | Path): Path to the PyTorch model weights file.
        imgsz (tuple[int, int]): Input image (height, width).
        batch_size (int): Batch size for the export sample input.
        device (str): Device to use for export (e.g., '0', '0,1,2,3', 'cpu').
        include (tuple[str]): Formats to export (e.g., 'torchscript', 'onnx').
        half (bool): If True, export with FP16 precision.
        inplace (bool): If True, set the YOLOv3 Detect module `inplace` option to True.
        optimize (bool): If True, optimize the TorchScript model for mobile inference.
        int8 (bool): If True, apply INT8 quantization for CoreML/OpenVINO models.
        dynamic (bool): If True, enable dynamic axes for ONNX/TensorRT models.
        simplify (bool): If True, simplify the ONNX model after export.
        opset (int): ONNX opset version.
        verbose (bool): If True, enable verbose logging for TensorRT engine export.
        workspace (int): Workspace size in GB for the TensorRT engine.
        nms (bool): If True, add Non-Maximum Suppression (NMS) to the exported CoreML model.

    Returns:
        (list[str]): Paths to the exported model files or directories.

    Notes:
        Each format may require extra packages (e.g., `onnx`, `coremltools`, TensorRT).
    """
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans
    if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
        raise NotImplementedError(
            "TensorFlow exports are not supported by this repository. "
            "Use the ultralytics package (https://docs.ultralytics.com/modes/export/) for TensorFlow formats."
        )
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
    if paddle:  # PaddlePaddle
        f[10], _ = export_paddle(model, im, file, metadata)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = "--half" if half else ""  # --half FP16 inference arg
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python detect.py --weights {f[-1]} {h}"
            f"\nValidate:        python val.py --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov3', 'custom', '{f[-1]}')"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return list of exported files/dirs


def parse_opt(known=False):
    """Parse command-line arguments for YOLOv3 model export configuration.

    Args:
        known (bool): If True, parse only known arguments and ignore the rest. Default is False.

    Returns:
        (argparse.Namespace): Namespace containing the export configuration parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov3-tiny.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 640], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv3 Detect() inplace=True")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/OpenVINO INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="CoreML: add NMS to model")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine, coreml, paddle",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Run the export pipeline for each weights path in the parsed options."""
    weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
    for weight in weights:
        opt.weights = weight
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
