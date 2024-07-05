# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""PyTorch utils."""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
warnings.filterwarnings("ignore", category=UserWarning)


def smart_inference_mode(torch_1_9=check_version(torch.__version__, "1.9.0")):
    """
    Applies `torch.inference_mode()` for PyTorch versions >= 1.9.0, otherwise uses `torch.no_grad()`.

    Args:
      torch_1_9 (bool): Indicates if the current PyTorch version is >= 1.9.0. Default is determined by `check_version`.

    Returns:
      Callable: A decorator that applies `torch.inference_mode()` or `torch.no_grad()` based on the PyTorch version.

    Examples:
      ```python
      @smart_inference_mode()
      def inference_function():
          # Your inference code here
      ```

    Note:
      This function leverages `torch.inference_mode()` for optimized inference operations in newer PyTorch versions and falls back to `torch.no_grad()` for versions below 1.9.0, ensuring compatibility and performance.
    """

    def decorate(fn):
        """Applies torch.inference_mode() if torch>=1.9.0, otherwise torch.no_grad(), as a decorator to functions."""
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def smartCrossEntropyLoss(label_smoothing=0.0):
    """
    Returns CrossEntropyLoss with optional label smoothing for PyTorch versions >= 1.10.0. Generates a warning if label
    smoothing is specified but an older PyTorch version is detected.

    Args:
        label_smoothing (float, optional): Factor for label smoothing. A value of 0.0 means no label smoothing.
            Default is 0.0.

    Returns:
        nn.CrossEntropyLoss: A CrossEntropyLoss object with the specified label smoothing if PyTorch version >= 1.10.0.

    Notes:
        Label smoothing is a regularization technique to make the model less confident and prevent overfitting by
        assigning some probability to incorrect classes. This feature was introduced in PyTorch 1.10.0.

    Example:
        ```python
        import torch.nn as nn
        from ultralytics import smartCrossEntropyLoss

        criterion = smartCrossEntropyLoss(label_smoothing=0.1)
        ```
    """
    if check_version(torch.__version__, "1.10.0"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f"WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0")
    return nn.CrossEntropyLoss()


def smart_DDP(model):
    """
    Initializes Distributed Data Parallel (DDP) for a model with version checks, excluding torch==1.12.0 due to known
    issues.

    Args:
        model (torch.nn.Module): The model to be wrapped with Distributed Data Parallel (DDP).

    Returns:
        torch.nn.parallel.DistributedDataParallel: The model wrapped with DDP if torch version is compatible.

    Raises:
        AssertionError: If torch version is 1.12.0 due to known DDP training issues.

    Note:
        For more details on the known DDP issues with torch==1.12.0, see:
        https://github.com/ultralytics/yolov5/issues/8395

    Example:
        ```python
        import torch
        from ultralytics import smart_DDP

        model = MyModel()
        model = smart_DDP(model)
        ```
    """
    assert not check_version(torch.__version__, "1.12.0", pinned=True), (
        "torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. "
        "Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395"
    )
    if check_version(torch.__version__, "1.11.0"):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def reshape_classifier_output(model, n=1000):
    """
    Reshapes the last layer of a model to have 'n' outputs, supporting various architectures like YOLOv3, ResNet, and
    EfficientNet.

    Args:
        model (torch.nn.Module): The neural network model whose output layer is to be reshaped.
        n (int): The number of outputs for the reshaped last layer. Defaults to 1000.

    Returns:
        torch.nn.Module: The model with the reshaped output layer.

    Notes:
        - The function adjusts `nn.Linear` and `nn.Conv2d` layers within the architectures.
        - For YOLOv3 models, it specifically updates the `Classify().linear` layer.
        - ResNet and EfficientNet architectures have their final `nn.Linear` layers adjusted.
        - In the case of a `nn.Sequential` container in the model, it identifies and modifies `nn.Linear` or `nn.Conv2d`
          layers accordingly.
    """
    from models.common import Classify

    name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
    if isinstance(m, Classify):  # YOLOv3 Classify() head
        if m.linear.out_features != n:
            m.linear = nn.Linear(m.linear.in_features, n)
    elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
        if m.out_features != n:
            setattr(model, name, nn.Linear(m.in_features, n))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = types.index(nn.Linear)  # nn.Linear index
            if m[i].out_features != n:
                m[i] = nn.Linear(m[i].in_features, n)
        elif nn.Conv2d in types:
            i = types.index(nn.Conv2d)  # nn.Conv2d index
            if m[i].out_channels != n:
                m[i] = nn.Conv2d(m[i].in_channels, n, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Context manager ensuring ordered execution in distributed training by synchronizing local masters first.

    Args:
        local_rank (int): Rank of the local process among distributed processes. Typically -1, 0, or a positive integer
                          representing the specific GPU assigned to the process.

    Returns:
        None

    Notes:
        In distributed training, it's crucial to control the execution order. This context manager facilitates that by
        synchronizing local rank 0 or -1 to run first, making sure that initial setup or shared resources are properly
        initialized or accessed before other ranks proceed. Appropriate in scenarios where certain operations need to be
        run by a single process before allowing others to continue.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def device_count():
    """
    Returns the count of available CUDA devices; supports Linux and Windows, using nvidia-smi.

    Returns:
        int: Number of available CUDA devices.

    Raises:
        AssertionError: If the operating system is not Linux or Windows.
        subprocess.CalledProcessError: If the subprocess command for nvidia-smi fails.

    Examples:
        ```python
        cuda_device_count = device_count()
        ```

    Notes:
        This function relies on the nvidia-smi command to count available CUDA devices and differentiates commands
        based on the operating system. Ensure nvidia-smi is installed and accessible in your environment.
    """
    assert platform.system() in ("Linux", "Windows"), "device_count() only supported on Linux or Windows"
    try:
        cmd = "nvidia-smi -L | wc -l" if platform.system() == "Linux" else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device="", batch_size=0, newline=True):
    """
    Selects the device for running models, handling CPU, GPU, and MPS with optional batch size divisibility check.

    Args:
        device (str): Specify the device to use ('cpu', 'cuda:<device_id>', 'mps'). Defaults to an empty string, which
            auto-selects the available hardware.
        batch_size (int): Batch size for training or inference, used to ensure divisibility when using multiple GPUs.
            Defaults to 0.
        newline (bool): Whether to append a newline to the log output. Defaults to True.

    Returns:
        torch.device: The selected torch device ('cpu', 'cuda:<device_id>', 'mps').

    Raises:
        AssertionError: If an invalid CUDA device is specified or if batch size is not divisible by the number of selected
            GPUs.

    Examples:
        ```python
        device = select_device(device="0,1", batch_size=64)
        model = model.to(device)
        ```
    """
    s = f"YOLOv3 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def time_sync():
    """
    Synchronizes PyTorch across available CUDA devices and returns current time in seconds.

    Returns:
        float: The current time in seconds since the epoch as a floating point number.

    Notes:
        This function ensures that all CUDA operations are finished before measuring the time, providing a more accurate
        timing of GPU operations. If CUDA is not available, it simply returns the current time.

    Examples:
        ```python
        start_time = time_sync()
        # perform operations
        end_time = time_sync()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        ```

    References:
        https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    """
    Profiles the speed, memory usage, and FLOPs of a given set of operations on a specified input.

    Args:
      input (torch.Tensor | list[torch.Tensor]): Input tensor or list of input tensors to be profiled.
      ops (callable | list[callable]): Operation or list of operations (e.g., layers, functions) to be applied to the input.
      n (int, optional): Number of iterations to run each profiling. Defaults to 10.
      device (torch.device | str | None, optional): Device to run the profiling on. If None, a suitable device is selected
        automatically. Defaults to None.

    Returns:
      list[dict[str, Any]]: List of profiling results where each entry contains:
        - 'params' (int): Number of parameters.
        - 'gflops' (float): Number of GFLOPs.
        - 'gpu_mem' (float): GPU memory usage in GB.
        - 'forward' (float): Average forward pass time in milliseconds.
        - 'backward' (float | None): Average backward pass time in milliseconds (if applicable).
        - 'input_shape' (tuple | str): Shape of input tensor or 'list' if input is a list.
        - 'output_shape' (tuple | str): Shape of output tensor or 'list' if output is a list.

    Example:
      Profile a few operations with a random input tensor:

      ```python
      import torch
      from torch import nn
      from ultralytics import profile

      input = torch.randn(16, 3, 640, 640)
      ops = [lambda x: x * torch.sigmoid(x), nn.SiLU()]
      results = profile(input, ops, n=100)
      ```

    Note:
      Ensure `thop` package is installed for FLOPs computation: `pip install thop`.
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    """
    Checks if a model is utilizing DataParallel (DP) or DistributedDataParallel (DDP) for training.

    Args:
        model (torch.nn.Module): The model to be checked.

    Returns:
        bool: True if the model is either wrapped in DataParallel or DistributedDataParallel; False otherwise.

    Notes:
        - This function is essential for verifying if a model has been parallelized, which can impact training behavior and performance.
        - Proper detection helps in adjusting subsequent operations accordingly to avoid errors related to distributed computing setups.

    Example:
        ```python
        import torch
        from torch.nn import DataParallel
        from torch.nn.parallel import DistributedDataParallel as DDP
        from ultralytics.utils import is_parallel

        model = nn.Linear(10, 2)
        dp_model = DataParallel(model)
        ddp_model = DDP(model)

        print(is_parallel(model))       # False
        print(is_parallel(dp_model))    # True
        print(is_parallel(ddp_model))   # True
        ```
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """
    Returns a single-GPU model if input model is using DataParallel (DP) or DistributedDataParallel (DDP).

    Args:
        model (nn.Module): A PyTorch model potentially wrapped in DataParallel or DistributedDataParallel.

    Returns:
        nn.Module: The underlying model without DataParallel or DistributedDataParallel wrapping.

    Examples:
        ```python
        model = torch.nn.DataParallel(my_model)
        single_gpu_model = de_parallel(model)
        ```

    Note:
        This function is useful when you need to access or manipulate the original model attributes or methods
        unaffected by the DP or DDP wrappers.
    """
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    """
    Initializes weights for various layers in a model, including Conv2d, BatchNorm2d, and certain activation functions.

    Args:
      model (torch.nn.Module): The neural network model whose weights are to be initialized.

    Returns:
      None

    Notes:
      - Conv2d layers are left unchanged.
      - BatchNorm2d layers have their epsilon (`eps`) and momentum parameters set to 1e-3 and 0.03, respectively.
      - Activation layers such as Hardswish, LeakyReLU, ReLU, ReLU6, and SiLU are also supported for potential modifications.

    Examples:
      ```python
      import torch.nn as nn
      from your_module import initialize_weights

      model = nn.Sequential(
          nn.Conv2d(1, 32, kernel_size=3, stride=1),
          nn.BatchNorm2d(32),
          nn.ReLU(inplace=True)
      )

      initialize_weights(model)
      ```

    References:
      - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
      - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
      - https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    """
    Finds indices of layers in the model that match the specified module class.

    Args:
        model (torch.nn.Module): The model in which to search for matching layers.
        mclass (type, optional): The type of layer to search for. Defaults to `torch.nn.Conv2d`.

    Returns:
        list of int: Indices of the layers in the model that match the specified module class.
    """
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    """
    Calculates and returns the global sparsity of a model.

    Args:
        model (torch.nn.Module): The model for which sparsity is to be calculated.

    Returns:
        float: The global sparsity of the model, defined as the ratio of zero-valued parameters to total parameters.

    Notes:
        Sparsity is a useful metric for understanding the proportion of zero-valued weights in a compressed or optimized model.
        This function iterates over all parameters in the input model and counts the total and zero-valued parameters to
        compute the sparsity.
    """
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    """
    Prunes Conv2d layers in a model to a specified global sparsity using L1 unstructured pruning.

    Args:
        model (torch.nn.Module): The model containing Conv2d layers to be pruned.
        amount (float): The fraction of connections to prune globally. Must be between 0 and 1. Default is 0.3.

    Returns:
        None

    Notes:
        This function modifies the model in place by pruning weights of Conv2d layers and then making the pruning
        permanent. The pruning is based on L1 norm, removing the smallest weights first.

    Example:
        ```python
        import torch
        import torchvision.models as models
        from ultrlalytics.utils import prune

        # Load a pre-trained model
        model = models.resnet18(pretrained=True)

        # Prune the model to 50% sparsity
        prune(model, amount=0.5)
        ```
    """
    import torch.nn.utils.prune as prune

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    LOGGER.info(f"Model pruned to {sparsity(model):.3g} global sparsity")


def fuse_conv_and_bn(conv, bn):
    """
    Fuses Conv2d and BatchNorm2d layers for efficiency.

    Args:
      conv (nn.Conv2d): Convolutional layer to be fused.
      bn (nn.BatchNorm2d): Batch normalization layer to be fused.

    Returns:
      nn.Conv2d: Fused convolutional layer.
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, imgsz=640):
    """
    Prints detailed information about the model, including layer configurations, parameters, gradients, and GFLOPs if
    verbose.

    Args:
        model (torch.nn.Module): The model to be analyzed.
        verbose (bool): Option to print detailed layer information, including mean and standard deviation of parameters.
        imgsz (int | list): Input image size for FLOPs calculation, as an integer or a list representing height and width.

    Returns:
        None

    Examples:
        ```python
        import torch
        from models.yolo import Model

        model = Model(cfg='models/yolov3.cfg')
        model_info(model, verbose=True)
        ```

    Notes:
        - If verbose is True, detailed information about each layer is printed, including the number of parameters, shape,
          and stats.
        - GFLOPs estimation is performed using thop, and the value is printed as part of the summary.
        - The function handles different image sizes and calculates the GFLOPs accordingly.

    See Also:
        - https://github.com/ultralytics/yolov5/issues/8395
    """
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1e9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f", {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs"  # 640x640 GFLOPs
    except Exception:
        fs = ""

    name = Path(model.yaml_file).stem.replace("yolov5", "YOLOv3") if hasattr(model, "yaml_file") else "Model"
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """
    Scales and optionally pads an image tensor to a specified ratio, maintaining its aspect ratio constrained by `gs`.

    Args:
        img (torch.Tensor): The image tensor to scale, with shape (B, C, H, W).
        ratio (float, optional): The scaling ratio to apply to the image dimensions. Default is 1.0.
        same_shape (bool, optional): If True, the image will be padded or cropped to maintain the original shape
                                     after scaling. Default is False.
        gs (int, optional): Grid size constraint for padding. The new dimensions will be multiples of `gs`. Default is 32.

    Returns:
        torch.Tensor: The scaled (and optionally padded) image tensor, with shape (B, C, H', W').

    Note:
        - This function uses bilinear interpolation for resizing.
        - If `same_shape` is False, the output shape will be multiples of `gs`.

    Example:
        ```python
        import torch
        from ultralytics import scale_img

        img = torch.randn(16, 3, 256, 416)
        scaled_img = scale_img(img, ratio=0.5, same_shape=True)
        ```
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """
    Copies attributes from object `b` to object `a`, with options to include or exclude specific attributes.

    Args:
        a (object): The target object to copy attributes to.
        b (object): The source object from which attributes are copied.
        include (tuple | list, optional): Specific attribute names to include in the copying process. Default is an empty tuple.
        exclude (tuple | list, optional): Specific attribute names to exclude from the copying process. Default is an empty tuple.

    Returns:
        None

    Notes:
        This function allows for selective copying of attributes. If `include` is specified, only those attributes are copied
        unless they are also in `exclude`. Attributes starting with an underscore are always excluded, unless explicitly
        included in `include`.
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    """
    Initializes a smart optimizer for YOLOv3 with custom parameter groups for different weight decays and biases.

    Args:
        model (nn.Module): The model whose parameters will be optimized.
        name (str): The type of optimizer to use. Options are 'Adam', 'AdamW', 'RMSProp', and 'SGD'. Default is 'Adam'.
        lr (float): The learning rate for the optimizer. Default is 0.001.
        momentum (float): The momentum factor for optimizers that use it (SGD and RMSProp). Default is 0.9.
        decay (float): The weight decay (L2 penalty) to apply on the parameters. Default is 1e-5.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        NotImplementedError: If an unsupported optimizer name is provided.

    Notes:
        - Separate parameter groups are created to apply different training configurations: one group for biases (no
          decay), one for batch normalization weights (no decay), and one for other weights (with decay).
        - This function currently supports 'Adam', 'AdamW', 'RMSProp', and 'SGD' optimizers.

    Example:
        ```python
        model = ...  # some nn.Module instance
        optimizer = smart_optimizer(model, name='SGD', lr=0.01, momentum=0.9, decay=5e-4)
        ```
    """
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer


def smart_hub_load(repo="ultralytics/yolov5", model="yolov5s", **kwargs):
    """
    Loads a YOLO model from the Ultralytics repository with advanced error handling and optional force reload.

    Args:
        repo (str, optional): The repository from which to load the model. Defaults to 'ultralytics/yolov5'.
        model (str, optional): The specific model to load from the repository. Defaults to 'yolov5s'.
        **kwargs: Additional keyword arguments for torch.hub.load.

    Returns:
        torch.nn.Module: Loaded YOLO model.

    Notes:
        - For `torch>=1.9.1`, validation is skipped to avoid GitHub API rate limit errors.
        - For `torch==1.12.0` and later, the `trust_repo` argument is set to True due to new requirements in torch hub.

    Example:
        ```python
        model = smart_hub_load(repo="ultralytics/yolov5", model="yolov5m", pretrained=True)
        ```

    See Also:
        https://github.com/ultralytics/yolov5
    """
    if check_version(torch.__version__, "1.9.1"):
        kwargs["skip_validation"] = True  # validation causes GitHub API rate limit errors
    if check_version(torch.__version__, "1.12.0"):
        kwargs["trust_repo"] = True  # argument required starting in torch 0.12
    try:
        return torch.hub.load(repo, model, **kwargs)
    except Exception:
        return torch.hub.load(repo, model, force_reload=True, **kwargs)


def smart_resume(ckpt, optimizer, ema=None, weights="yolov5s.pt", epochs=300, resume=True):
    """
    Resumes or fine-tunes training from a checkpoint with optimizer and EMA (Exponential Moving Average) support,
    updating epochs based on progress.

    Args:
      ckpt (dict): Checkpoint dictionary containing training state (e.g., optimizer state, current epoch).
      optimizer (torch.optim.Optimizer): Optimizer instance to be resumed.
      ema (optional): EMA instance to be resumed. Defaults to None.
      weights (str): Path to the model checkpoint file. Defaults to "yolov5s.pt".
      epochs (int): Total number of epochs to train. Defaults to 300.
      resume (bool): Flag indicating whether to resume training or start fine-tuning. Defaults to True.

    Returns:
      None

    Notes:
      - If the `optimizer` state is present in the checkpoint and not None, it is loaded.
      - If `ema` and its state are present in the checkpoint, they are loaded as well.
      - The resume flag checks if the start epoch is greater than 0, asserting training can be resumed.
      - Handles fine-tuning by adjusting the total number of epochs if the training should continue beyond the loaded checkpoint's epoch.

    Example:
    ```python
    # Example usage
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ckpt = torch.load('path/to/checkpoint.pt')
    smart_resume(ckpt, optimizer, weights='best.pt', epochs=150, resume=True)
    ```

    See also:
      - [YOLOv3 repository](https://github.com/ultralytics/yolov5)
    """
    best_fitness = 0.0
    start_epoch = ckpt["epoch"] + 1
    if ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
        best_fitness = ckpt["best_fitness"]
    if ema and ckpt.get("ema"):
        ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
        ema.updates = ckpt["updates"]
    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        LOGGER.info(f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs")
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt["epoch"]  # finetune additional epochs
    return best_fitness, start_epoch, epochs


class EarlyStopping:
    # YOLOv3 simple early stopper
    def __init__(self, patience=30):
        """
        Initializes the EarlyStopping mechanism.

        Args:
            patience (int | float): The number of epochs to wait for an improvement before stopping the training process.
                Setting to 'float("inf")' allows indefinite waiting. Defaults to 30.

        Returns:
            None

        Notes:
            EarlyStopping is useful to prevent overfitting by halting training once no further performance gain is observed
            on a validation set. This is especially important in deep learning models where excessive overtraining can lead
            to poorer generalization on unseen data.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Updates stopping criteria based on the current epoch and fitness value, determining whether to halt training.

        Args:
            epoch (int): The current epoch number.
            fitness (float): The current fitness value (e.g., mAP).

        Returns:
            bool: True if training should stop based on early stopping criteria, False otherwise.

        Notes:
            - This function should be called at the end of each epoch to monitor training progress.
            - The 'fitness' value should be a scalar metric indicating model performance (higher is better).
            - Upon determining that training should stop, a log message will be generated indicating the reason for stopping
            and the best epoch observed.
        """
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
            )
        return stop


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """
        Initializes the Exponential Moving Average (EMA) for a model, setting up its state and parameters.

        Args:
            model (torch.nn.Module): The model to apply EMA to.
            decay (float, optional): EMA decay rate. Defaults to 0.9999.
            tau (int, optional): Controls the ramp-up decay to help early epochs. Defaults to 2000.
            updates (int, optional): Number of EMA updates, typically starts at 0. Defaults to 0.

        Returns:
            None

        Notes:
            - Keeps a moving average of everything in the model's state_dict (parameters and buffers).
            - Sets the model to evaluation mode.

        Example:
            ```python
            model = YourModel()
            ema = ModelEMA(model)
            ```
        """
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Updates the Exponential Moving Average (EMA) of the model parameters.

        Args:
          model (torch.nn.Module): The PyTorch model whose parameters are used to update the EMA.

        Returns:
          None
        """
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """
        Updates EMA attributes by copying from model, excluding 'process_group' and 'reducer' by default.

        Args:
            model (nn.Module): The source model from which attributes are copied.
            include (tuple): Tuple of attribute names to include in the update. Defaults to an empty tuple, meaning all attributes are included unless specified in exclude.
            exclude (tuple): Tuple of attribute names to exclude from the update. Defaults to ('process_group', 'reducer').

        Returns:
            None

        Note:
            This function iterates through the source model's attributes, copying them to the EMA model except for those
            specified in the exclude tuple or those not in the include tuple if include is specified.

        Example:
            ```python
            ema = ModelEMA(model)
            ema.update_attr(some_model, include=('attr1', 'attr2'), exclude=('process_group', 'reducer'))
            ```
        """
        copy_attr(self.ema, model, include, exclude)
