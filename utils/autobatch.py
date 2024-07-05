# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Auto-batch utils."""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    """
    Checks and computes the optimal training batch size for YOLOv3, given the model and image size.

    Args:
        model (torch.nn.Module): The YOLOv3 model for which the batch size is being checked.
        imgsz (int): The input image size (default is 640).
        amp (bool): Automatic mixed precision flag. If True, uses torch.cuda.amp for half-precision training (default is True).

    Returns:
        int: The optimal batch size for training the provided YOLOv3 model with the given image size.

    Examples:
        ```python
        import torch
        from yolov3 import YOLOv3  # hypothetical import
        from utils.auto_batch import check_train_batch_size

        # Initialize model
        model = YOLOv3()

        # Determine optimal batch size
        optimal_batch_size = check_train_batch_size(model, imgsz=640, amp=True)
        print(f'Optimal Batch Size: {optimal_batch_size}')
        ```
    """
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    """
    Estimates the optimal batch size for training a YOLOv3 model based on available CUDA memory.

    Args:
        model (torch.nn.Module): The YOLOv3 model to be evaluated.
        imgsz (int): Image size used for inference, defaults to 640.
        fraction (float): Fraction of available CUDA memory to use, defaults to 0.8.
        batch_size (int): Initial batch size to start estimation, defaults to 16.

    Returns:
        int: The estimated optimal batch size for the given model and image size.

    Notes:
        - If CUDA is not detected, the default CPU batch size is returned.
        - Ensure `torch.backends.cudnn.benchmark` is set to False for accurate estimation.
        - The function profiles the model by evaluating it with various batch sizes and fits a polynomial to estimate the
        optimal batch size.
        - In case of CUDA anomalies or if the computed batch size is outside the safe range, the initial batch size is used
        with warnings logged.

    Example:
        ```python
        import torch
        from ultralytics import autobatch

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
        optimal_batch_size = autobatch(model)
        print(f"Optimal batch size: {optimal_batch_size}")
        ```

    References:
        https://github.com/ultralytics/ultralytics
    """
    # Usage:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    # Check device
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for --imgsz {imgsz}")
    device = next(model.parameters()).device  # get model device
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f"{prefix}{e}")

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[: len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        LOGGER.warning(f"{prefix}WARNING ⚠️ CUDA anomaly detected, recommend restart environment and retry command.")

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
    return b
