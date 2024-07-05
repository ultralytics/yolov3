# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        """
        Initializes a module to compute the weighted or unweighted sum of two or more inputs, with optional learning
        weights.

        Args:
            n (int): The number of input layers to sum.
            weight (bool): If True, adds learnable weights to the input layers before summing. Defaults to False.

        Returns:
            None: This method does not return any value (None).

        Notes:
            The weighted sum implementation follows the idea proposed in the paper "Bag of Freebies for Training Object Detection Neural Networks" (https://arxiv.org/abs/1911.09070).

        Example:
            ```python
            import torch
            from ultralytics import Sum

            # Assume feature maps fm1, fm2, and fm3 have the same shape
            fm1, fm2, fm3 = torch.rand(1, 256, 13, 13), torch.rand(1, 256, 13, 13), torch.rand(1, 256, 13, 13)

            # Create a Sum module to unweighted sum of three feature maps
            sum_module = Sum(n=3, weight=False)
            result = sum_module([fm1, fm2, fm3])
            ```
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """
        Performs forward pass, computing the weighted/unweighted sum of input elements.

        Args:
            x (list[torch.Tensor]): A list of input tensors whose weighted/unweighted sum needs to be computed.

        Returns:
            torch.Tensor: The resultant tensor after computing the sum of input tensors with or without weights.

        Note:
            Refer to the paper (https://arxiv.org/abs/1911.09070) for more information on the weighted sum operation.

        Example:
            ```python
            from ultralytics import Sum
            import torch

            model = Sum(n=3, weight=True)
            x = [torch.randn(1, 3, 224, 224) for _ in range(3)]
            output = model(x)
            ```
        """
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        """
        Initializes MixConv2d with device-specific mixed depth-wise convolution layers.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int], optional): Tuple of kernel sizes for each group. Defaults to (1, 3).
            s (int, optional): Convolution stride value. Defaults to 1.
            equal_ch (bool, optional): If True, assigns equal intermediate channels per group. If False, assigns equal
                weight per group based on kernel sizes. Defaults to True.

        Returns:
            None

        Note:
            This layer is based on techniques described in https://arxiv.org/abs/1907.09595.

        Examples:
            ```python
            # Example instantiation
            mixconv = MixConv2d(c1=64, c2=128, k=(3, 5), s=1, equal_ch=False)
            ```
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Applies mixed depth-wise convolutions followed by batch normalization and SiLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape `(N, C_in, H, W)` where `N` is batch size, `C_in` is the number of channels,
                `H` is height, and `W` is width.

        Returns:
            torch.Tensor: Tensor after application of mixed convolutions, batch normalization, and SiLU activation,
                maintaining shape `(N, C_out, H_out, W_out)` where `C_out` is determined by the number of convolution
                operations defined in the initialization, and `H_out`, `W_out` can change according to the convolution
                operations and stride.
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        """
        Initializes an ensemble of models to combine their outputs.

        Notes:
            This class extends `torch.nn.ModuleList` and can be used to manage and compute the combined output
            of multiple models. It is useful in scenarios where model ensembling is applied to improve
            predictions by aggregating outputs from different trained models.
        """
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Applies an ensemble of models to the input, with options for augmentation, profiling, and visualization.

        Args:
          x (torch.Tensor): Input tensor to be processed by the ensemble of models.
          augment (bool): If True, applies test-time augmentation. Default is False.
          profile (bool): If True, profiles model runtime and memory usage. Default is False.
          visualize (bool): If True, enables visualization of specific layers or outputs. Default is False.

        Returns:
          torch.Tensor: Output tensor after applying the ensemble of models. The returned tensor will combine
          results from all models in the ensemble using concatenation along the channel dimension.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Attempts to load one or more model weights and returns a model or an ensemble of models.

    Args:
        weights (str | list[str]): Path(s) to the weight file(s) to load.
        device (str | torch.device, optional): Device to load the model on (e.g., 'cpu', 'cuda'). Defaults to None, which
            means the model will be loaded on the default device.
        inplace (bool, optional): Whether to load the model layers with inplace operations. Defaults to True.
        fuse (bool, optional): Whether to fuse model layers to reduce model size and inference time. Defaults to True.

    Returns:
        torch.nn.Module: Loaded model or ensemble of models in evaluation mode.

    Notes:
        - The method supports loading from individual or multiple weight files.
        - It ensures compatibility with various torch versions and performs necessary model updates.
        - When multiple weights are provided, it returns an ensemble of models with properties from the first loaded model.

    Example:
    ```python
    model = attempt_load('weights.pt', device='cuda')
    ```

    For more information, visit: https://github.com/ultralytics/ultralytics.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
