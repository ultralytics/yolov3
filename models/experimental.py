# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Experimental modules."""

import math

import numpy as np
import torch
from torch import nn
from ultralytics.utils.patches import torch_load

from utils.downloads import attempt_download


class Sum(nn.Module):
    """Computes the weighted or unweighted sum of multiple input layers per https://arxiv.org/abs/1911.09070."""

    def __init__(self, n, weight=False):  # n: number of inputs
        """Initialize a module to compute the weighted or unweighted sum of n inputs, with optional learnable weights.

        See https://arxiv.org/abs/1911.09070 for the weighted feature fusion details.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Performs forward pass, blending `x` elements with optional learnable weights.

        See https://arxiv.org/abs/1911.09070 for more.
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
    """Implements mixed depth-wise convolutions for efficient neural networks; see https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        """Initialize MixConv2d with mixed depth-wise convolution layers; see https://arxiv.org/abs/1907.09595."""
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
        """Applies a series of convolutions, batch normalization, and SiLU activation to input tensor `x`."""
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Combines outputs from multiple models to improve inference results."""

    def __init__(self):
        """Initializes an ensemble of models to combine their outputs."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Applies ensemble of models on input `x`, with options for augmentation, profiling, and visualization,
        returning inference outputs.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """Load a single model or an ensemble of models from one or more checkpoints.

    Args:
        weights (str | list[str]): Path or list of paths to model checkpoint(s); missing files are auto-downloaded.
        device (torch.device, optional): Device to load the model onto.
        inplace (bool): Use inplace ops (e.g. slice assignment) for compatibility across torch versions.
        fuse (bool): Fuse Conv2d and BatchNorm2d layers for faster inference.

    Returns:
        (torch.nn.Module): The loaded model, or an `Ensemble` if multiple weights are provided.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch_load(attempt_download(w), map_location="cpu")  # load
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
                m.anchor_grid = [torch.zeros(1)] * m.nl
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
