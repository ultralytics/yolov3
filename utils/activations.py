# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Activation functions."""

import torch
import torch.nn.functional as F
from torch import nn


class SiLU(nn.Module):
    """Applies the SiLU activation function to the input tensor as described in https://arxiv.org/pdf/1606.08415.pdf."""

    @staticmethod
    def forward(x):
        """Apply the SiLU activation `x * sigmoid(x)` to the input tensor."""
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Applies the Hardswish activation function to the input tensor `x`."""

    @staticmethod
    def forward(x):
        """Apply the Hardswish activation, using an export-friendly formulation for TorchScript, CoreML, and ONNX."""
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX


class Mish(nn.Module):
    """Applies the Mish activation function to improve model performance; see https://github.com/digantamisra98/Mish."""

    @staticmethod
    def forward(x):
        """Apply the Mish activation `x * tanh(softplus(x))` to the input tensor."""
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    """Applies the memory-efficient Mish activation function for improved model performance and reduced memory usage."""

    class F(torch.autograd.Function):
        """Memory-efficient implementation of the Mish activation function for enhanced model performance."""

        @staticmethod
        def forward(ctx, x):
            """Compute the Mish activation forward pass and save the input for the backward pass."""
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            """Compute the gradient of the Mish activation with respect to the saved input."""
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        """Apply the memory-efficient Mish activation to the input tensor."""
        return self.F.apply(x)


class FReLU(nn.Module):
    """Implements the FReLU activation, combining ReLU and convolution from https://arxiv.org/abs/2007.11824."""

    def __init__(self, c1, k=3):  # ch_in, kernel
        """Initialize FReLU with a depthwise conv and batch norm parameterized by channel count `c1` and kernel `k`."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """Apply FReLU, returning the elementwise max of the input and its batch-normed depthwise convolution."""
        return torch.max(x, self.bn(self.conv(x)))


class AconC(nn.Module):
    r"""ACON activation (activate or not) AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable
    parameter according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1):
        """Initialize AconC with learnable per-channel parameters p1, p2, and beta for `c1` input channels."""
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        """Apply the AconC activation with a fixed learnable beta to the input tensor."""
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    r"""ACON activation (activate or not) MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated
    by a small network according to "Activate or Not: Learning Customized
    Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1, k=1, s=1, r=16):  # ch_in, kernel, stride, r
        """Initialize MetaAconC with a small bottleneck network (kernel `k`, stride `s`, reduction `r`) that generates
        beta.
        """
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        """Apply the MetaAconC activation, generating beta from the input's spatial average via the bottleneck network."""
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x
