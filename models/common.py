# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    Automatically calculates the required padding for convolutional layers to maintain the same spatial dimensions,
    and optionally adjusts for dilation.
    
    Args:
      k (int | list[int]): The size of the convolutional kernel. Can be a single integer or a list of integers.
      p (int | list[int] | None, optional): The padding value(s). Can be a single integer, a list of integers, or 
          None to calculate the padding automatically. Defaults to None.
      d (int, optional): The dilation rate. Defaults to 1.
    
    Returns:
      int | list[int]: The calculated padding value(s). Returns a single integer if `k` is an integer, or a list 
          of integers if `k` is a list.
    
    Notes:
      - Padding is calculated to ensure the output dimensions are the same as the input dimensions.
      - Adjusts the kernel size internally when dilation is greater than 1 to compute the effective kernel size.
    
    Example:
      ```python
      k = 3
      p = autopad(k)  # Returns 1
      p = autopad(k, d=2)  # Returns 2 to account for dilation
      k = [3, 5]
      p = autopad(k)  # Returns [1, 2]
      ```
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initializes a standard Conv2D layer with batch normalization and optional activation.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple[int, int], optional): Kernel size. Default is 1.
            s (int | tuple[int, int], optional): Stride size. Default is 1.
            p (int | tuple[int, int] | None, optional): Padding. Default is None, which auto-calculates padding.
            g (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
            d (int | tuple[int, int], optional): Dilation rate. Default is 1.
            act (bool | nn.Module, optional): If True, uses the default activation function (SiLU). If a nn.Module is
                provided, uses that as the activation function. Default is True.
        
        Returns:
            None: This is an initializer, so it does not return anything.
        
        Notes:
            This class constructs a conv-bn-act block, which is a common building block for convolutional neural networks.
            The autopad function is used to automatically calculate the required padding to maintain the output dimensions
            given the kernel size and dilation factor.
        
        Example:
            ```python
            # Example of constructing a Conv2D layer with specific parameters:
            conv_layer = Conv(c1=32, c2=64, k=3, s=2, p=1, g=1, d=1, act=True)
            ```
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Applies convolution, batch normalization, and activation to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor with shape [N, C_in, H, W], where:
                              N - batch size
                              C_in - number of input channels
                              H - height of the input feature map
                              W - width of the input feature map
        
        Returns:
            torch.Tensor: Output tensor with shape [N, C_out, H_out, W_out], where:
                          N - batch size
                          C_out - number of output channels
                          H_out - height of the output feature map
                          W_out - width of the output feature map
        
        Example:
            ```python
            import torch
            from ultralytics import Conv
            
            # Define a Conv layer
            conv_layer = Conv(c1=3, c2=16, k=3, s=1)
            
            # Create a random tensor with shape [8, 3, 64, 64]
            x = torch.randn(8, 3, 64, 64)
            
            # Apply the Conv layer
            output = conv_layer(x)
            print(output.shape)  # Expected output shape: torch.Size([8, 16, 64, 64])
            ```
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Performs convolution followed by activation on the input tensor `x` using a fused approach.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C_in, H, W].
        
        Returns:
            torch.Tensor: Output tensor after applying convolution and activation, of shape [N, C_out, H_out, W_out].
        
        Note:
            Fusing the batch normalization layer into the convolution layer can improve inference speed by reducing the 
            number of operations.
        
        Example:
            ```python
            import torch
            from ultralytics import Conv
        
            model = Conv(3, 16, k=3, s=1)
            input_tensor = torch.randn(1, 3, 64, 64)  # [N, C_in, H, W]
            output_tensor = model.forward_fuse(input_tensor)
            print(output_tensor.shape)  # Should be [N, C_out, H_out, W_out]
            ```
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """
        Initializes a Depth-wise Convolution layer with batch normalization and optional activation.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple[int, int], optional): Kernel size for the convolution. Defaults to 1.
            s (int | tuple[int, int], optional): Stride of the convolution. Defaults to 1.
            d (int, optional): Dilation rate for dilated convolution. Defaults to 1.
            act (bool | nn.Module, optional): Activation function. Defaults to True. If True, uses SiLU activation; if 
                False, no activation is applied; if an nn.Module, applies the given activation function.
        
        Returns:
            None
        
        Example:
            ```python
            # Example usage of DWConv
            import torch
            from ultralytics import DWConv
        
            x = torch.randn(1, 32, 64, 64)  # input tensor with shape (batch_size, channels, height, width)
            dwconv = DWConv(c1=32, c2=64, k=3, s=1, d=1, act=True)
            y = dwconv(x)  # output tensor
            print(y.shape)  # Output tensor shape will be (1, 64, 62, 62) depending on padding settings
            ```
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """
        Initializes a depth-wise or transpose convolution layer with specified in/out channels, kernel size,
        stride, and padding.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size. Default is 1.
            s (int): Stride value. Default is 1.
            p1 (int): Padding added to all four sides of the input. Default is 0.
            p2 (int): Additional padding applied to the output. Default is 0.
        
        Returns:
            None
        
        Examples:
            ```python
            from ultralytics.modules.common import DWConvTranspose2d
            dwconv_transpose = DWConvTranspose2d(c1=32, c2=64, k=3, s=2, p1=1, p2=1)
            ```
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        """
        Initializes a Transformer layer as per https://arxiv.org/abs/2010.11929, sans LayerNorm, with specified embedding 
        dimension and number of heads.
        
        Args:
            c (int): Embedding dimension of the transformer, representing the number of expected features in the input.
            num_heads (int): Number of attention heads in the multihead attention mechanism.
        
        Returns:
            None
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """
        Performs a forward pass through the Transformer layer with multi-head attention and residual connections.
        
        Args:
        x (torch.Tensor): Input tensor of shape [batch, seq_len, features], where 'batch' is the number of samples, 
        'seq_len' is the sequence length, and 'features' is the number of features per sequence element.
        
        Returns:
        torch.Tensor: Output tensor of shape [batch, seq_len, features] after applying attention and feed-forward 
        transformations.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        """
        Initializes a Transformer block consisting of optional convolution, linear, and multiple transformer layers.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            num_heads (int): Number of attention heads in each Transformer layer.
            num_layers (int): Number of Transformer layers to stack within the block.
        
        Returns:
            None
        
        Examples:
            ```python
            transformer_block = TransformerBlock(c1=256, c2=512, num_heads=8, num_layers=6)
            ```
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """
        Forward pass for the Transformer Block, applying optional convolution and sequential transformer layers to 
        input tensor `x`.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, c1, height, width), where c1 is the number of 
                              channels.
        
        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, c2, height, width), where c2 is the number 
                          of output channels.
        
        Notes:
            For more information, refer to the paper: https://arxiv.org/abs/2010.11929. 
        
        Example:
            ```python
            transformer_block = TransformerBlock(c1=64, c2=128, num_heads=4, num_layers=2)
            input_tensor = torch.rand(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
            output_tensor = transformer_block(input_tensor)
            print(output_tensor.shape)  # torch.Size([8, 128, 32, 32])
            ```
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        """
        Initializes a standard bottleneck layer with optional shortcut connection. 
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to use shortcut connection. Default is True.
            g (int): Number of groups for group convolution. Default is 1.
            e (float): Expansion factor to determine hidden channels. Default is 0.5.
        
        Returns:
            None
        
        Note:
            This module is designed to streamline the creation of bottleneck layers for convolutional neural networks, typically
            used in deep learning models for feature extraction.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Executes a forward pass of the bottleneck layer, performing convolution, batch normalization, and optional
        shortcut addition.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C_in, H, W], where N is the batch size, C_in is the number of input
                channels, H and W are the height and width of the input feature map respectively.
        
        Returns:
            torch.Tensor: Output tensor of shape [N, C_out, H, W], where C_out is the number of output channels, after
            applying bottleneck transformation and optional shortcut connection.
        
        Notes:
            - If `shortcut` is enabled and input channels are equal to output channels (c1 == c2), the input tensor `x` is
              added to the output of the convolutional layers to form the final output.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """
        Initializes a Cross Stage Partial (CSP) Bottleneck layer for enhanced feature extraction, with configurable input/output
        channels, number of bottleneck layers, group convolution, and expansion factor.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers to include. Default is 1.
            shortcut (bool): Whether to add a shortcut connection. Default is True.
            g (int): Number of groups to use for convolutions. Default is 1.
            e (float): Expansion ratio for hidden channels. Default is 0.5.
        
        Returns:
            None
        
        Notes:
            This module implements the CSPNet architecture which helps to enrich the gradient combination with feature fusion,
            facilitating the feature propagation across the network while reducing the computation cost. More details at
            https://github.com/WongKinYiu/CrossStagePartialNetworks.
        
        Examples:
            ```python
            from ultralytics.models.common import BottleneckCSP
        
            # Example of creating a CSP bottleneck with 64 input channels, 128 output channels:
            bottleneck = BottleneckCSP(c1=64, c2=128, n=3, shortcut=False, g=2, e=0.75)
        
            # Forward pass using the created bottleneck
            input_tensor = torch.randn(1, 64, 128, 128)
            output_tensor = bottleneck(input_tensor)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Processes input tensor through the CSP Bottleneck, combining outputs through convolution, activation, and 
        normalization layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C_in, H, W].
        
        Returns:
            torch.Tensor: Output tensor of shape [N, C_out, H, W] after processing through the BottleneckCSP layers.
        
        Notes:
            The CSP Bottleneck is designed to enhance feature gradients, reduce model size, and improve computation time by
            leveraging Cross-Stage Partial Networks. For more details, refer to 
            https://github.com/WongKinYiu/CrossStagePartialNetworks.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes a Cross Convolution Downsample module by combining 1D and 2D convolutions, with optional shortcuts.
        
        Args:
            c1 (int): The number of input channels.
            c2 (int): The number of output channels.
            k (int): The size of the convolution kernel. Defaults to 3.
            s (int): The convolution stride. Defaults to 1.
            g (int): Number of blocked connections from input channels to output channels. Defaults to 1.
            e (float): Expansion ratio for the hidden channels. Defaults to 1.0.
            shortcut (bool): Whether to include a shortcut connection if the input and output channels match. Defaults to False.
        
        Returns:
            None
        
        Example:
            ```python
            import torch
            from ultralytics import CrossConv
        
            x = torch.randn(1, 32, 64, 64)  # input tensor with shape (batch_size, channels, height, width)
            model = CrossConv(32, 64, k=3, s=1, g=1, e=1.0, shortcut=True)
            output = model(x)
            print(output.shape)  # should be (1, 64, 64, 64)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Performs forward pass using sequential 1D and 2D convolutions with optional shortcut addition.
        
        Args:
            x (torch.Tensor): Input tensor with shape [N, C_in, H, W].
        
        Returns:
            torch.Tensor: Output tensor after applying cross convolutions, with shape [N, C_out, H_out, W_out].
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """
        Initializes a CSP bottleneck structure with 3 convolutional layers, providing options for shortcuts, group
        convolutions, and an expansion factor.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck layers. Defaults to 1.
            shortcut (bool): Whether to add a shortcut connection from input to output. Defaults to True.
            g (int): Number of groups for the group convolution. Defaults to 1.
            e (float): Expansion factor for the hidden channels. Defaults to 0.5.
        
        Returns:
            None
        
        Notes:
            This module is part of the YOLOv3 architecture and can be used to create complex feature extraction layers.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Processes the input tensor `x` through sequential convolutions and bottlenecks, combining intermediate results 
        for feature extraction.
        
        Args:
            x (torch.Tensor): Input tensor with shape [N, C, H, W], where N is the batch size, C is the number of channels, 
                H is the height, and W is the width.
        
        Returns:
            torch.Tensor: Output tensor with processed features, maintaining the input shape [N, C_out, H, W].
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes a C3x module with cross-convolutions, extending the C3 module with customizable parameters.
        
        Args:
          c1 (int): Input number of channels.
          c2 (int): Output number of channels.
          n (int, optional): Number of Bottleneck layers. Defaults to 1.
          shortcut (bool, optional): Whether to add shortcut connections. Defaults to True.
          g (int, optional): Number of convolution groups. Defaults to 1.
          e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
        
        Returns:
          None
        
        Notes:
          - This class builds upon the CSP Bottleneck architecture with an added cross-convolution mechanism for enhanced
            feature extraction.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes a CSP Bottleneck module with an added TransformerBlock for enhanced feature extraction and attention.
        
        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            n (int): Number of Bottleneck layers (default is 1).
            shortcut (bool): Whether to use shortcut connections (default is True).
            g (int): Number of groups for group convolution (default is 1).
            e (float): Expansion factor for hidden channels (default is 0.5).
        
        Returns:
            None
        
        Notes:
            This module extends the typical CSP Bottleneck by incorporating a transformer block, allowing for adaptive 
            feature representation leveraging both convolutional and transformer-based approaches. It acts as an advanced 
            network building block for scenarios requiring both efficient local feature extraction and long-range 
            dependencies in vision tasks.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes C3SPP module, extending C3 with Spatial Pyramid Pooling for enhanced feature extraction.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int], optional): Tuple of kernel sizes for max pooling in the Spatial Pyramid Pooling layer. Defaults to (5, 9, 13).
            n (int, optional): Number of Bottleneck layers to use. Defaults to 1.
            shortcut (bool, optional): Whether to add shortcut connections. Defaults to True.
            g (int, optional): Number of groups for grouped convolution. Defaults to 1.
            e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
        
        Returns:
            None
        
        Notes:
            This module incorporates Spatial Pyramid Pooling to capture multi-scale context by applying max pooling operations
            with different kernel sizes.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initializes C3Ghost module with Ghost Bottlenecks for efficient feature extraction.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int, optional): Number of bottlenecks to include. Default is 1.
            shortcut (bool, optional): Whether to include a shortcut to input connections. Default is True.
            g (int, optional): Number of groups for group convolutions. Default is 1.
            e (float, optional): Expansion factor for hidden channels. Default is 0.5.
        
        Returns:
            None
        
        Example:
            ```python
            model = C3Ghost(c1=64, c2=128, n=2, shortcut=True, g=1, e=0.5)
            x = torch.randn(1, 64, 256, 256)
            output = model(x)
            ```
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initializes SPP layer with specified input/output channels and kernel sizes for spatial pyramid pooling.
        
        Args:
          c1 (int): Number of input channels.
          c2 (int): Number of output channels.
          k (tuple[int]): Tuple of kernel sizes for pooling layers. Default is (5, 9, 13).
        
        Returns:
          None
        
        Notes:
          Implements the Spatial Pyramid Pooling (SPP) technique as described in the paper: https://arxiv.org/abs/1406.4729
        
        Example:
          ```python
          spp_layer = SPP(c1=64, c2=128, k=(5, 9, 13))
          ```
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Forward pass through the Spatial Pyramid Pooling (SPP) layer, applying multiple max pooling operations.
        
        Args:
          x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        
        Returns:
          torch.Tensor: Output tensor after applying convolution and spatial pyramid pooling, with concatenated features
            from multiple pooling layers. The shape will be (N, C_out, H_out, W_out).
        
        Example:
          ```python
          spp_layer = SPP(c1=512, c2=1024, k=(5, 9, 13))
          input_tensor = torch.randn(1, 512, 32, 32)
          output_tensor = spp_layer(input_tensor)
          ```
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv3 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        """
        Initializes the Spatial Pyramid Pooling - Fast (SPPF) layer with specified input and output channels and kernel size.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for max pooling. Defaults to 5.
          
        Returns:
            None
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Performs a forward pass through the SPPF layer, combining convolutions and max pooling operations.
        
        Args:
          x (torch.Tensor): Input tensor of shape [N, C, H, W], where N is the batch size, C is the number
                            of channels, H is the height, and W is the width.
        
        Returns:
          torch.Tensor: Output tensor of shape [N, C_out, H_out, W_out], where C_out and H/W_out depend on
                        the configurations of the convolutions and max pooling layers.
        
        Example usage:
          ```python
          sppf = SPPF(256, 512, k=5)
          output = sppf(input_tensor)
          ```
        
        Notes:
          - The SPPF layer is designed for fast spatial pyramid pooling, commonly used in YOLOv3 models.
          - For more details, refer to the implementation of the SPP layer as described in https://arxiv.org/abs/1406.4729.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """
        Focus.__init__()
        
        Initializes Focus module to focus width and height information into channel space with configurable convolution parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple, optional): Kernel size for the convolutional layer. Defaults to 1.
            s (int | tuple, optional): Stride for the convolutional layer. Defaults to 1.
            p (int | tuple, optional): Padding for the convolutional layer. Defaults to None, which enables autopadding.
            g (int, optional): Number of groups for the convolution. Defaults to 1.
            act (bool | nn.Module, optional): Activation function to apply. If True, uses the default activation 
                (nn.SiLU). If a module is provided, uses that module as the activation function.
                If False, uses nn.Identity(). Defaults to True.
        
        Returns:
            Focus instance.
            
        Notes:
            This module is used to reduce the spatial dimensions (width and height) while increasing the depth (number 
            of channels) for subsequent layers. Suitable for feature extraction where spatial locality is important.
            
        Examples:
            ```python
            focus_layer = Focus(c1=3, c2=32, k=3, s=1)
            outputs = focus_layer(inputs)
            ```
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        """
        Focuses spatial information from input tensor into channel space and applies a convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W].
        
        Returns:
            torch.Tensor: Output tensor with focused spatial information and applied convolution, having shape [N, C_out, H_out, W_out].
        
        Examples:
            ```python
            focus_layer = Focus(c1=3, c2=64)
            input_tensor = torch.rand(1, 3, 640, 640)
            output_tensor = focus_layer(input_tensor)
            ```
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        """
        Initialize a GhostConv layer with input/output channels, kernel size, stride, groups, and optional activation.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size for the convolution. Defaults to 1.
            s (int, optional): Stride for the convolution. Defaults to 1.
            g (int, optional): Number of groups for the convolution. Defaults to 1.
            act (bool | nn.Module, optional): Activation function to use after convolution. Defaults to True, which 
                applies the default activation (SiLU). If set to False, no activation will be used. Custom nn.Module can 
                also be provided.
            
        Returns:
            None
        
        Notes:
            This module implements the Ghost Convolution as described in https://github.com/huawei-noah/ghostnet.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Executes forward pass, applying convolutions and concatenating results from GhostConv module.
        
        Args:
            x (torch.Tensor): Input tensor with shape [N, C_in, H, W].
        
        Returns:
            torch.Tensor: Output tensor of shape [N, C_out, H_out, W_out], resulting from the application of sequential convolutions.
        
        Notes:
            - Implements Ghost Convolution as described in https://github.com/huawei-noah/ghostnet.
            - Efficiently decomposes standard convolutions into smaller, computationally cheaper operations with minimal loss in representation power.
        
        Example:
            ```python
            import torch
            from ultralytics.nn.modules import GhostConv
        
            input_tensor = torch.randn(1, 64, 128, 128)  # Batch size of 1, 64 input channels, 128x128 spatial dimensions
            ghost_conv = GhostConv(64, 128)  # Initialize GhostConv module
            output_tensor = ghost_conv(input_tensor)  # Execute forward pass
            print(output_tensor.shape)  # Expected shape [1, 128, 128, 128]
            ```
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        """
        Initializes the GhostBottleneck module with specified input/output channels, kernel size, and stride.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depth-wise convolution. Default is 3.
            s (int): Stride for depth-wise convolution. Default is 1.
        
        Returns:
            None
        
        Note:
            For more information, please refer to https://github.com/huawei-noah/ghostnet
        
        Example:
            ```python
            from ultralytics.modules import GhostBottleneck
        
            # Initialize GhostBottleneck with input channels=32, output channels=64, kernel size=3, and stride=1
            ghost_bottleneck = GhostBottleneck(c1=32, c2=64, k=3, s=1)
            input_tensor = torch.randn(1, 32, 128, 128)
            output_tensor = ghost_bottleneck(input_tensor)
            ```
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """
        ```python
        Performs a forward pass through the GhostBottleneck network, combining convolution and shortcut paths.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C_in, H, W], where `N` is the batch size, `C_in` is the number
                of input channels, `H` and `W` are the height and width of the input feature map respectively.
        
        Returns:
            torch.Tensor: Output tensor of shape [N, C_out, H_out, W_out], where `C_out` is the number of output channels,
                `H_out` and `W_out` are the height and width of the output feature map respectively, 
                resulted from the combination of convolution and shortcut paths.
        ```
        """
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        """
        Initializes the Contract module.
        
        The Contract module refines the input dimensions by reducing width and height while increasing the channel depth,
        which is particularly useful for downsampling. By default, the module uses a gain factor of 2 to effectively reduce
        the spatial dimensions and increase the number of channels, enabling more compact and efficient representation
        of the input data.
        
        Args:
            gain (int): The factor by which to contract the width and height dimensions. Default is 2.
        
        Returns:
            None: This is an initializer method and does not return a value.
        
        Notes:
            This module is useful in scenarios where spatial downsampling is necessary while retaining critical
            information in the channel dimension, making it beneficial for use in convolutional neural networks.
            
        Example:
            ```python
            import torch
            from ultralytics.models.common import Contract
        
            # Initialize the Contract module with a gain of 2
            contract_layer = Contract(gain=2)
        
            # Create a dummy input tensor of shape [1, 64, 80, 80]
            x = torch.randn(1, 64, 80, 80)
        
            # Apply the Contract layer
            output = contract_layer(x)
        
            # The output will have shape [1, 256, 40, 40]
            print(output.shape)
            ```
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Contract.forward
            """Contracts the spatial dimensions (height and width) of the input tensor, expanding the channels dimension.
        
            The function reshapes the input tensor by contracting its spatial dimensions (height and width), effectively 
            increasing the number of channels. The contraction factor is defined by the `gain` attribute specified during 
            initialization.
        
            Args:
                x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        
            Returns:
                torch.Tensor: Output tensor with contracted spatial dimensions and expanded channels dimension. The shape 
                              will be (batch_size, channels * gain^2, height / gain, width / gain).
        
            Example:
                ```python
                contract = Contract(gain=2)
                output = contract.forward(input_tensor)
                # If input_tensor shape is (1, 64, 80, 80), the output shape will be (1, 256, 40, 40).
                ```
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        """
        Increases spatial dimensions by a factor of `gain`, reducing the number of channels accordingly, e.g., from 
        (1,64,80,80) to (1,16,160,160).
        
        Args:
            gain (int, optional): Factor by which to expand spatial dimensions, with a corresponding reduction in channels.
                Default is 2.
        
        Returns:
            None (NoneType)
        
        Note:
            The `gain` factor should be compatible with the input dimensions to avoid indivisible operations.
        
        Example:
            ```python
            from ultralytics import Expand
            import torch
        
            x = torch.randn(1, 64, 80, 80)
            expand = Expand(gain=2)
            output = expand(x)
            print(output.shape)  # Should output: torch.Size([1, 16, 160, 160])
            ```
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """
        Expands the spatial dimensions of the input tensor by a given factor while reducing the number of channels.
        
        Args:
            x (torch.Tensor): Input tensor of shape `(B, C, H, W)` where `B` is the batch size, `C` is the number of 
                              channels, `H` is the height, and `W` is the width.
                        
        Returns:
            torch.Tensor: Output tensor with expanded spatial dimensions and reduced number of channels, reshaped to 
                          `(B, C / gain^2, H * gain, W * gain)`.
        
        Example:
            ```python
            expand = Expand(gain=2)
            input_tensor = torch.randn(1, 64, 80, 80)
            output_tensor = expand(input_tensor)
            print(output_tensor.shape)  # torch.Size([1, 16, 160, 160])
            ```
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        """
        Initializes a module to concatenate tensors along a specified dimension.
        
        Args:
            dimension (int): The dimension along which to concatenate the tensors. Default is 1.
        
        Returns:
            None (returns the instance of the class itself)
        
        Examples:
            ```python
            concat = Concat(dimension=1)
            ```
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenates a list of tensors along a specified dimension.
        
        Args:
            x (list of torch.Tensor): A list of tensors to concatenate along the given dimension.
        
        Returns:
            torch.Tensor: A single tensor obtained by concatenating the tensors in `x` along the specified dimension.
        """
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv3 MultiBackend class for python inference on various backends
    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """
        Initializes a multi-backend detection model, supporting various deep learning frameworks and devices.
        
        Args:
            weights (str | list): Path to the model weights file(s). Default is "yolov5s.pt".
            device (torch.device): Device to run the model on, such as `torch.device('cpu')` or `torch.device('cuda:0')`.
                Default is torch.device('cpu').
            dnn (bool): Flag to use OpenCV DNN for ONNX models. Default is False.
            data (str | None): Path to the dataset configuration. Default is None.
            fp16 (bool): Flag to use Floating Point 16 precision. Default is False.
            fuse (bool): Flag to fuse model layers for optimization. Default is True.
        
        Returns:
            None
        
        Notes:
            Supported weight formats include:
              - PyTorch: *.pt
              - TorchScript: *.torchscript
              - ONNX Runtime: *.onnx
              - ONNX OpenCV DNN: *.onnx with --dnn flag
              - OpenVINO: *_openvino_model
              - CoreML: *.mlmodel
              - TensorRT: *.engine
              - TensorFlow SavedModel: *_saved_model
              - TensorFlow GraphDef: *.pb
              - TensorFlow Lite: *.tflite
              - TensorFlow Edge TPU: *_edgetpu.tflite
              - PaddlePaddle: *_paddle_model
        
        Example:
            ```python
            model = DetectMultiBackend(weights="yolov5s.pt", device=torch.device('cuda:0'))
            ```
        """
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wraps a frozen TensorFlow GraphDef for inference, returning a pruned function for specified inputs
                and outputs.
                """
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                """Extracts and sorts non-input (output) tensor names from a TensorFlow GraphDef, excluding 'NoOp'
                prefixed tensors.
                """
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv3 TF.js inference is not supported")
        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi

            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix(".pdiparams")
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """
        Performs YOLOv3 inference on an input image tensor, optionally with augmentation and visualization.
        
        Args:
            im (torch.Tensor): Input image tensor with shape [batch_size, channels, height, width].
            augment (bool, optional): If True, applies image augmentations during inference. Defaults to False.
            visualize (bool, optional): If True, enables visualization of intermediate layers during inference. 
                                        Defaults to False.
        
        Returns:
            torch.Tensor | list[torch.Tensor]: Inference results from the model. If the results are from multiple 
                                               outputs, returns a list of tensors.
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Converts a Numpy array to a PyTorch tensor on the specified device, or returns input if not a Numpy array.
        
        Args:
            x (numpy.ndarray): Input array to be converted.
        
        Returns:
            torch.Tensor | Any: PyTorch tensor if input was a Numpy array, otherwise returns the input unchanged.
        
        Examples:
            ```python
            import numpy as np
            array = np.array([1, 2, 3])
            tensor = model.from_numpy(array)
            ```
        """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warms up the model by running inference once with a dummy input of specified shape.
        
        Args:
            imgsz (tuple[int]): Shape of the dummy input tensor to warm up the model, default is (1, 3, 640, 640).
        
        Returns:
            None
        
        Notes:
            The warmup process runs one or two inferences with random data depending on the model type. This is particularly
            useful for optimizing GPU memory allocations and other computational overheads, leading to a more stable
            and predictable runtime behavior during actual inference. The function executes if the model is supported by
            PyTorch, TorchScript, ONNX, TensorRT, TensorFlow (SavedModel and GraphDef), or Triton, and is not targeting a CPU
            device, unless using Triton.
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines model type from a file path or URL.
        
        Args:
            p (str): The path to the model file, either local or a URL.
        
        Returns:
            tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
            A tuple of boolean values indicating the model type:
            - PyTorch (.pt)
            - TorchScript (.torchscript)
            - ONNX (.onnx)
            - OpenVINO (.xml)
            - TensorRT (.engine)
            - CoreML (.mlmodel)
            - TensorFlow SavedModel (_saved_model)
            - TensorFlow GraphDef (.pb)
            - TensorFlow Lite (.tflite)
            - TensorFlow Edge TPU (.edgetpu.tflite)
            - TF.js
            - PaddlePaddle (_paddle_model)
            - Triton Inference Server
        
        Note:
            The function checks the suffix of the file name against the supported export formats and determines the
            model type based on the suffix. For URLs, it checks if the URL might be associated with a Triton inference
            server.
            
        Examples:
            ```python
            model_path = "path/to/model.onnx"
            model_types = _model_type(model_path)
            print(model_types)  # (False, False, True, False, False, False, False, False, False, False, False, False, False)
            ```
        
            ```python
            model_url = "http://localhost:8000/v2/models/yolov5/versions/1"
            is_triton = _model_type(model_url)
            print(is_triton)  # (False, False, False, False, False, False, False, False, False, False, False, False, True)
            ```
        """
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """
        Loads metadata from a YAML file and returns specific keys.
        
        Args:
            f (Path): The path to the YAML file from which metadata is to be loaded.
        
        Returns:
            (int, list[str]): A tuple containing:
                - stride (int): The stride value from the metadata.
                - names (list[str]): List of class names from the metadata.
            
        Example:
            ```python
            stride, names = self._load_metadata(Path("./data/meta.yaml"))
            # stride: 32, names: ['person', 'bicycle', 'car', ...]
            ```
        
        Notes:
            This method expects the YAML file to have 'stride' and 'names' keys. If the file does not exist,
            it will return `None`.
        """
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv3 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        """
        Initializes the AutoShape module, setting up the model for robust inference from various input data formats 
        including cv2, numpy, PIL, and torch, while enabling preprocessing, inference, and non-max suppression.
        
        Args:
            model (torch.nn.Module): The model to be wrapped for robust input handling and enhanced inference.
            verbose (bool): If True, logs additional information during initialization. Default is True.
        
        Returns:
            None: This is an initializer method and does not return a value.
        
        Notes:
            This class is designed to facilitate seamless inference across different data representations and ensure that
            preprocessing, inference, and post-processing (like non-max suppression) are all handled efficiently within one
            module.
        
        Example:
        ```python
        # Assuming `model` is a loaded YOLOv3 model
        autoshape_model = AutoShape(model)
        ```
        """
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        """
        Applies a function to model tensors, adjusting strides and grids.
        
        Args:
          fn (Callable): The function to apply to each tensor in the model.
        
        Returns:
          AutoShape: The current instance with the function applied to its tensors.
        
        Notes:
          The function fn is typically used for operations like casting the tensors to a different device or type.
        ```
        """
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        """
        Performs inference on various input sources with optional augmentation and profiling.
        
        Args:
         ims (str | Path | np.ndarray | torch.Tensor | list): Input images, can be a file path, URI, numpy array, torch 
          tensor, or a list of such inputs.
         size (int | tuple[int, int], optional): Inference size, either an integer (for square resizing) or a tuple 
          (height, width). Defaults to 640.
         augment (bool, optional): If True, applies augmentations during inference. Defaults to False.
         profile (bool, optional): If True, profiles the inference time for different phases. Defaults to False.
        
        Returns:
         list: List of inference results represented as a list of dictionaries, each containing:
          - boxes (np.ndarray): Detected bounding boxes.
          - confidences (np.ndarray): Confidence scores for each bounding box.
          - class_ids (np.ndarray): Class IDs for each bounding box.
        
        Examples:
        ```python
        from PIL import Image
        import torch
        
        # Initialize the model (assuming `model` is a pre-loaded DetectMultiBackend instance)
        autoshape = AutoShape(model)
        
        # Example using a file path
        results = autoshape.forward('data/images/zidane.jpg')
        print(results)
        
        # Example using a numpy array (OpenCV)
        img = cv2.imread('data/images/zidane.jpg')
        results = autoshape.forward(img)
        print(results)
        
        # Example using a PIL Image
        img = Image.open('data/images/zidane.jpg')
        results = autoshape.forward(img)
        print(results)
        
        # Example using a torch tensor
        img = torch.randn(1, 3, 640, 640)
        results = autoshape.forward(img)
        print(results)
        
        # Example using multiple images list (PIL)
        imgs = [Image.open('data/images/zidane.jpg'), Image.open('data/images/bus.jpg')]
        results = autoshape.forward(imgs)
        print(results)
         ```
        """
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f"image{i}"  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
                files.append(Path(f).with_suffix(".jpg").name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(
                    y if self.dmb else y[0],
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic,
                    self.multi_label,
                    max_det=self.max_det,
                )  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv3 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        """
        Initializes a Detections object representing detection results from YOLOv3 inference.
        
        Args:
            ims (list of np.ndarray): List of input images as numpy arrays.
            pred (list of torch.Tensor): List of prediction tensors, each containing bounding boxes, confidence scores, 
                and class labels.
            files (list of str | Path): List of filenames associated with the input images.
            times (tuple of float, optional): Tuple representing profiling times (in milliseconds) for different stages of 
                inference. Default is (0, 0, 0).
            names (list of str | None, optional): List of class names. Default is `None`.
            shape (tuple, optional): Shape of the output predictions. Default is `None`.
        
        Returns:
            None
        
        Notes:
            The `Detections` object contains several attributes, including:
            - `ims`: List of input images.
            - `pred`: List of prediction tensors with bounding boxes, scores, and class labels.
            - `xyxy`: Prediction tensors in `xyxy` format.
            - `xywh`: Prediction tensors in `xywh` format.
            - `xyxyn`: Normalized `xyxy` predictions.
            - `xywhn`: Normalized `xywh` predictions.
            - `files`: Filenames corresponding to input images.
            - `names`: Class names for the predictions.
            - `times`: Profiling times for inference stages.
            - `n`: Number of input images.
            - `t`: Timestamps in milliseconds, computed per image.
        
        Example:
            ```python
            detections = Detections(ims, pred, files)
            print(detections.xyxy)  # Access xyxy formatted predictions
            ```
        """
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        """
        Runs inference on input images and performs various post-processing steps such as displaying, saving, cropping, and 
        rendering annotated results.
        
        Args:
            pprint (bool): If True, pretty prints the detection results and profiling info. Default is False.
            show (bool): If True, displays the images with annotations. Default is False.
            save (bool): If True, saves the annotated images to disk. Default is False.
            crop (bool): If True, saves cropped regions of detected objects. Default is False.
            render (bool): If True, updates the `ims` attribute with rendered images for further processing. Default is False.
            labels (bool): If True, includes class labels in the annotations. Default is True.
            save_dir (Path): Directory path where outputs should be saved if `save` or `crop` is True. Default is Path("").
        
        Returns:
            str | None: Pretty printed output string if `pprint` is True, otherwise None.
        
        Notes:
            - The function assumes that attribute `self.pred` contains detected objects in the form [xyxy, confidence, class].
            - The function utilizes utility methods such as `Annotator` and `save_one_box` for drawing and saving operations.
        
        Example:
            ```python
            detections = Detections(ims, pred, files, times, names, shape)
            result_str = detections._run(pprint=True, show=True, save=True, crop=True, save_dir=Path("./outputs"))
            ```
        """
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(", ")
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))
                    im = annotator.im
            else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display

                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays detected objects on images with optional class labels.
        
        Args:
            labels (bool, optional): Whether to display labels for detected objects. Defaults to True.
        
        Returns:
            None
        
        Notes:
            This method attempts to show images using the default image viewer. In Jupyter environments, `IPython.display.display`
            is used for inline visualization. If not supported, an exception is raised. For additional functionality, refer to the
            Detections class documentation at https://github.com/ultralytics/ultralytics
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves the detection results to a specified directory, optionally with labels.
        
        Args:
            labels (bool): If True, labels are displayed on the saved images. Defaults to True.
            save_dir (str | Path): Directory where the results will be saved. Defaults to 'runs/detect/exp'.
            exist_ok (bool): If True, allows overwriting files if they already exist. Defaults to False.
        
        Returns:
            None
        
        Example:
            ```python
            detections = model(images)
            detections.save(labels=True, save_dir='runs/detect/experiment1', exist_ok=True)
            ```
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detected objects from all images, optionally saving them to a specified directory.
        
        Args:
            save (bool, optional): If True, the cropped images will be saved. Defaults to True.
            save_dir (str, optional): Directory to save cropped images. Defaults to "runs/detect/exp".
            exist_ok (bool, optional): If True, existing directories will be used without conflict. Defaults to False.
        
        Returns:
            list[dict]: List of dictionaries containing cropped image data, labels, and confidence scores.
        
        Examples:
            ```python
            detections.crop(save=True, save_dir='runs/detect/exp')
            ```
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """
        Renders the labeled detections on the images.
        
        Args:
            labels (bool): If True, include labels in the rendered image. Default is True.
        
        Returns:
            None
        
        Examples:
            ```python
            detections.render(labels=True)
            ```
        Notes:
            This method annotates the detections onto the images stored in `self.ims` and has no return value.
            It manipulates the internal state of the Detections object.
        """
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Returns a copy of detection results as pandas DataFrames for different bounding box formats.
        
        Returns:
            Detections: A `Detections` object containing pandas DataFrames of detection results with columns specifying bounding
            box coordinates (xmin, ymin, xmax, ymax for xyxy format; xcenter, ycenter, width, height for xywh format), confidence
            score, class id, and class name. The DataFrames are stored in attributes: 'xyxy', 'xyxyn', 'xywh', 'xywhn'.
        
        Example:
            ```python
            detections = model(imgs)  # Perform inference
            results = detections.pandas()
            print(results.xyxy[0])  # Print the DataFrame for the first image in xyxy format
            ```
            
        Note:
            The returned DataFrames have the same information as the detection tensors, but are formatted for easier analysis
            and manipulation using pandas.
        """
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        """
        Converts the Detections instance into a list of individual Detection objects.
        
        Returns:
            list[Detections]: A list of Detections objects, each corresponding to a single image and its detections.
        
        Example:
        ```python
        # Assuming 'detections' is an instance of Detections
        detection_list = detections.tolist()
        single_detection = detection_list[0]  # Access detections for the first image
        ```
        """
        r = range(self.n)  # iterable
        return [
            Detections(
                [self.ims[i]],
                [self.pred[i]],
                [self.files[i]],
                self.times,
                self.names,
                self.s,
            )
            for i in r
        ]

    def print(self):
        """
        Prints a descriptive summary of the current detection results to the LOGGER.
        
        Returns:
            None
        
        Examples:
            The following example demonstrates how to use this method:
        
            ```python
            detections.print()
            ```
        
        Notes:
            Ensure that the `LOGGER` is properly configured to capture the printed output.
        """
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        """
        Returns the number of detections in the current instance.
        
        Returns:
            int: The number of detections.
        
        Examples:
            ```python
            detections = Detections(...)
            num_detections = len(detections)
            ```
        """
        return self.n

    def __str__(self):  # override print(results)
        """
        Returns a string representation of the Detections object, summarizing detection results.
        
        Returns:
            str: Formatted string that includes the number of detections per image, class names, confidence scores, and
            profiling times.
        """
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """
        Returns a string representation for debugging, including class info and current object state.
        
        Returns:
            str: String representation of the Detections instance suitable for debugging.
        ```
        
        Notes:
        Attributes provide details about internal states such as predictions, images, class names, etc.
        Suitable for quick inspection of the current detection results in debugging scenarios.
        ```
        """
        return f"YOLOv3 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    # YOLOv3 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        """
        Initializes the Proto module for YOLOv3 segmentation, setting up convolutional layers and upsampling.
        
        Args:
            c1 (int): Number of input channels.
            c_ (int): Number of prototype channels. Default is 256.
            c2 (int): Number of output channels for masks. Default is 32.
        
        Returns:
            None
        
        Notes:
            This module is a part of the YOLOv3 architecture, specifically designed for segmentation tasks. It consists
            of convolutional layers followed by upsampling to generate high-resolution mask predictions.
        
        Examples:
            ```python
            model = Proto(c1=64, c_=128, c2=16)
            ```
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """
        Performs forward pass, upsampling, and applying convolutions for YOLOv3 segmentation.
        
        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, channels_in, height, width]`.
        
        Returns:
            torch.Tensor: Output tensor with segmentation masks, shape `[batch_size, num_masks, height_out, width_out]`.
        
        Example:
            ```python
            proto = Proto(512)
            output = proto(input_tensor)
            ```
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv3 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """
        Initializes a YOLOv3 classification head with convolution, pooling, and dropout layers for feature extraction and 
        classification.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels (classes).
            k (int): Kernel size of the convolution (default: 1).
            s (int): Stride of the convolution (default: 1).
            p (int | None): Padding for the convolution. If None, padding is auto-calculated (default: None).
            g (int): Number of groups for the convolution (default: 1).
            dropout_p (float): Probability of an element to be zeroed in the dropout layer (default: 0.0).
        
        Returns:
            nn.Module: A PyTorch module representing the YOLOv3 classification head.
        
        Examples:
            ```python
            from ultralytics.models.common import Classify
        
            model = Classify(c1=3, c2=1000, k=1, s=1, dropout_p=0.5)
            output = model(torch.randn(1, 3, 224, 224))  # forward pass with dummy input
            ```
        Notes:
            For more details on YOLOv3 classification, refer to the YOLOv3 architecture documentation at 
            https://github.com/ultralytics/yolov3.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """
        Performs a forward pass through the YOLOv3 classification head.
        
        Args:
            x (torch.Tensor | list[torch.Tensor]): Input tensor(s) of shape [N, C, H, W] or a list of such tensors.
        
        Returns:
            torch.Tensor: Output tensor of shape [N, num_classes], where `num_classes` is the number of output classes.
            
        Note:
            If the input `x` is a list of tensors, they are concatenated along the channel dimension before processing.
            
        Example:
            ```python
            # Assuming input batch of images represented as a torch tensor with shape [N, C, H, W]
            inputs = torch.randn(8, 3, 224, 224)  # batch of 8 RGB images of size 224x224
            classify_layer = Classify(3, 1000)  # for example, classifying into 1000 classes
            outputs = classify_layer(inputs)
            
            print(outputs.shape)  # Output will be [8, 1000]
            ```
        ```
        """
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
