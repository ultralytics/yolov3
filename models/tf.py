# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv3
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras

from models.common import (
    C3,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3x,
    Concat,
    Conv,
    CrossConv,
    DWConv,
    DWConvTranspose2d,
    Focus,
    autopad,
)
from models.experimental import MixConv2d, attempt_load
from models.yolo import Detect, Segment
from utils.activations import SiLU
from utils.general import LOGGER, make_divisible, print_args


class TFBN(keras.layers.Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):
        """
        Initializes TFBN with weights, wrapping TensorFlow's BatchNormalization layer with specific initializers.

        Args:
            w (torch.nn.BatchNorm2d): A PyTorch BatchNorm2d layer whose parameters are used to initialize the
                corresponding TensorFlow BatchNormalization layer.

        Returns:
            None

        Notes:
            The TFBN class is designed to bridge PyTorch BatchNorm2d layers with TensorFlow's BatchNormalization layers,
            allowing for seamless integration and transfer of parameters between the two frameworks. This is particularly
            useful when converting models from PyTorch to TensorFlow.

        Example:
            ```python
            import torch
            from models.common import TFBN

            # Create a PyTorch BatchNorm2d layer
            pytorch_bn = torch.nn.BatchNorm2d(num_features=64)

            # Initialize a corresponding TensorFlow BatchNormalization layer using TFBN
            tensorflow_bn = TFBN(pytorch_bn)
            ```
        """
        super().__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()),
            epsilon=w.eps,
        )

    def call(self, inputs):
        """
        Applies batch normalization on the input tensor using the initialized parameters.

        Args:
            inputs (tf.Tensor): Input tensor to which batch normalization is applied.

        Returns:
            tf.Tensor: The normalized tensor after applying batch normalization.
        """
        return self.bn(inputs)


class TFPad(keras.layers.Layer):
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):
        """
        Initializes a padding layer for spatial dimensions 1 and 2.

        Args:
            pad (int | tuple[int, int] | list[int, int]): The amount of padding to apply to the spatial dimensions.
                If an int is provided, the same padding is applied to both dimensions. If a tuple or list of two
                integers is provided, the corresponding padding is applied to each spatial dimension.

        Returns:
            None

        Notes:
            This layer adds zero-padding to the height and width dimensions of the input tensor, typically used in
            deep learning models to maintain spatial dimensions after convolution operations.

        Examples:
            ```python
            pad_layer = TFPad(2)  # Creates a padding of 2 on height and width dimensions.
            padded_output = pad_layer(input_tensor)
            ```
        """
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        """Call(inputs: tf.Tensor) -> tf.Tensor:"""
            Applies constant padding to inputs with `pad` specifying padding width in spatial dimensions 1 and 2.
        
            Args:
                inputs (tf.Tensor): Input tensor to be padded, typically of shape (batch, height, width, channels).
        
            Returns:
                tf.Tensor: Padded tensor with the same batch size and number of channels, but potentially different 
                spatial dimensions (height and width) due to padding.
        
            Examples:
                ```python
                import tensorflow as tf
                from models.tf import TFPad
        
                # Initialize padding layer. Here pad = (1, 2) means 1 pixel on height and 2 pixels on width.
                padding_layer = TFPad((1, 2))
        
                # Create a sample input tensor of shape (batch, height, width, channels) = (1, 5, 5, 1)
                input_tensor = tf.random.normal((1, 5, 5, 1))
        
                # Apply padding
                output_tensor = padding_layer.call(input_tensor)
        
                print(output_tensor.shape)  # Should print (1, 7, 9, 1) due to 1+1 and 2+2 padding in height and width
                ```
        """
        return tf.pad(inputs, self.pad, mode="constant", constant_values=0)


class TFConv(keras.layers.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initializes a standard convolutional layer with customizable parameters for filters, kernel size, stride, padding, groups, and activation.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size. Defaults to 1.
            s (int, optional): Stride size. Defaults to 1.
            p (int | None, optional): Padding size. If None, 'SAME' padding is used for stride 1 and 'VALID' for others.
                Defaults to None.
            g (int, optional): Number of groups for convolution. Currently must be 1 as TensorFlow v2.2 Conv2D does not support
                'groups'. Defaults to 1.
            act (bool, optional): Whether to use activation function after the convolution. Defaults to True.
            w (torch.nn.Module, optional): Weights from a PyTorch model to initialize the convolutional layer. 
        
        Notes:
            TensorFlow convolution padding behavior differs from PyTorch (refer to 
            https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch).
        
        Returns:
            None
        
        Examples:
            ```python
            conv_layer = TFConv(c1=32, c2=64, k=3, s=1, p=1, g=1, act=True, w=some_weights)
            ```
        """
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        conv = keras.layers.Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding="SAME" if s == 1 else "VALID",
            use_bias=not hasattr(w, "bn"),
            kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()),
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def call(self, inputs):
        """
        Executes the convolution, batch normalization, and activation on the input data.
        
        Args:
            inputs (tf.Tensor): A 4D TensorFlow tensor with shape (batch_size, height, width, channels).
        
        Returns:
            tf.Tensor: A 4D tensor transformed by convolution, batch normalization, and activation, with the same
            shape as `inputs`.
        
        Notes:
            This layer performs convolution with the specified number of filters, kernel size, stride,
            padding, and groups. Batch normalization and activation functions are applied if specified.
            TensorFlow's padding behavior is adjusted to ensure consistency with PyTorch's conventions.
        """
        return self.act(self.bn(self.conv(inputs)))


class TFDWConv(keras.layers.Layer):
    # Depthwise convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):
        """
        Initializes a depthwise convolutional layer with customizable kernel size, stride, padding, activation, and weights.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels. Must be a multiple of `c1`.
            k (int, optional): Kernel size for the depthwise convolution. Default is 1.
            s (int, optional): Stride size for the depthwise convolution. Default is 1.
            p (int | tuple[int, int], optional): Padding size. Default is None.
            act (bool, optional): Whether to apply activation after the convolution. Default is True.
            w (torch.nn.Module, optional): PyTorch weights to initialize the depthwise convolution. Default is None.
        
        Returns:
            None: This function initializes the depthwise convolutional layer without returning a value.
        
        Notes:
            TensorFlow convolution padding is handled differently than in PyTorch. 
            See: https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorc
        """
        super().__init__()
        assert c2 % c1 == 0, f"TFDWConv() output={c2} must be a multiple of input={c1} channels"
        conv = keras.layers.DepthwiseConv2D(
            kernel_size=k,
            depth_multiplier=c2 // c1,
            strides=s,
            padding="SAME" if s == 1 else "VALID",
            use_bias=not hasattr(w, "bn"),
            depthwise_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()),
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def call(self, inputs):
        """
        Applies depthwise convolution, batch normalization, and activation to the input tensor.
        
        Args:
            inputs (tf.Tensor): Input tensor to the depthwise convolution layer.
        
        Returns:
            tf.Tensor: Processed tensor with depthwise convolution, optional batch normalization, and activation applied.
        """
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(keras.layers.Layer):
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        """
        Initializes TFDWConvTranspose2d with ch_in, k, and padding parameters, setting up the Conv2DTranspose layers.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels. Must be equal to `c1`.
            k (int): Kernel size. Must be set to 4.
            s (int): Stride. Defines the stride length for the convolution.
            p1 (int): First padding parameter. Must be set to 1.
            p2 (int): Second padding parameter. Specifies additional padding.
            w (torch.nn.Module): PyTorch model weights to initialize TensorFlow layer weights.
        
        Returns:
            None
        """
        super().__init__()
        assert c1 == c2, f"TFDWConv() output={c2} must be equal to input={c1} channels"
        assert k == 4 and p1 == 1, "TFDWConv() only valid for k=4 and p1=1"
        weight, bias = w.weight.permute(2, 3, 1, 0).numpy(), w.bias.numpy()
        self.c1 = c1
        self.conv = [
            keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=k,
                strides=s,
                padding="VALID",
                output_padding=p2,
                use_bias=True,
                kernel_initializer=keras.initializers.Constant(weight[..., i : i + 1]),
                bias_initializer=keras.initializers.Constant(bias[i]),
            )
            for i in range(c1)
        ]

    def call(self, inputs):
        """
        Performs a forward pass by applying parallel depthwise Conv2DTranspose operations on split input tensors and
        concatenates the results.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels) to be processed.
        
        Returns:
            tf.Tensor: Output tensor with the same shape as the input tensor.
        
        Examples:
            ```python
            # Initialize the layer
            layer = TFDWConvTranspose2d(c1=64, c2=64, k=4, s=2, p1=1, p2=0, w=pretrained_weights)
            
            # Apply to input tensor
            output = layer(tf.random.normal([1, 32, 32, 64]))
            ```
            
        Notes:
            The TFDWConvTranspose2d layer is configured for specific kernel size and padding parameters, k=4 and p1=1,
            for depthwise transposed convolutions.
        """
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        """
        Initializes the TFFocus layer to efficiently transform spatial (width, height) information into channel space via a convolutional layer.
        
        Args:
            c1 (int): The number of input channels.
            c2 (int): The number of output channels.
            k (int, optional): Kernel size for the convolutional layer. Default is 1.
            s (int, optional): Stride for the convolutional layer. Default is 1.
            p (int | None, optional): Padding for the convolution. Default is None.
            g (int, optional): Number of groups for the convolution. Default is 1.
            act (bool, optional): If True, applies activation after convolution. Default is True.
            w (object | None, optional): Weights and initializers for the layer, if any. Default is None.
        
        Returns:
            None
        """
        super().__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        """
        Efficiently reduces spatial dimensions by 2 and increases the channel dimensions by 4.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, width, height, channels) for processing.
                
        Returns:
            tf.Tensor: Processed tensor with reduced spatial dimensions and increased channels, shape (batch_size, width/2, 
            height/2, 4*channels).
        
        Example:
            ```python
            focus_layer = TFFocus(c1=64, c2=128)
            output = focus_layer(input_tensor)
            ```
        """
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, groups, expansion
        """
        Initializes a standard bottleneck layer with Depthwise convolution, optional shortcut, and channel expansion.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): If True, adds a residual shortcut connection. Defaults to True.
            g (int): Number of groups for group convolution. Defaults to 1.
            e (float): Expansion ratio for the hidden channels. Defaults to 0.5.
            w (object): Weights for initializing the convolutional layers.
        
        Returns:
            None
        
        Notes:
            This implementation follows the design principles of YOLOv3's bottleneck layers, tailored for TensorFlow.
        
        Example:
            ```python
            bottleneck_layer = TFBottleneck(c1=256, c2=512, shortcut=True, g=1, e=0.5, w=weights)
            output = bottleneck_layer(input_tensor)
            ```
        
        See Also:
            - [TensorFlow BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)
            - [YOLOv3 Architecture](https://github.com/ultralytics/yolov5/pull/1127)
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        class TFBottleneck(keras.layers.Layer):
            # Standard bottleneck
            def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):
                """
                Initializes a standard bottleneck layer with optional shortcut, channel expansion, and group
                convolutions.

                Args:
                    c1 (int): Number of input channels.
                    c2 (int): Number of output channels.
                    shortcut (bool, optional): Whether to use shortcut connection. Defaults to True.
                    g (int, optional): Number of groups for group convolution. Defaults to 1.
                    e (float, optional): Expansion ratio for hidden channels. Defaults to 0.5.
                    w: (optional): Weights for initializing the convolutional layers. Defaults to None.
                """
                super().__init__()
                c_ = int(c2 * e)  # hidden channels
                self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1 if w else None)
                self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2 if w else None)
                self.add = shortcut and c1 == c2
        
            def call(self, inputs):
                """
                Executes a bottleneck layer with optional shortcut; returns either input + convoluted input or just
                convoluted input.

                Args:
                    inputs (tf.Tensor): Input tensor to the bottleneck layer.

                Returns:
                    tf.Tensor: The output tensor after applying the bottleneck transformation.
                """
                if self.add:
                    return inputs + self.cv2(self.cv1(inputs))
                else:
                    return self.cv2(self.cv1(inputs))
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFCrossConv(keras.layers.Layer):
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        """
        Initializes a cross convolutional layer with flexible configurations for channel sizes, kernel size, stride, groups, 
        expansion factor, shortcut connection, and custom weights.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple[int, int], optional): Kernel size. Defaults to 3.
            s (int | tuple[int, int], optional): Stride. Defaults to 1.
            g (int, optional): Number of groups for group convolution. Defaults to 1.
            e (float, optional): Expansion factor for the hidden channels. Defaults to 1.0.
            shortcut (bool, optional): Whether to add a shortcut connection. Defaults to False.
            w (object, optional): Pre-trained weights for the layer. Defaults to None.
        
        Returns:
            None: This constructor does not return any value.
        
        Examples:
            ```python
            # Initialize a cross convolutional layer with default parameters
            layer = TFCrossConv(c1=64, c2=128)
            
            # Initialize with custom kernel size and stride
            layer = TFCrossConv(c1=64, c2=128, k=(3, 5), s=(1, 2))
        
            # Initialize with a shortcut connection and specific weights
            layer = TFCrossConv(c1=64, c2=128, shortcut=True, w=my_pretrained_weights)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        """
        Executes cross convolutional layers, optionally adding a shortcut connection if input and output channels match.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).
        
        Returns:
            tf.Tensor: Output tensor after applying cross convolution, with optional shortcut connection if configured.
        
        Notes:
            - The input tensor should be a 4D tensor with dimensions [B, C, H, W] where B is the batch size, C is the
              number of channels, H is the height, and W is the width.
            - This function utilizes two sequential convolutions: one with a (1, k) kernel and another with a (k, 1) kernel.
            - The use of shortcuts is contingent upon whether it is enabled and if input channels equal output channels.
        """
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFConv2d(keras.layers.Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        """
        Initializes a 2D convolutional layer compatible with TensorFlow 2.2+, substituting PyTorch's nn.Conv2D.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int | tuple[int, int]): Size of the convolving kernel.
            s (int, optional): Stride of the convolution. Defaults to 1.
            g (int, optional): Number of grouped convolutions. Must be 1 as TensorFlow v2.2 doesn't support groups. Defaults to 1.
            bias (bool, optional): Boolean flag to include bias term. Defaults to True.
            w (torch.nn.Conv2d, optional): Pre-trained PyTorch Conv2d layer from which to import weights. Defaults to None.
        
        Returns:
            None
        
        Notes:
            The TFConv2d class is designed to closely mimic the behavior and initialization of PyTorch’s nn.Conv2d within a
            TensorFlow environment, allowing for interoperability and flexible usage in both frameworks. Example usage can be
            seen in tasks involving the YOLOv3 model within the Ultralytics framework.
        """
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        self.conv = keras.layers.Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding="VALID",
            use_bias=bias,
            kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer=keras.initializers.Constant(w.bias.numpy()) if bias else None,
        )

    def call(self, inputs):
        """
        Executes a forward pass of convolutional operation using the initialized weights and biases.
        
        Args:
            inputs (tf.Tensor): Input tensor to apply the convolution on, with shape [batch_size, height, width, channels].
        
        Returns:
            tf.Tensor: The output tensor after applying the convolution, with shape [batch_size, new_height, new_width, filters].
        
        Notes:
            - This function is a TensorFlow-based implementation substituting PyTorch's `nn.Conv2D`.
            - The `groups` argument is not supported in TensorFlow v2.2 Conv2D, so `g` must always be 1.
        
        Example:
        ```python
        # Example usage of TFConv2d
        conv_layer = TFConv2d(c1=3, c2=64, k=3, s=1, bias=True, w=pretrained_weights)
        output = conv_layer(input_tensor)
        ```
        """
        Return self.conv(inputs)

        class TFBottleneckCSP(keras.layers.Layer):     # CSP Bottleneck
        https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes a CSP Bottleneck layer with specified channel parameters, optionally including a shortcut, group
        convolutions, and expansion factor.
        
        Args:
            c1 (int): The number of input channels.
            c2 (int): The number of output channels.
            n (int): Number of bottleneck layers to apply. Default is 1.
            shortcut (bool): Whether to include a shortcut connection. Default is True.
            g (int | None): Number of groups for group convolutions. Default is 1.
            e (float | None): Expansion factor for the bottleneck layer. Default is 0.5.
            w (object | None): Weights to initialize the layer.
        
        Returns:
            None
        
        Notes:
            This layer follows the Cross Stage Partial Networks (CSPNet) design as outlined in the paper:
            https://github.com/WongKinYiu/CrossStagePartialNetworks
        
        Examples:
            ```python
            from ultralytics import TFBottleneckCSP
            
            # Initialize the TFBottleneckCSP layer with specific parameters
            csp_layer = TFBottleneckCSP(c1=64, c2=128, n=1, shortcut=True, g=1, e=0.5, w=weights)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2)
        self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3)
        self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4)
        self.bn = TFBN(w.bn)
        self.act = lambda x: keras.activations.swish(x)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """
        Executes the forward pass for the CSP bottleneck layer by combining features through convolutions, activations, and 
        batch normalization.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape [B, H, W, C], where B is the batch size, H is the height, W is the width, 
                and C is the number of channels.
        
        Returns:
            tf.Tensor: Output tensor after applying the CSP bottleneck operations.
        """
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes a CSP Bottleneck layer with 3 convolutions, useful for channel manipulation and feature integration.
        
        Args:
            c1 (int): The number of input channels.
            c2 (int): The number of output channels.
            n (int): The number of bottleneck layers to include. Default is 1.
            shortcut (bool): If True, adds a residual connection between input and output. Default is True.
            g (int): Number of groups for group convolutions. Default is 1.
            e (float): Expansion factor for hidden channels. Default is 0.5.
            w (object): Weights and biases used to initialize convolutional layers.
        
        Returns:
            None
        
        Example:
            ```python
            # Example of creating TFC3 layer with custom parameters
            from some_module import WeightsLoader
            weights = WeightsLoader(...)  # provide a mechanism to load weights
            tfc3_layer = TFC3(c1=64, c2=128, n=3, shortcut=True, g=2, e=0.5, w=weights)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """
        Executes the forward pass for a CSP Bottleneck layer with 3 convolutions.
        
        Args:
            inputs (tf.Tensor): Input tensor with shape [batch_size, height, width, channels].
        
        Returns:
            tf.Tensor: Output tensor after applying the three convolutional bottleneck layers and concatenation.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFC3x(keras.layers.Layer):
    # 3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes a TFC3x layer in the Ultralytics library, designed for cross-convolutions, which expands and concatenates
        features for given channel inputs and outputs, with optional shortcut connections.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck layers to apply in sequence. Default is 1.
            shortcut (bool): Whether to use shortcut (skip) connections. Default is True.
            g (int): Number of groups for grouped convolution. Default is 1.
            e (float): Expansion factor for channels. Default is 0.5.
            w (object): Weights for initializing layer parameters. Should include submodules with `cv1`, `cv2`, `cv3`, and `m`
                        weights.
        
        Returns:
            None
        
        Examples:
            ```python
            layer = TFC3x(c1=64, c2=128, n=2, shortcut=True, g=1, e=0.5, w=weights)
            output = layer(inputs)
            ```
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential(
            [TFCrossConv(c_, c_, k=3, s=1, g=g, e=1.0, shortcut=shortcut, w=w.m[j]) for j in range(n)]
        )

    def call(self, inputs):
        """
        TFC3x.call
        Executes the forward pass of a CSP Bottleneck layer with cross-convolutions, combining features through sequential 
        convolutions and concatenation.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).
        
        Returns:
            tf.Tensor: Output tensor after applying the forward pass operations (batch_size, new_height, new_width, new_channels).
        
        Notes:
            This method integrates features from the input tensor through a sequential application of cross-convolutions and 
            concatenation, useful for deep learning models that require efficient channel manipulation and feature integration within
            YOLO-style architectures. Ensure that the input tensor dimensions align with the initialized layer parameters for proper 
            execution.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        """
        Initializes a Spatial Pyramid Pooling (SPP) layer for YOLOv3-SPP with specified input/output channels and kernel sizes.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (tuple[int], optional): Kernel sizes for the pooling layers. Default is (5, 9, 13).
            w (object, optional): Pre-trained weights for the layer. Default is None.
        
        Returns:
            None: This function does not return a value; it initializes the SPP layer.
        
        Notes:
            The SPP layer is designed to increase the receptive field by pooling features at different scales,
            which facilitates capturing objects at various sizes in the input image.
        
        Example:
            ```python
            # Initialize TFSPP layer with 256 input channels, 512 output channels, and default kernel sizes.
            spp_layer = TFSPP(256, 512)
        
            # Use the layer in a model
            outputs = spp_layer(inputs)
            ```
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding="SAME") for x in k]

    def call(self, inputs):
        """
        Executes the Spatial Pyramid Pooling (SPP) layer transformation on the input tensor, concatenating multiple 
        max-pooled feature maps.
                
        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, height, width, channels].
        
        Returns:
            tf.Tensor: Transformed tensor after applying SPP and convolution layers, shape [batch_size, new_height,
               new_width, new_channels].
        
        Examples:
            ```python
            spp_layer = TFSPP(c1=256, c2=512)
            input_tensor = tf.random.normal([1, 32, 32, 256])
            output_tensor = spp_layer(input_tensor)
            print(output_tensor.shape)  # Expected output shape [1, 32, 32, 512]
            ```
        
        Notes:
            The SPP layer allows the model to extract features at multiple scales by using different sizes of max-pooling 
            operations, enhancing the receptive field and robustness to object scaling.
        """
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(keras.layers.Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):
        """
        Initializes a Spatial Pyramid Pooling-Fast (SPPF) layer, a fast implementation of SPP for YOLO models.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for max pooling layers. Default is 5.
            w (ModuleType): Weights from PyTorch model, optional. Used to initialize the TensorFlow layers.
        
        Returns:
            None
        
        Examples:
            ```python
            # Initialize TFSPPF layer
            sppf_layer = TFSPPF(c1=512, c2=1024, k=5, w=weights)
            ```
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME")

    def call(self, inputs):
        """
        TFSPPF is a class that implements a TensorFlow Keras Layer for the Spatial Pyramid Pooling-Fast operation. 
        It is initialized with input and output channels, kernel size, and optional weights for the convolutional layers.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the MaxPool operation. Defaults to 5.
            w (object, optional): Pretrained weights for initialization. Defaults to None.
        
        Returns:
            tf.Tensor: Resulting tensor after applying the Spatial Pyramid Pooling-Fast (SPPF) operation.
        """
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFDetect(keras.layers.Layer):
    # TF YOLOv3 Detect layer
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):  # detection layer
        """
        Initializes YOLOv3 detection layer with specified parameters such as number of classes, anchors, channels, image size, and weights.
        
        Args:
            nc (int): Number of classes.
            anchors (tuple): Tuple of anchor box dimensions.
            ch (tuple): Tuple of input channels.
            imgsz (tuple[int, int]): Image size as a tuple of height and width.
            w (object): Weight object containing model parameters for initialization.
        
        Returns:
            None: This is an initializer method and does not return any value.
        
        Notes:
            - The function initializes several internal attributes such as stride, number of outputs, number of layers, and anchors.
            - It uses TensorFlow-specific functions and Keras layers to construct the YOLOv3 detection model architecture.
            - Ensure TensorFlow, Keras, and related dependencies are properly installed and configured in your environment.
        
        Examples:
            ```python
            # Example usage:
            import tensorflow as tf
            from models.yolo import TFDetect
        
            # Initialize TFDetect with custom parameters
            detect_layer = TFDetect(nc=80, anchors=anchors, ch=(128, 256, 512), imgsz=(640, 640), w=custom_weights)
            ```
        """
        super().__init__()
        self.stride = tf.convert_to_tensor(w.stride.numpy(), dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float32)
        self.anchor_grid = tf.reshape(self.anchors * tf.reshape(self.stride, [self.nl, 1, 1]), [self.nl, 1, -1, 1, 2])
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.training = False  # set to False after building model
        self.imgsz = imgsz
        for i in range(self.nl):
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def call(self, inputs):
        """
        TFDetect.call(inputs)
        Applies YOLOv3 detection layer on the input tensor, transforming the shape and performing inference.
        
        Args:
          inputs (list of tf.Tensor): A list of input tensors for each YOLO layer, where each tensor corresponds to different
            feature levels and is expected to have dimensions [batch_size, H, W, C].
        
        Returns:
          list of tf.Tensor | None: If in inference mode, returns a list of output tensors for each YOLO layer, with transform
            shape (batch_size, ny*nx, num_anchors, num_outputs). Each output tensor has detections for the corresponding feature
            level. In training mode, returns None.
        
        Notes:
          - Shapes and dimensions of tensors need to be consistent with the model's configuration, especially the image size
            and number of classes.
          - This function handles both training and inference modes, transforming the input tensors differently based on the
            mode.
          - The input layers should be pre-defined with appropriate channels and configuration before calling this function.
        
        Example usage:
        ```python
        # Assuming `model` is an instance of TFDetect and input_tensors is a list of tf.Tensors
        output_tensors = model.call(input_tensors)
        ```
        """
        Z = []  # inference output x = [] for i in range(self.nl): x.append(self.m[i](inputs[i])) # x(bs,20,20,255) to
        x(bs,3,20,20,85)

                ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
                x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no])

                if not self.training:  # inference
                    y = x[i]
                    grid = tf.transpose(self.grid[i], [0, 2, 1, 3]) - 0.5
                    anchor_grid = tf.transpose(self.anchor_grid[i], [0, 2, 1, 3]) * 4
                    xy = (tf.sigmoid(y[..., 0:2]) * 2 + grid) * self.stride[i]  # xy
                    wh = tf.sigmoid(y[..., 2:4]) ** 2 * anchor_grid
                    # Normalize xywh to 0-1 to reduce calibration error
                    xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                    wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                    y = tf.concat([xy, wh, tf.sigmoid(y[..., 4 : 5 + self.nc]), y[..., 5 + self.nc :]], -1)
                    z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no]))

            return tf.transpose(x, [0, 2, 1, 3]) if self.training else (tf.concat(z, 1),)

        @staticmethod
        def _make_grid(nx=20, ny=20):
        """
        Generates a grid of shape [1, 1, ny * nx, 2] with ranges [0, nx) and [0, ny) for object detection.
        
        Args:
            nx (int): The number of grid cells along the x-axis.
            ny (int): The number of grid cells along the y-axis.
        
        Returns:
            tf.Tensor: A TensorFlow tensor of shape [1, 1, ny * nx, 2] containing the grid coordinates.
        
        Notes:
            The grid is used in YOLOv3 for aligning the predicted bounding boxes with the actual positions in the image.
            Each cell in the grid corresponds to an anchor box's center position.
        
        Examples:
            ```python
            grid = TFDetect._make_grid(20, 20)
            print(grid.shape)  # Expected output: (1, 1, 400, 2)
            ```
        
            The grid has integers from 0 to nx and 0 to ny for the x and y coordinates, respectively, for each grid cell.
        """
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        grid = tf.stack((xv, yv), axis=2)
        return tf.reshape(grid, [1, 1, ny * nx, 2])
        """
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


class TFSegment(TFDetect):
    # YOLOv3 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
        """
        Initializes a YOLOv3 Segment head with customizable parameters for segmentation models.

        Args:
            nc (int): Number of classes. Default is 80.
            anchors (tuple): Tuple containing anchor details.
            nm (int): Number of masks. Default is 32.
            npr (int): Number of protos. Default is 256.
            ch (list): List of input channels.
            imgsz (tuple): Image size in pixels. Default is (640, 640).
            w: Weights for initializing the layers.

        Returns:
            None

        Notes:
            This class inherits from `TFDetect` and adds functionalities needed for segmentation tasks by defining additional
            convolutional layers and proto layers to handle segmentation masks. It retains the detection capabilities of the
            base YOLOv3 model while extending it towards pixel-level segmentation.
        """
        super().__init__(nc, anchors, ch, imgsz, w)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]  # output conv
        self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto)  # protos
        self.detect = TFDetect.call

    def call(self, x):
        """
        Executes the forward pass for the segment head, returning predictions and prototype masks.

        Args:
            x (List[tf.Tensor]): List of input tensors, typically the output feature maps from the backbone network.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Predictions and prototype masks. Predictions have shape (batch_size, num_boxes,
                num_outputs) and prototype masks have shape (batch_size, num_protos, height, width).

        Notes:
            This method relies on the `TFDetect.call` method for detection components, extending it with
            prototype mask generation specific to segmentation tasks. The prototype masks are essential for
            downstream segmentation refinements.

        Examples:
            ```python
            segment = TFSegment(nc=80, anchors=anchors, nm=32, npr=256, ch=channels, imgsz=(640, 640), w=weights)
            predictions, protos = segment(inputs)
            ```
        """
        p = self.proto(x[0])
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        p = tf.transpose(p, [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p)


class TFProto(keras.layers.Layer):
    def __init__(self, c1, c_=256, c2=32, w=None):
        """
        Initializes a TFProto layer with specified convolutional and upsampling parameters, enabling advanced feature
        processing.

        Args:
            c1 (int): Number of input channels.
            c_ (int, optional): Number of hidden channels. Default is 256.
            c2 (int, optional): Number of output channels. Default is 32.
            w (object, optional): Collection of weights for initializing the layer, including convolutional layers.

        Returns:
            None

        Notes:
            This class is used for processing and refining features within a TensorFlow-based YOLOv3 model.
            The convolutional layers are initialized based on the provided weights, enhancing model training and inference.

        Examples:
            ```python
            proto_layer = TFProto(c1=256, c_=256, c2=32, w=weights)
            ```
        """
        super().__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
        self.upsample = TFUpsample(None, scale_factor=2, mode="nearest")
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
        self.cv3 = TFConv(c_, c2, w=w.cv3)

    def call(self, inputs):
        """
        Handles the forward pass for the TFProto layer, performing consecutive convolution and upsample operations on
        the input tensor.

        Args:
            inputs (tf.Tensor): A TensorFlow tensor representing the input features to be processed.

        Returns:
            tf.Tensor: A TensorFlow tensor resulting from the series of convolutions and upsample operations on the input features.

        Example:
            ```python
            proto_layer = TFProto(c1=64, c_=128, c2=32)
            processed_features = proto_layer(input_tensor)
            ```
        Notes:
            The layer performs three convolution operations interleaved with an upsample operation. The upsample operation increases the spatial dimensions of the features by a factor of 2, using nearest-neighbor interpolation.
        ```
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))


class TFUpsample(keras.layers.Layer):
    # TF version of torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):  # warning: all arguments needed including 'w'
        """
        Initializes an upsampling layer using nearest-neighbor interpolation with configurable size, scale factor, mode,
        and optional weights.

        Args:
          size (int | None): The target size for the output tensor. If None, scale_factor is used.
          scale_factor (int | None): The multiplier for scaling the input tensor. Must be a multiple of 2.
          mode (str): Interpolation mode to be used for upsampling (e.g., 'nearest').
          w (torch.nn.Module, optional): Weights for the layer, default is None.

        Returns:
          None (None): This constructor does not return any value.

        Raises:
          AssertionError: If the scale_factor is not a multiple of 2.

        Notes:
          - This class mimics the behavior of `torch.nn.Upsample` using TensorFlow's functionalities.
          - Ensure the scale_factor is a positive multiple of 2 for proper functionality.

        Example:
          ```python
          upsample_layer = TFUpsample(size=None, scale_factor=4, mode='nearest')
          upscaled_tensor = upsample_layer(input_tensor)
          ```
        """
        super().__init__()
        assert scale_factor % 2 == 0, "scale_factor must be multiple of 2"
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode)
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # with default arguments: align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))

    def call(self, inputs):
        """
        Upsamples the input tensor to a specified size or scale factor using a specified interpolation mode.

        Args:
          inputs (tf.Tensor): The input tensor to be upsampled.

        Returns:
          tf.Tensor: The upsampled tensor with increased spatial dimensions.
        """
        return self.upsample(inputs)


class TFConcat(keras.layers.Layer):
    # TF version of torch.concat()
    def __init__(self, dimension=1, w=None):
        """
        Initializes the TFConcat layer for combining multiple input tensors along the specified dimension using
        TensorFlow.

        Args:
            dimension (int): Dimension along which to concatenate the input tensors. For consistency with the TF model,
                this should always be 1.
            w (Optional[tf.Tensor]): A TensorFlow tensor containing weights for the layer. Defaults to None.

        Returns:
            None. This is an initializer method and doesn't return any value.

        Notes:
            The TFConcat layer is designed as a TensorFlow equivalent to PyTorch's `torch.cat` function. The dimension argument
            should always be set to 1 when converting from NCHW (channels-first) format in PyTorch to NHWC (channels-last) format
            in TensorFlow.
        """
        super().__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def call(self, inputs):
        """
        Concatenates input tensors along the NHWC dimension.

        Args:
            inputs (list[tf.Tensor]): List of tensors to concatenate along the NHWC dimension.

        Returns:
            tf.Tensor: Concatenated tensor along the NHWC dimension.

        Examples:
            ```python
            concat_layer = TFConcat()
            tensor1 = tf.random.normal(shape=(1, 32, 32, 3))
            tensor2 = tf.random.normal(shape=(1, 32, 32, 3))
            output = concat_layer([tensor1, tensor2])
            print(output.shape)  # Expected output shape: (1, 32, 32, 6)
            ```

        Notes:
            This function is specifically designed to work with tensors in NHWC format. Ensure the input tensors conform to this format for expected behavior.
        """
        return tf.concat(inputs, self.d)


def parse_model(d, ch, model, imgsz):  # model_dict, input_channels(3)
    """
    Parses the model configuration and constructs a Keras model with defined layer connectivity.

    Args:
        d (dict): Model configuration dictionary.
        ch (list[int]): List of input channel dimensions.
        model (nn.Module): Pre-trained PyTorch model.
        imgsz (tuple[int, int]): Image size (height, width) in pixels.

    Returns:
        tuple:
            - keras.Sequential: Compiled Keras model.
            - list[int]: List of save indices indicating layers to save during inference.

    Notes:
        This function evaluates string representations of layers and settings within d to dynamically build the model. It
        converts certain PyTorch layers and utilities to their TensorFlow equivalents through the use of `eval()`.

    Examples:
        ```python
        from models.yolo import Model
        model_cfg = {
            "anchors": [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
            "nc": 80,
            "depth_multiple": 0.33,
            "width_multiple": 0.50,
            "backbone": [
                [1, 3, 'Conv', [64, 3, 1]],
                [1, 1, 'BottleneckCSP', [128]],
                # ... more layers
            ],
            "head": [
                [1, 1, 'SPP', [512]],
                [1, 1, 'Detect', [80, [116, 90, 156, 198, 373, 326]]]
                # ... more layers
            ],
        }
        channels = [3]
        model = Model(cfg=model_cfg)
        keras_model, save_layers = parse_model(model_cfg, channels, model, imgsz=(640, 640))
        ```

        The returned `keras_model` can be used in TensorFlow workflows, while the `save_layers` list specifies which layers
        need to be saved for inference time operations.
    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"]
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m_str = m
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            nn.Conv2d,
            Conv,
            DWConv,
            DWConvTranspose2d,
            Bottleneck,
            SPP,
            SPPF,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3x,
        ]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3x]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m in [Detect, Segment]:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
            args.append(imgsz)
        else:
            c2 = ch[f]

        tf_m = eval("TF" + m_str.replace("nn.", ""))
        m_ = (
            keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)])
            if n > 1
            else tf_m(*args, w=model.model[i])
        )  # module

        torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in torch_m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return keras.Sequential(layers), sorted(save)


class TFModel:
    # TF YOLOv3 model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, model=None, imgsz=(640, 640)):  # model, channels, classes
        """
        Initializes the YOLOv3 model for TensorFlow with provided configuration, channels, classes, optional pre-loaded
        model, and input image size.

        Args:
            cfg (str | dict): The path to the model configuration file or a dictionary containing the model configuration.
            ch (int): Number of input channels.
            nc (int, optional): Number of object classes for detection. Overrides the value in the configuration if provided.
            model (torch.nn.Module, optional): A pre-loaded PyTorch model from which to load weights.
            imgsz (tuple[int, int], optional): The dimensions (height, width) of the input images, default is (640, 640).

        Returns:
            None

        Note:
            Example usage:
            ```python
            model = TFModel(cfg='yolov5s.yaml', ch=3, nc=80)
            ```
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)

    def predict(
        self,
        inputs,
        tf_nms=False,
        agnostic_nms=False,
        topk_per_class=100,
        topk_all=100,
        iou_thres=0.45,
        conf_thres=0.25,
    ):
        """
        Performs inference on input data using a pre-trained YOLOv3 model, including optional post-processing with
        TensorFlow NMS.

        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, height, width, channels] representing the input images.
            tf_nms (bool): Flag to apply TensorFlow Non-Maximum Suppression (NMS) for post-processing. Defaults to False.
            agnostic_nms (bool): Flag to apply class-agnostic NMS. Effective only if `tf_nms` is True. Defaults to False.
            topk_per_class (int): Maximum number of detections to retain per class, used for NMS. Defaults to 100.
            topk_all (int): Maximum total number of detections to retain per image, used for NMS. Defaults to 100.
            iou_thres (float): IoU threshold for NMS. Defaults to 0.45.
            conf_thres (float): Confidence threshold to filter predictions. Defaults to 0.25.

        Returns:
            (tuple): If `tf_nms` is enabled, returns a tuple containing the NMS results, where each element is a tensor of shape
            [num_detections, 6] representing [x1, y1, x2, y2, confidence, class]; otherwise, returns the raw model outputs.

        Examples:
            ```python
            model = TFModel(cfg='path/to/yolov5.yaml', ch=3, nc=80)
            inputs = tf.random.uniform((1, 640, 640, 3))
            outputs = model.predict(inputs, tf_nms=True, conf_thres=0.3)
            ```

        Notes:
            This function performs YOLOv3-specific operations to transform model outputs into detection boxes. When `tf_nms` is
            enabled, it utilizes TensorFlow's `combined_non_max_suppression` for efficient post-processing. Use the appropriate
            threshold values and top-K limits to tailor the performance and output as per the use case.
        """
        y = []  # outputs
        x = inputs
        for m in self.model.layers:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output

        # Add TensorFlow NMS
        if tf_nms:
            boxes = self._xywh2xyxy(x[0][..., :4])
            probs = x[0][:, :, 4:5]
            classes = x[0][:, :, 5:]
            scores = probs * classes
            if agnostic_nms:
                nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
            else:
                boxes = tf.expand_dims(boxes, 2)
                nms = tf.image.combined_non_max_suppression(
                    boxes, scores, topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes=False
                )
            return (nms,)
        return x  # output [1,6300,85] = [xywh, conf, class0, class1, ...]
        # x = x[0]  # [x(1,6300,85), ...] to x(6300,85)
        # xywh = x[..., :4]  # x(6300,4) boxes
        # conf = x[..., 4:5]  # x(6300,1) confidences
        # cls = tf.reshape(tf.cast(tf.argmax(x[..., 5:], axis=1), tf.float32), (-1, 1))  # x(6300,1)  classes
        # return tf.concat([conf, cls, xywh], 1)

    @staticmethod
    def _xywh2xyxy(xywh):
        """
        Converts bounding boxes from [x, y, w, h] to [x1, y1, x2, y2] format, where (x1, y1) represents the top-left
        corner, and (x2, y2) represents the bottom-right corner.

        Args:
            xywh (tf.Tensor): Tensor of shape (..., 4) containing bounding boxes in [x, y, w, h] format, where (x, y)
            represents the center of the box, and (w, h) are the width and height of the box.

        Returns:
            tf.Tensor: Tensor of the same shape as `xywh` containing converted bounding boxes in [x1, y1, x2, y2] format.

        Example:
            ```python
            xywh_tensor = tf.constant([[50.0, 50.0, 100.0, 100.0], [30.0, 40.0, 60.0, 80.0]])  # (2, 4) shape
            xyxy_tensor = TFModel._xywh2xyxy(xywh_tensor)
            ```
        """
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    def call(self, input, topk_all, iou_thres, conf_thres):
        """
        Agnostic non-maximum suppression (NMS).

        Args:
          input (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): A tuple containing tensors for boxes, scores, and classes.
          topk_all (int): Maximum number of total detections to retain.
          iou_thres (float): Intersection over Union (IoU) threshold for NMS.
          conf_thres (float): Confidence threshold for detections.

        Returns:
          Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Filtered boxes, scores, detected classes, and number of detected
          boxes (all are tensors).

        Example:
          ```python
          agnostic_nms = AgnosticNMS()
          boxes, scores, classes, num = agnostic_nms((boxes, scores, classes), topk_all=100, iou_thres=0.45, conf_thres=0.25)
          ```
        """
        return tf.map_fn(
            lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
            input,
            fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
            name="agnostic_nms",
        )

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):  # agnostic NMS
        """
        AgnosticNMS._nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25): Performs non-maximum suppression (NMS) on
        bounding boxes, considering IoU, confidence, and top-K thresholds.

        Args:
         x (tuple): A tuple containing:
           - boxes (tf.Tensor): Tensor of shape (num_boxes, 4) representing bounding boxes in [x1, y1, x2, y2] format.
           - classes (tf.Tensor): Tensor of shape (num_boxes, num_classes) representing class probabilities.
           - scores (tf.Tensor): Tensor of shape (num_boxes, num_classes) representing classification scores per class.
         topk_all (int, optional): Maximum number of total top-K predictions to keep after applying NMS. Defaults to 100.
         iou_thres (float, optional): Intersection-over-union (IoU) threshold for performing NMS. Defaults to 0.45.
         conf_thres (float, optional): Confidence threshold for filtering predictions before applying NMS. Defaults to 0.25.

        Returns:
         tuple: A tuple consisting of:
           - padded_boxes (tf.Tensor): Tensor of shape (topk_all, 4) representing the filtered bounding boxes after NMS.
           - padded_scores (tf.Tensor): Tensor of shape (topk_all,) with the corresponding scores.
           - padded_classes (tf.Tensor): Tensor of shape (topk_all,) with the respective class indices.
           - valid_detections (tf.int32): Number of valid detections after NMS which is ≤ topk_all.
        """
        boxes, classes, scores = x
        class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
        scores_inp = tf.reduce_max(scores, -1)
        selected_inds = tf.image.non_max_suppression(
            boxes, scores_inp, max_output_size=topk_all, iou_threshold=iou_thres, score_threshold=conf_thres
        )
        selected_boxes = tf.gather(boxes, selected_inds)
        padded_boxes = tf.pad(
            selected_boxes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
            mode="CONSTANT",
            constant_values=0.0,
        )
        selected_scores = tf.gather(scores_inp, selected_inds)
        padded_scores = tf.pad(
            selected_scores,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        selected_classes = tf.gather(class_inds, selected_inds)
        padded_classes = tf.pad(
            selected_classes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        valid_detections = tf.shape(selected_inds)[0]
        return padded_boxes, padded_scores, padded_classes, valid_detections


def activations(act=nn.SiLU):
    """
    Returns the corresponding TensorFlow activation function based on the provided PyTorch activation.

    Args:
        act (nn.Module): A PyTorch activation function module, e.g., nn.LeakyReLU, nn.Hardswish, nn.SiLU.

    Returns:
        function: A TensorFlow activation function corresponding to the provided PyTorch activation function.

    Notes:
        - If the provided activation function is `nn.LeakyReLU`, the corresponding TensorFlow activation function will be
        `keras.activations.relu` with `alpha=0.1`.
        - If the provided activation function is `nn.Hardswish`, the corresponding TensorFlow activation function will be
        implemented as `x * tf.nn.relu6(x + 3) * 0.166666667`.
        - If the provided activation function is `nn.SiLU` or `SiLU`, the corresponding TensorFlow activation function will
        be `keras.activations.swish`.
        - For any other activation function, it directly returns the input activation function.

    Example:
        ```python
        from torch import nn
        import tensorflow as tf
        from tensorflow import keras

        tf_activation = activations(nn.SiLU())
        assert tf_activation(tf.constant([1.0, 2.0, 3.0])) == keras.activations.swish(tf.constant([1.0, 2.0, 3.0]))
        ```
    """
    if isinstance(act, nn.LeakyReLU):
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif isinstance(act, nn.Hardswish):
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif isinstance(act, (nn.SiLU, SiLU)):
        return lambda x: keras.activations.swish(x)
    else:
        raise Exception(f"no matching TensorFlow activation found for PyTorch activation {act}")


def representative_dataset_gen(dataset, ncalib=100):
    """
    Generates a representative dataset for TensorFlow Lite (TFLite) model conversion by yielding normalized numpy arrays
    from the input dataset.

    Args:
      dataset (iterable): An iterable yielding the dataset items, typically from a data loader.
      ncalib (int, optional): The maximum number of calibration samples to yield. Defaults to 100.

    Yields:
      list: A list containing a single numpy array representing the normalized image data with shape (1, height, width, channels), suitable for TFLite quantization.

    Notes:
      This function is typically used during TFLite model conversion to provide a sample dataset for quantization calibration. Below is an example of how to use it.

    Example:
      ```python
      import tensorflow as tf
      from some_data_loader import load_data
      dataset = load_data()
      converter = tf.lite.TFLiteConverter.from_keras_model(model)
      converter.representative_dataset = lambda: representative_dataset_gen(dataset)
      tflite_model = converter.convert()
      ```

    For more details, refer to the TensorFlow documentation on TFLite representative datasets:
    https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization.
    """
    for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
        im = np.transpose(img, [1, 2, 0])
        im = np.expand_dims(im, axis=0).astype(np.float32)
        im /= 255
        yield [im]
        if n >= ncalib:
            break


def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # inference size h,w
    batch_size=1,  # batch size
    dynamic=False,  # dynamic batch size
):
    # PyTorch model
    """
    Exports and summarizes both PyTorch and TensorFlow models for YOLOv5-based object detection.

    Args:
        weights (str | Path): Path to the weights file. Default is 'ROOT / "yolov5s.pt"'.
        imgsz (tuple[int, int]): Inference image size as (height, width). Default is (640, 640).
        batch_size (int): Batch size for inference. Default is 1.
        dynamic (bool): Flag for dynamic batch size in Keras model. Default is False.

    Returns:
        None

    Notes:
        - This function outputs the model architecture and summary in the terminal.
        - It initializes models with dummy inputs for both PyTorch and TensorFlow frameworks.
        - The Keras model is compiled and summarized to provide a comprehensive overview of the model structure.

    Examples:
        To run the function with default parameters:
        ```python
        run()
        ```

        To run the function with a custom weights path and batch size:
        ```python
        run(weights="/path/to/weights.pt", batch_size=4)
        ```

    Links:
        Original YOLOv5 implementation and usage: https://github.com/ultralytics/yolov5
        TensorFlow, Keras and TFLite implementation details: https://github.com/ultralytics/yolov5/pull/1127
    """
    im = torch.zeros((batch_size, 3, *imgsz))  # BCHW image
    model = attempt_load(weights, device=torch.device("cpu"), inplace=True, fuse=False)
    _ = model(im)  # inference
    model.info()

    # TensorFlow model
    im = tf.zeros((batch_size, *imgsz, 3))  # BHWC image
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    _ = tf_model.predict(im)  # inference

    # Keras model
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    keras_model.summary()

    LOGGER.info("PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.")


def parse_opt():
    """
    Parses command line arguments for configuring the model, including weights path, image size, batch size, and dynamic
    batching.

    Args:
      --weights (str): Path to the weights file. Default is "ROOT/yolov3-tiny.pt".
      --imgsz | --img | --img-size (list of int): Image size for inference in [height, width] format. Default is [640].
      --batch-size (int): Batch size for inference. Default is 1.
      --dynamic (bool): Enables dynamic batch size if specified.

    Returns:
      Namespace: Parsed command line arguments.

    Example:
      To run the function from the command line:
      ```python
      $ python models/tf.py --weights yolov3-tiny.pt --imgsz 640 640 --batch-size 2 --dynamic
      ```

    Notes:
      - The `--imgsz` argument can take either one value or two values. If a single value is provided, it will be duplicated to represent both height and width.
      - For available arguments and further usage, refer to the README or relevant documentation at https://github.com/ultralytics/yolov5/pull/1127.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov3-tiny.pt", help="weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch size")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Main function to execute the YOLOv3 model run with the provided options.

    Args:
        opt (argparse.Namespace): CLI argument parser with model configuration including:
            weights (str): Path to the weights file.
            imgsz (list[int]): Image size for inference [height, width].
            batch_size (int): Batch size for inference.
            dynamic (bool): Flag for dynamic batch size.

    Returns:
        None

    Example:
        ```python
        if __name__ == "__main__":
            opt = parse_opt()
            main(opt)
        ```

    Notes:
        - Ensure to provide the correct path to the weights file.
        - The image size should be specified as a list of two integers [width, height].
        - Use the `--dynamic` flag if the batch size should be determined dynamically during execution.

    References:
        - For more details, refer to: https://github.com/ultralytics/yolov5
    """
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
