# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Activation functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    # SiLU activation https://arxiv.org/pdf/1606.08415.pdf
    @staticmethod
    def forward(x):
        """
        Applies the SiLU (Sigmoid Linear Unit) activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor to apply the SiLU activation function.

        Returns:
            torch.Tensor: Tensor with the SiLU activation function applied to each element.

        References:
            For more details on the SiLU activation function, refer to
            [SiLU: Sigmoid-Weighted Linear Units for Neural Network Function Approximation](https://arxiv.org/pdf/1606.08415.pdf).

        Examples:
            ```python
            import torch
            from ultrainytics import SiLU

            input_tensor = torch.randn(3, 3)
            silu = SiLU()
            output_tensor = silu.forward(input_tensor)
            ```
        """
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    # Hard-SiLU activation
    @staticmethod
    def forward(x):
        """
        Applies the Hardswish activation function, modifying the input tensor `x` as per the Hard-SiLU definition.

        Args:
            x (torch.Tensor): Input tensor to which the Hardswish activation function is applied.

        Returns:
            torch.Tensor: Tensor after applying the Hardswish activation function.

        Notes:
            The Hardswish activation function is designed to offer a computationally efficient approximation to
            the SiLU (Sigmoid Linear Unit) activation, making it suitable for various deep learning frameworks
            such as TorchScript, CoreML, and ONNX.
        """
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX


class Mish(nn.Module):
    # Mish activation https://github.com/digantamisra98/Mish
    @staticmethod
    def forward(x):
        """
        Applies the Mish activation function to the input tensor `x`, enhancing model performance and convergence.

        Args:
            x (torch.Tensor): The input tensor upon which the Mish activation function will be applied.

        Returns:
            torch.Tensor: A tensor with the Mish activation applied.

        Reference:
            https://github.com/digantamisra98/Mish

        Example:
            ```python
            import torch
            from ultralytics import Mish

            x = torch.tensor([1.0, -1.0, 0.0])
            activated_x = Mish.forward(x)
            print(activated_x)
            ```
        """
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    # Mish activation memory-efficient
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            """
            Applies the Mish activation function in a memory-efficient manner.

            Args:
                ctx (torch.autograd.function.FunctionCtx): Context object used for storing information for backward computation.
                x (torch.Tensor): Input tensor on which the Mish activation function is to be applied.

            Returns:
                torch.Tensor: Tensor after applying the Mish activation function in a memory-efficient manner.
            """
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            """
            Computes gradient of the Mish activation function for backpropagation, returning the derivative with respect
            to the input.

            Args:
                ctx (torch.autograd.function.FunctionCtx): Context object containing saved tensors from the forward pass.
                grad_output (torch.Tensor): Gradient of the loss with respect to the output of the forward pass.

            Returns:
                torch.Tensor: Gradient of the loss with respect to the input of the forward pass.

            Notes:
                This method ensures that the gradients are computed in a memory-efficient manner, compatible with the
                autograd system in PyTorch, enabling efficient backpropagation in deep neural networks utilizing the
                Mish activation function.
            """
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        """
        Applies the Mish activation function in a memory-efficient manner, optimizing neural network performance.

        Args:
            x (torch.Tensor): Input tensor to which the activation function will be applied.

        Returns:
            torch.Tensor: Tensor after applying the Mish activation function.

        Example:
            ```python
            import torch
            from ultralytics import MemoryEfficientMish

            data = torch.randn(1, 3, 224, 224)
            activation = MemoryEfficientMish()
            result = activation(data)
            ```

        Note:
            This function is useful for enhancing model performance and convergence by effectively introducing non-linearities.
        """
        return self.F.apply(x)


class FReLU(nn.Module):
    # FReLU activation https://arxiv.org/abs/2007.11824
    def __init__(self, c1, k=3):  # ch_in, kernel
        """
        Initializes the FReLU activation function with the specified number of input channels and kernel size.

        Args:
            c1 (int): Number of input channels.
            k (int, optional): Kernel size for the depthwise convolution. Default is 3.

        Returns:
            None

        Note:
            This activation function is based on the approach proposed in "Funnel Activation for Visual Recognition",
            which can be accessed at https://arxiv.org/abs/2007.11824.

        Examples:
            ```python
            frelu = FReLU(64, k=3)
            ```
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """
        Performs FReLU activation on the input tensor, combining convolution and ReLU operations.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is batch size, C is channel size, H is height, and W is width.

        Returns:
            torch.Tensor: Output tensor of the same shape as input (N, C, H, W) after applying FReLU activation.

        Reference:
            https://arxiv.org/abs/2007.11824

        Example:
            ```python
            import torch
            from ultralytics import FReLU

            x = torch.randn(1, 16, 64, 64)  # Example input tensor
            frelu = FReLU(c1=16)  # Initialize FReLU with 16 channels
            output = frelu.forward(x)  # Apply FReLU activation
            print(output.shape)  # Output tensor shape
            ```
        """
        return torch.max(x, self.bn(self.conv(x)))


class AconC(nn.Module):
    r"""ACON activation (activate or not)
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1):
        """
        Initializes the ACON activation with learnable parameters p1, p2, and beta for custom activation functions.

        Args:
          c1 (int): Number of input channels.

        Returns:
          None

        Notes:
          The ACON activation function implements the formula: (p1*x - p2*x) * sigmoid(beta*(p1*x - p2*x)) + p2*x, where
          beta is a learnable parameter. This is based on the "Activate or Not: Learning Customized Activation" paper,
          available at: https://arxiv.org/pdf/2009.04759.pdf.
        """
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        """
        Provides a parameterized activation function `AconC` that applies a custom activation mechanism on the input
        tensor `x`.

        Args:
            x (torch.Tensor): Input tensor to which the ACON activation function will be applied.

        Returns:
            torch.Tensor: Activated tensor after applying the ACON activation function as defined in the paper https://arxiv.org/pdf/2009.04759.pdf.

        Notes:
            - The function performs the activation as specified:
              \( (p1 * x - p2 * x) * \sigma(\beta * (p1 * x - p2 * x)) + p2 * x \)
            - Here, `p1`, `p2`, and `beta` are learnable parameters that are initialized during the creation of the `AconC` layer.

        Example:
            ```python
            import torch
            from ultralytics import AconC

            aconc_layer = AconC(c1=64)
            input_tensor = torch.randn(1, 64, 32, 32)
            output_tensor = aconc_layer(input_tensor)
            print(output_tensor.shape)  # should print torch.Size([1, 64, 32, 32])
            ```
        """
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    r"""ACON activation (activate or not)
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1, k=1, s=1, r=16):  # ch_in, kernel, stride, r
        """
        Initializes the MetaAconC activation function with learnable parameters, where the beta parameter is generated by a small internal neural network, as described in https://arxiv.org/pdf/2009.04759.pdf.

        Args:
            c1 (int): The number of input channels.
            k (int, optional): The kernel size of the intermediate fully connected layer. Defaults to 1.
            s (int, optional): The stride of the intermediate fully connected layer. Defaults to 1.
            r (int, optional): The reduction factor for computing the number of output channels of the intermediate layer. Defaults to 16.

        Returns:
            None: This method initializes the MetaAconC module and sets up its internal layers and parameters.

        Example:
            ```python
            import torch
            from ultralytics import MetaAconC

            # Creating an instance of MetaAconC with 64 input channels
            activation = MetaAconC(64)

            # Applying the activation to an input tensor
            x = torch.randn(1, 64, 8, 8)
            output = activation(x)
            ```
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
        """
        Applies the MetaACON activation function to the input tensor `x`, utilizing a small network for dynamic
        parameter generation.

        Args:
            x (torch.Tensor): Input tensor to be activated, typically of shape `(batch_size, channels, height, width)`.

        Returns:
            torch.Tensor: Activated tensor of the same shape as the input. The activation function is defined as
            `(p1 * x - p2 * x) * sigmoid(beta * (p1 * x - p2 * x)) + p2 * x`, where `beta` is dynamically generated by
            a small network.

        Notes:
            This implementation follows the paper "Activate or Not: Learning Customized Activation"
            (https://arxiv.org/pdf/2009.04759.pdf). It includes a bug fix that removes the batch normalization layers to
            address instabilities when the batch size is 1.

        Example:
            ```python
            import torch
            from ultralytics.nn.modules import MetaAconC

            meta_aconc = MetaAconC(c1=128)
            x = torch.randn(4, 128, 32, 32)  # Example input tensor
            output = meta_aconc(x)
            print(output.shape)  # Should output torch.Size([4, 128, 32, 32])
            ```
        """
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x
