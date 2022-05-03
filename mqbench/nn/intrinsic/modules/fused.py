from torch.nn.intrinsic import _FusedModule
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, ReLU, ConvTranspose2d, Conv2d
from mqbench.nn.modules import FrozenBatchNorm2d

class LinearBn1d(_FusedModule):
    r"""This is a sequential container which calls the Linear and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, bn):
        assert type(linear) == Linear and type(bn) == BatchNorm1d, \
            'Incorrect types for input modules{}{}'.format(
                type(linear), type(bn))
        super().__init__(linear, bn) 

class ConvTransposeBn2d(_FusedModule):
    def __init__(self, deconv, bn):
        assert type(deconv) == ConvTranspose2d and type(bn) == BatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(deconv), type(bn))
        super().__init__(deconv, bn)


class ConvTransposeBnReLU2d(_FusedModule):
    def __init__(self, deconv, bn, relu):
        assert type(deconv) == ConvTranspose2d and type(bn) == BatchNorm2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}{}'.format(
                type(deconv), type(bn), type(relu))
        super().__init__(deconv, bn, relu)


class ConvTransposeReLU2d(_FusedModule):
    def __init__(self, deconv, relu):
        assert type(deconv) == ConvTranspose2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(deconv), type(relu))
        super().__init__(deconv, relu)
class ConvBn2d(_FusedModule):
    def __init__(self, conv, bn):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super().__init__(conv, bn)


class ConvBnReLU2d(_FusedModule):
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}{}'.format(
                type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)


class ConvReLU2d(_FusedModule):
    def __init__(self, conv, relu):
        assert type(conv) == Conv2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super().__init__(conv, relu)


class ConvFreezebn2d(_FusedModule):
    def __init__(self, conv, bn):
        assert type(conv) == Conv2d and type(bn) == FrozenBatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super().__init__(conv, bn)

class ConvFreezebnReLU2d(_FusedModule):
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv2d and type(bn) == FrozenBatchNorm2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}{}'.format(
                type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)

class ConvTransposeFreezebn2d(_FusedModule):
    def __init__(self, deconv, bn):
        assert type(deconv) == ConvTranspose2d and type(bn) == FrozenBatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(deconv), type(bn))
        super().__init__(deconv, bn)

class ConvTransposeFreezebnReLU2d(_FusedModule):
    def __init__(self, deconv, bn, relu):
        assert type(deconv) == ConvTranspose2d and type(bn) == FrozenBatchNorm2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}{}'.format(
                type(deconv), type(bn), type(relu))
        super().__init__(deconv, bn, relu)
