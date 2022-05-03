import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.quantize_base import QuantizeBase


_version_under_1100 = int(torch.__version__.split('.')[1]) < 10

def _rectified_sigmoid(alpha, zeta, gamma):
    """Function to generate rounding mask.

    Args:
        x (torch.Tensor):
        zeta (torch.Tensor):
        gamma (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    return ((zeta - gamma) * torch.sigmoid(alpha) + gamma).clamp(0, 1)


def adaround_forward(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha, zeta, gamma, hard_value=False):
    if ch_axis != -1:
        new_shape = [1] * len(x.shape)
        new_shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(new_shape)
        zero_point = zero_point.reshape(new_shape)
    x = torch.floor(x / scale)
    if hard_value:
        x += (alpha >= 0).float()
    else:
        x += _rectified_sigmoid(alpha, zeta, gamma)
    x += zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = (x - zero_point) * scale
    return x


class AdaRoundFakeQuantize(QuantizeBase):
    """This is based on the fixedpointquantize. Because adaround only works at FC and Conv, there is an extra variables
    to define the state and could only serve as weight quantizer.
    self.adaround basicquantize (False) adaroundquantize(True)
    """

    def __init__(self, observer, **observer_kwargs):
        super(AdaRoundFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.adaround = False

    def init(self, weight_tensor: torch.Tensor, round_mode='learned_hard_sigmoid', ):
        self.adaround = True
        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1
        self.round_mode = round_mode

        # self.soft_targets = False  # delete this
        self.gamma, self.zeta = -0.1, 1.1
        self.init_alpha(x=weight_tensor.data.clone())

    def init_alpha(self, x: torch.Tensor):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = Parameter(alpha)
        else:
            raise NotImplementedError

    def rectified_sigmoid(self):
        """Function to generate rounding mask.

        Args:
            x (torch.Tensor):
            zeta (torch.Tensor):
            gamma (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)

    def get_hard_value(self, X):
        X = adaround_forward(X, self.scale.data, self.zero_point.data.long(), self.quant_min,
                             self.quant_max, self.ch_axis, self.alpha, self.zeta, self.gamma, hard_value=True)
        return X

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if not self.adaround:
                if self.is_per_channel:
                    X = torch.fake_quantize_per_channel_affine(
                        X, self.scale,
                        self.zero_point.long() if _version_under_1100 else self.zero_point,
                        self.ch_axis, self.quant_min, self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                        X, self.scale.item(), self.zero_point.item(),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = adaround_forward(X, self.scale.data, self.zero_point.data.long(), self.quant_min,
                                         self.quant_max, self.ch_axis, self.alpha, self.zeta, self.gamma)
                else:
                    raise NotImplementedError
        return X

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List',
                   self.zero_point if self.ch_axis == -1 else 'List')
