'''this is for activation quantizer in BRECQ and QDrop'''
import torch
from torch.nn.parameter import Parameter
from mqbench.fake_quantize.quantize_base import QuantizeBase


class QDropFakeQuantize(QuantizeBase):
    """This is based on the fixedfakequantize.
    And we wrap scale as parameter, where BRECQ and QDrop both learn the scale.
    """

    def __init__(self, observer, **observer_kwargs):
        super(QDropFakeQuantize, self).__init__(observer, **observer_kwargs)
        # self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.scale = Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.prob = 1.0  # 1.0 means no drop;

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.resize_(_zero_point.shape)

            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            x_orig = X
            if self.is_per_channel:
                X = _fake_quantize_learnable_per_channel_affine_training(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max)
            else:
                X = _fake_quantize_learnable_per_tensor_affine_training(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max)
            if self.prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.prob, X, x_orig)
                return x_prob
        return X

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale.data if self.ch_axis == -1 else 'List',
                   self.zero_point if self.ch_axis == -1 else 'List')


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def _fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def _fake_quantize_learnable_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant
