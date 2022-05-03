import math

import torch

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils import is_tracing_state


def dsq_function_per_tensor(x, scale, zero_point, quant_min, quant_max, alpha):
    tanh_scale = 1 / (1 - alpha)
    tanh_k = math.log((tanh_scale + 1) / (tanh_scale - 1))

    x = x / scale + zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = x.floor() + (tanh_scale * torch.tanh(tanh_k * (x - x.floor() - 0.5))) * 0.5 + 0.5
    x = (x.round() - x).detach() + x
    x = (x - zero_point) * scale

    return x


def dsq_function_per_channel(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):

    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)

    tanh_scale = 1 / (1 - alpha)
    tanh_k = math.log((tanh_scale + 1) / (tanh_scale - 1))

    x = x / scale + zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = x.floor() + (tanh_scale * torch.tanh(tanh_k * (x - x.floor() - 0.5))) * 0.5 + 0.5
    x = (x.round() - x).detach() + x
    x = (x - zero_point) * scale

    return x


class DSQFakeQuantize(QuantizeBase):
    def __init__(self, observer, alpha=0.4, **observer_kwargs):
        super(DSQFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.alpha = alpha

    def forward(self, X):
        if self.training:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point.float())

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                if is_tracing_state():
                    X = FakeQuantizeDSQPerchannel.apply(
                        X, self.scale, self.zero_point, self.quant_min, self.quant_max, self.ch_axis, self.alpha)
                else:
                    X = dsq_function_per_channel(
                        X, self.scale, self.zero_point, self.quant_min, self.quant_max, self.ch_axis, self.alpha)
            else:
                if is_tracing_state():
                    X = FakeQuantizeDSQPertensor.apply(
                        X, self.scale, self.zero_point, self.quant_min, self.quant_max, self.alpha)
                else:
                    X = dsq_function_per_tensor(
                        X, self.scale, self.zero_point, self.quant_min, self.quant_max, self.alpha)

        return X


class FakeQuantizeDSQPerchannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):
        return dsq_function_per_channel(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha)

    @staticmethod
    def symbolic(g, x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):
        return g.op("::FakeQuantizeDSQPerchannel", x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max, alpha_f=alpha)


class FakeQuantizeDSQPertensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max, alpha):
        return dsq_function_per_tensor(x, scale, zero_point, quant_min, quant_max, alpha)

    @staticmethod
    def symbolic(g, x, scale, zero_point, quant_min, quant_max, alpha):
        return g.op("::FakeQuantizeDSQPertensor", x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max, alpha_f=alpha)
