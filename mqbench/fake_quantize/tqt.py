from functools import partial

import torch

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils import is_symmetric_quant


class TqtFakeQuantize(QuantizeBase):
    def __init__(self, observer, scale=1., zero_point=0., **observer_kwargs):
        super(TqtFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([scale]))
        self.register_buffer('zero_point', torch.tensor([zero_point]))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.quant_type = None
        self.mth = None

        class PerChannelLoadHook:
            def __init__(self, module):
                self.hook = module._register_load_state_dict_pre_hook(partial(self.hook_fn, module=module))

            def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                        module):
                if module.ch_axis == -1:
                    # no per-channel parameters
                    return
                for module_key, param in module._parameters.items():
                    if module_key not in ["scale", "zero_point"]:
                        continue
                    candidate = prefix + module_key
                    if candidate in state_dict:
                        input_param = state_dict[candidate]
                        if param.shape != input_param.shape:
                            param.data = torch.ones_like(input_param, dtype=param.dtype, device=param.device)

            def close(self):
                self.hook.remove()

        self.load_state_dict_hook = PerChannelLoadHook(self)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List',
                   self.zero_point if self.ch_axis == -1 else 'List')

    def forward(self, X):
        # Learnable fake quantize have to zero_point.float() to make it learnable.
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled[0] == 1:
            assert is_symmetric_quant(self.qscheme)
            "TQT is a symmetric quantization FakeQuantize Op."
            self.zero_point.data.zero_()
            assert self.is_per_channel is False
            "TQT is a per-tensor quantization FakeQuantize Op."
            X = FakeQuantizeTqtAffine.apply(X, self.scale, self.zero_point, self.quant_min, self.quant_max, self.mth)
        return X

    def set_quant_type(self, quant_type):
        if quant_type in ['input', 'tensor', 'param']:
            self.quant_type = quant_type
            self.activation_post_process.set_quant_type(quant_type)
            self.set_forward_method()
        else:
            raise ValueError(f'The quant type {quant_type} of TQTQuantizer is not right.')

    def set_forward_method(self):
        self.mth = torch.tensor(3) if self.quant_type == 'param' else torch.tensor(2)

def _fake_quantize_tqt_affine_training(x, scale, zero_point, quant_min, quant_max, mth):
    if scale < 2 ** -15:
        max_scale = 0
    else:
        max_scale = 1 / scale
        max_scale = torch.floor(max_scale.log2())
    scale = 1 / (2 ** max_scale)
    if mth == 3:
        new_x = torch.clamp(scale_round(x / scale), quant_min, quant_max) * scale
    elif mth == 2:
        new_x = torch.clamp(x / scale, quant_min, quant_max)
        new_x = scale_floor_ceil(new_x)
        new_x *= scale
    else:
        raise ValueError(f'Invalid method {mth} encoding!')
    return new_x


def scale_round(t):
    return (torch.round(t) - t).detach() + t

def scale_floor_ceil(t):
    return (torch.where((t < 0) & (t - t.floor() == 0.5), t.ceil(), t.round()) - t).detach() + t

def _t(x, t):
    return torch.tensor(x).type_as(t)

class FakeQuantizeTqtAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max, mth):
        qx = _fake_quantize_tqt_affine_training(x, scale, zero_point, quant_min, quant_max, mth)
        ctx.save_for_backward(x, scale, _t(quant_min, x), _t(quant_max, x))
        return qx

    @staticmethod
    def backward(ctx, grad_outputs):
        x, s, qmin, qmax = ctx.saved_tensors
        scaled_x = x / s
        rounded_scaled_x = torch.where(
            (scaled_x < 0) & (scaled_x - torch.floor(scaled_x) == 0.5),
            torch.ceil(scaled_x), torch.round(scaled_x)
        )

        is_lt_min = rounded_scaled_x < qmin
        is_gt_max = rounded_scaled_x > qmax
        is_ge_min_and_le_max = ~is_lt_min & ~is_gt_max

        grad_x = grad_outputs.clone()
        grad_x = torch.where(is_ge_min_and_le_max, grad_x, 0 * grad_x)
        return grad_x.to(grad_outputs.device), None, None, None, None, None

    @staticmethod
    def symbolic(g, x, scale, zero_point, quant_min, quant_max, mth):
        return g.op("::FakeQuantizeTqtAffine", x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max)
