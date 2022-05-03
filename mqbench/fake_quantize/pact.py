import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.quantize_base import QuantizeBase


class PACTFakeQuantize(QuantizeBase):
    def __init__(self, observer, alpha=6.0, **observer_kwargs):
        super(PACTFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.alpha = Parameter(torch.tensor([alpha]))
        if not self.is_symmetric_quant:
            self.n_alpha = Parameter(torch.tensor([-alpha]))
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'alpha={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.alpha)

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            X = torch.where(X > self.alpha, self.alpha, X)
            self.activation_post_process.max_val.data.fill_(self.alpha.data[0])
            if X.min() < 0:
                if self.is_symmetric_quant:
                    X = torch.where(X < -self.alpha, -self.alpha, X)
                    self.activation_post_process.min_val.data.fill_(-self.alpha[0].data)
                else:
                    X = torch.where(X < self.n_alpha, self.n_alpha, X)
                    self.activation_post_process.min_val.data.fill_(self.n_alpha[0].data)
            else:
                self.activation_post_process.min_val.data.fill_(0.)

            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            X = torch.fake_quantize_per_tensor_affine(
                X, self.scale.item(), self.zero_point.item(), self.quant_min, self.quant_max)

        return X