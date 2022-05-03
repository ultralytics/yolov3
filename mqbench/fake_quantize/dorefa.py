import torch

from mqbench.fake_quantize.quantize_base import QuantizeBase


_version_under_1100 = int(torch.__version__.split('.')[1]) < 10

class DoReFaFakeQuantize(QuantizeBase):
    def __init__(self, observer, **observer_kwargs):
        super(DoReFaFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))

    def forward(self, X):
        X = torch.tanh(X)
        X = X.div(X.abs().max() + 1e-5)

        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, 
                    self.zero_point.long() if _version_under_1100 else self.zero_point,
                    self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale.item(), self.zero_point.item(), self.quant_min, self.quant_max)
        return X