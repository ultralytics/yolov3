import torch

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils import no_jit_trace


class NNIEFakeQuantize(QuantizeBase):
    def __init__(self, observer, **observer_kwargs):
        super(NNIEFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('data_max', torch.tensor(float('-inf')))

    def forward(self, X):
        with no_jit_trace():
            if self.observer_enabled[0] == 1:
                self.activation_post_process(X.detach())
            data_max = torch.max(-self.activation_post_process.min_val, self.activation_post_process.max_val)
            self.data_max = torch.max(data_max, self.data_max)
        if self.fake_quant_enabled[0] == 1:
            X = NNIEQuantizeFunc.apply(X, self.data_max)
        return X


class NNIEQuantizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_max):
        z = (16 * torch.log2(data_max.double())).round() - 127
        x = x.double()
        pos_idx = x > 2 ** ((z - 16) / 16)
        neg_idx = x < - 2 ** ((z + 1 - 16) / 16)
        zero_idx = (x >= - 2 ** ((z + 1 - 16) / 16)) & (x < 2 ** ((z - 16) / 16))
        x[zero_idx] = 0
        x[pos_idx] = 2 ** ((torch.clamp(torch.round(16 * torch.log2(x[pos_idx]) - z), 0, 127) + z) / 16)
        x[neg_idx] = - 2 ** ((torch.clamp(torch.round(16 * torch.log2(-x[neg_idx]) - z), 1, 127) + z) / 16)
        x = x.float()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None

    @staticmethod
    def symbolic(g, x, data_max):
        return g.op("::NNIEQuantize", x, data_max)