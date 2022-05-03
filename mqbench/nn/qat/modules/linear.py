import torch.nn.qat.modules as nnqat
import torch.nn.functional as F


class Linear(nnqat.Linear):
    def __init__(self, in_features, out_features, bias=True, qconfig=None, device=None, dtype=None):
        assert hasattr(qconfig, 'bias'), 'The qconfig should provide bias observer settings for the QAT module!'
        super().__init__(in_features, out_features, bias=bias, qconfig=qconfig, device=device, dtype=dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.bias_fake_quant = qconfig.bias(**factory_kwargs)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias_fake_quant(self.bias)) 
