import torch.nn.qat.modules as nnqat

from mqbench.quantization.default_bias_fake_quant import bias_fake_quantizer

class Conv2d(nnqat.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', qconfig=None, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig) 
        self.bias_fake_quant = bias_fake_quantizer()

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias_fake_quant(self.bias)) 
