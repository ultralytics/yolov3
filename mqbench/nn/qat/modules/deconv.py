import torch.nn as nn 
import torch.nn.functional as F

from mqbench.nn.intrinsic import ConvTransposeReLU2d


class ConvTranspose2d(nn.ConvTranspose2d):
    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, output_padding=0, 
                 groups=1, bias=True, dilation=1,
                 padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding, output_padding=output_padding,
                         groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig 
        self.weight_fake_quant = qconfig.weight()
        # ConvTranspose do per-channel quantize on output channel.
        if self.weight_fake_quant.ch_axis != -1:
            self.weight_fake_quant.ch_axis = 1
            self.weight_fake_quant.activation_post_process.ch_axis = 1

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )
        return F.conv_transpose2d(
            x, self.weight_fake_quant(self.weight), self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__ 
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        if type(mod) == ConvTransposeReLU2d:
            mod = mod[0]
        qconfig = mod.qconfig
        qat_deconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, 
                         stride=mod.stride, padding=mod.padding, output_padding=mod.output_padding,
                         groups=mod.groups, bias=mod.bias is not None, dilation=mod.dilation,
                         padding_mode=mod.padding_mode, qconfig=qconfig) 
        qat_deconv.weight = mod.weight
        qat_deconv.bias = mod.bias
        return qat_deconv