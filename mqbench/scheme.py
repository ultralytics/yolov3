import torch 


class QuantizeScheme(object):
    """Describe quantization scheme.
    """
    def __init__(self, symmetry=True, per_channel=False, pot_scale=False, bit=8, **kwargs):
        self.symmetry = symmetry
        self.per_channel = per_channel
        self.pot_scale = pot_scale
        self.bit = bit
        if self.per_channel:
            self.torch_qscheme = torch.per_channel_symmetric if self.symmetry else torch.per_channel_affine
        else:
            self.torch_qscheme = torch.per_tensor_symmetric if self.symmetry else torch.per_tensor_affine
        self.kwargs = kwargs

    def to_observer_params(self):
        naive_para = {
            'quant_min': -2 ** (self.bit - 1) if self.symmetry else 0,
            'quant_max': 2 ** (self.bit - 1) - 1 if self.symmetry else 2 ** self.bit - 1,
            'dtype': torch.qint8 if self.symmetry else torch.quint8,
            'pot_scale': self.pot_scale,
            'qscheme': self.torch_qscheme,
            'reduce_range': False,
            'ch_axis': 0 if self.per_channel else -1
        }
        naive_para.update(self.kwargs)
        return naive_para

    def __str__(self):
        return "Symmetric: {} / Bitwidth: {} / Per channel: {} / Pot scale: {} / Extra kwargs: {}".format(
            self.symmetry, self.bit, self.per_channel, self.pot_scale, self.kwargs)
