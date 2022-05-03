import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Embedding):
    r"""
    We release the restrict of scheme type.
    TODO: Delete this module since this project support torch1.10.

    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.
    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
    for documentation.
    Similar to `torch.nn.Embedding`, with FakeQuantize modules initialized to
    default.
    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Embedding

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, device=None, dtype=None, qconfig=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight,
                         **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'

        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)
        # Embedding do per-channel quantize on embedding channel.
        if self.weight_fake_quant.ch_axis != -1:
            self.weight_fake_quant.ch_axis = 1
            self.weight_fake_quant.activation_post_process.ch_axis = 1

    def forward(self, input) -> Tensor:
        return F.embedding(input, self.weight_fake_quant(self.weight), self.padding_idx,
                           self.max_norm, self.norm_type, self.scale_grad_by_freq,
                           self.sparse)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_embedding_bag = cls(mod.num_embeddings, mod.embedding_dim, mod.padding_idx,
                                mod.max_norm, mod.norm_type, mod.scale_grad_by_freq,
                                mod.sparse, mod.weight, qconfig=qconfig)

        return qat_embedding_bag

    def to_float(self):
        embedding_bag = torch.nn.Embedding(self.num_embeddings, self.embedding_dim, self.padding_idx,
                                           self.max_norm, self.norm_type, self.scale_grad_by_freq,
                                           self.sparse, None, self.device, self.dtype)
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag