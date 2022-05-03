import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Linear
from torch.nn.intrinsic import _FusedModule
from torch.nn.parameter import Parameter

from mqbench.nn.intrinsic import LinearBn1d


class LinearBn1d(Linear, _FusedModule):
    _version = 2
    _FLOAT_MODULE = LinearBn1d

    def __init__(self,
                 # ConvNd args
                 in_features, out_features, bias,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        Linear.__init__(self, in_features, out_features, False)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actully for Linear, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(LinearBn1d, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self 

    def _forward(self, input):
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        # input.shape = (batch_size, in_features, *)
        # scale_factor.shape = (out_feature, )
        # self.weight.shape = (out_feature, in_feature, *)
        # self.bias.shape = (out_feature, *)
        # output.shape = (batch_size, out_feature, *)
        if self.bn.affine:
            scale_factor = self.bn.weight / running_std
        else:
            scale_factor = 1. / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(input.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original Linear
        # will be added later
        # Linear layer takes permuted input since the format is (batch_size, *, in_features)
        linear_out = F.linear(input, scaled_weight)
        linear_orig = linear_out / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            linear_orig = linear_orig + self.bias.reshape(bias_shape)
        linear_out = self.bn(linear_orig)
        return linear_out

    def extra_repr(self):
        return super(LinearBn1d, self).extra_repr()

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                'bn.weight': 'gamma',
                'bn.bias': 'beta',
                'bn.running_mean': 'running_mean',
                'bn.running_var': 'running_var',
                'bn.num_batches_tracked': 'num_batches_tracked',
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(LinearBn1d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(linear.in_features, linear.out_features, False,
                           bn.eps, bn.momentum,
                           False,
                           qconfig)
        qat_linearbn.weight = linear.weight
        qat_linearbn.bias = linear.bias
        qat_linearbn.bn.weight = bn.weight
        qat_linearbn.bn.bias = bn.bias
        qat_linearbn.bn.running_mean = bn.running_mean
        qat_linearbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        return qat_linearbn
