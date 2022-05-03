from copy import deepcopy

import torch


def fuse_deconv_bn_weights(deconv_w, deconv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if deconv_b is None:
        deconv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    shape = [1] * len(deconv_w.shape)
    shape[1] = -1
    deconv_w = deconv_w * (bn_w * bn_var_rsqrt).reshape(shape)
    deconv_b = (deconv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(deconv_w), torch.nn.Parameter(deconv_b)


def fuse_deconv_bn_eval(deconv, bn):
    assert (not (deconv.training or bn.training)), 'Fusion only for eval!'

    fused_deconv = deepcopy(deconv)
    fused_deconv.weight, fused_deconv.bias = fuse_deconv_bn_weights(
        deconv.weight, deconv.bias, bn.running_mean, bn.running_var, bn.eps,
        bn.weight, bn.bias)
    return fused_deconv