import torch
from spring.linklink.nn import SyncBatchNorm2d


def replace_bn_to_syncbn(model, custombn=SyncBatchNorm2d):
    if type(model) in [torch.nn.BatchNorm2d]:
        return _replace_bn(model, custombn)

    elif type(model) in [torch.nn.intrinsic.qat.ConvBn2d, torch.nn.intrinsic.qat.ConvBnReLU2d]:
        model.bn = _replace_bn(model.bn, custombn)
        return model

    elif type(model) in [torch.nn.intrinsic.BNReLU2d]:
        model[0] = _replace_bn(model[0], custombn)
        return model

    else:
        for name, module in model.named_children():
            setattr(model, name, replace_bn_to_syncbn(module))
        return model


def _replace_bn(bn, custombn):
    syncbn = custombn(bn.num_features, bn.eps, bn.momentum, bn.affine)
    if bn.affine:
        syncbn.weight = bn.weight
        syncbn.bias = bn.bias
    syncbn.running_mean = bn.running_mean
    syncbn.running_var = bn.running_var
    return syncbn
