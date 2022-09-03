import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()

        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:  # 'same' yes: input dim = output dim --> add a n-pixel boarder of zero
            ih, iw = x.size()[2:]

            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)

            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)

            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt

            padding = (pl, pr, pt, pb)
        else:  # 'same' no: input dim =/= output dim
            padding = self.padding
        return padding
    
    def forward(self, x):  # x = adv_patch image
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level

        # NB in F.pad (https://pytorch.org/docs/stable/nn.functional.html):
        # 2nd parameter is 'pad (tuple)', i.e. m-elements tuple, where m/2 <= input dimensions and m is even
        # here m = 4 (left, right, top, bottom)

        # PADDING:
        x = F.pad(x, self._padding(x), mode='reflect')  # put the zeros according to self._padding(x) --> l, r, t, b

        # prepare to do MEDIAN:
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])

        # NB unfold: unfold(dimension, size, step) -> Tensor
        # Returns a tensor which contains all slices of size `size` from `self` tensor along the dimension `dimension`
        # Step between two slices is given by `step`
        # Here size = kernel, step = stride, dimensions 2 and 3 respectively

        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]  #[:4] --> prob. kernel_size = 3, so I take values at pos 0, 1, 2, 3 ??
        return x