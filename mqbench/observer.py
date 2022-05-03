import math
from functools import partial
from typing import Tuple
from copy import deepcopy
import torch
from torch.quantization.observer import _ObserverBase

from mqbench.utils import sync_tensor, pot_quantization, is_symmetric_quant
from mqbench.utils.logger import logger


class ObserverBase(_ObserverBase):
    '''
        Support per-tensor / per-channel.
        dtype: quant min/max can be infered using dtype, we actually do not need this.
        qscheme: quantization scheme
        reduce_range: special for fbgemm to avoid overflow
        quant_min: fix point value min
        quant_max: fix point value max
        ch_axis: per-channel axis or per-tensor(-1)
        above is similiar to torch observer.
        pot_scale: indecate wheather scale is power of two.
    '''

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False,
                 factory_kwargs=None):
        factory_kwargs = deepcopy(factory_kwargs)
        self.not_calc_quant_min_max = factory_kwargs.pop('not_calc_quant_min_max', False) if isinstance(factory_kwargs, dict) else False
        super(ObserverBase, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max)
        # for compatibility with 1.10, prevent the value of self.quant_min,self.quant_max being modified
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.quant_min, self.quant_max = self._calculate_qmin_qmax()
        self.ch_axis = ch_axis
        self.pot_scale = pot_scale
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

        class PerChannelLoadHook:
            def __init__(self, module):
                self.hook = module._register_load_state_dict_pre_hook(partial(self.hook_fn, module=module))

            def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                        module):
                if module.ch_axis == -1:
                    # no per-channel parameters
                    return
                for module_key, param in module._buffers.items():
                    if module_key not in ['min_val', 'max_val']:
                        continue
                    candidate = prefix + module_key
                    if candidate in state_dict:
                        input_param = state_dict[candidate]
                        if param.shape != input_param.shape:
                            param.data = torch.ones_like(input_param, dtype=param.dtype, device=param.device)

            def close(self):
                self.hook.remove()

        self.load_state_dict_hook = PerChannelLoadHook(self)

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        if self.pot_scale:
            scale = pot_quantization(scale)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point

    @torch.jit.export
    def _calculate_qmin_qmax(self) -> Tuple[int, int]:
        r"""Calculates actual qmin and qmax based on the quantization range,
        observer datatype and if range is reduced.
        """
        if self.has_customized_qrange:
            # This initialization here is to be resolve TorchScript compilation issues and allow
            # using of refinement to decouple initial_qmin and initial_qmax from quantization range.
            # The actual values of initial_qmin and initial_qmax will be reset below.
            initial_quant_min, initial_quant_max = 0, 255
            # The following assignment of self.qmin and self.qmax to the local variables and the if check refine the
            # attribute from Optional valid integers for use, based on TorchScript's requirements.
            custom_quant_min, custom_quant_max = self.quant_min, self.quant_max
            if custom_quant_min is not None and custom_quant_max is not None:
                initial_quant_min, initial_quant_max = (
                    custom_quant_min,
                    custom_quant_max,
                )

            qrange_len = initial_quant_max - initial_quant_min + 1
            if is_symmetric_quant(self.qscheme):
                quant_min, quant_max = -qrange_len // 2, qrange_len // 2 - 1
            else:
                quant_min, quant_max = 0, qrange_len - 1
            if self.reduce_range:
                quant_min, quant_max = quant_min // 2, quant_max // 2
            if self.not_calc_quant_min_max:
                quant_min, quant_max = self.quant_min, self.quant_max
        else:
            # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
            if self.dtype == torch.qint8:
                if self.reduce_range:
                    quant_min, quant_max = -64, 63
                else:
                    quant_min, quant_max = -128, 127
            elif self.dtype == torch.quint8:
                if self.reduce_range:
                    quant_min, quant_max = 0, 127
                else:
                    quant_min, quant_max = 0, 255
            else:
                quant_min, quant_max = 0, 15
        return quant_min, quant_max

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={} ch_axis={} pot={}".format(self.min_val if self.ch_axis == -1 else 'List',
                                                                 self.max_val if self.ch_axis == -1 else 'List',
                                                                 self.ch_axis, self.pot_scale)


class MinMaxObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False,
                 factory_kwargs=None):
        super(MinMaxObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                             ch_axis, pot_scale, factory_kwargs)

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)

        return x


class MinMaxFloorObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset with floor but round.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False,
                 factory_kwargs=None):
        super(MinMaxFloorObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                  ch_axis, pot_scale, factory_kwargs)
        '''
        The quant_type could be 'input', 'param', 'tensor', the co-responding
        range is 1, 5, 5,
        mth is 2, 3, 2
        '''
        self.quant_type = None


    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            logger.warn('The per-tensor observer does not support per-channel min-max!')
            min_val_cur, max_val_cur = torch._aminmax(x)

        self.min_val = min_val_cur
        self.max_val = max_val_cur
        self._x = x
        return x

    def calculate_qparams(self):
        if self.quant_type is None:
            raise ValueError('You should set the observer type before forward!')
        else:
            scale_range = 1 if self.quant_type == 'input' else 5
            mth = 3 if self.quant_type == 'param' else 2
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = scale.data * 0 + max(self.min_val / self.quant_min, self.max_val / self.quant_max)
        if scale < 2 ** -15:
            max_scale = 0
        else:
            max_scale = 1 / scale
            max_scale = torch.floor(max_scale.log2())
        min_loss = torch.tensor([float('inf')])
        final_scale = max_scale
        max_scale = int(max_scale)
        for s in range(max_scale, max_scale + scale_range):
            _s = 1 / 2 ** s
            if mth == 3:
                new_x = _s * torch.clamp(torch.round(self._x / _s), self.quant_min, self.quant_max)
            elif mth == 2:
                new_x = torch.clamp(self._x / _s, self.quant_min, self.quant_max)
                new_x = torch.where((new_x < 0) & (new_x - new_x.floor() == 0.5), new_x.ceil(), new_x.round())
                new_x *= _s
            loss = ((new_x - self._x)**2).sum()
            min_loss = min_loss.to(loss.device)
            if loss < min_loss:
                min_loss = loss
                final_scale = s
        final_scale = min(final_scale, 12)
        scale = scale.data * 0 + 1 / (2 ** final_scale)
        zero_point = torch.zeros_like(zero_point)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point

    def set_quant_type(self, qtype):
        self.quant_type = qtype


class EMAMinMaxObserver(ObserverBase):
    """Moving average min/max among batches.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, ema_ratio=0.9,
                 factory_kwargs=None):
        super(EMAMinMaxObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                ch_axis, pot_scale, factory_kwargs)
        self.ema_ratio = ema_ratio

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (1.0 - self.ema_ratio)
        return x


class PoTModeObserver(ObserverBase):
    r"""Records the most frequent Potscale of ``x``."""
    """
    Borrow from vitis
    https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_pytorch/pytorch_binding/pytorch_nndct/quantization/torchquantizer.py
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, factory_kwargs=None):
        super(PoTModeObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max, ch_axis, pot_scale, factory_kwargs)
        self.quant_type = None
        self.counter = [0] * 20

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            logger.warn('The per-tensor observer does not support per-channel min-max!')
            min_val_cur, max_val_cur = torch._aminmax(x)

        self.min_val = min_val_cur
        self.max_val = max_val_cur
        self._x = x
        return x

    def calculate_qparams(self):
        if self.quant_type is None:
            raise ValueError('You should set the observer type before forward!')
        else:
            scale_range = 1 if self.quant_type == 'input' else 5
            mth = 3 if self.quant_type == 'param' else 2
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = scale.data * 0 + max(self.min_val / self.quant_min, self.max_val / self.quant_max)
        if scale < 2 ** -15:
            max_scale = 0
        else:
            max_scale = 1 / scale
            max_scale = torch.floor(max_scale.log2())
        min_loss = torch.tensor([float('inf')])
        final_scale = max_scale
        max_scale = int(max_scale)
        for s in range(max_scale, max_scale + scale_range):
            _s = 1 / 2 ** s
            if mth == 3:
                new_x = _s * torch.clamp(torch.round(self._x / _s), self.quant_min, self.quant_max)
            elif mth == 2:
                new_x = torch.clamp(self._x / _s, self.quant_min, self.quant_max)
                new_x = torch.where((new_x < 0) & (new_x - new_x.floor() == 0.5), new_x.ceil(), new_x.round())
                new_x *= _s
            loss = ((new_x - self._x)**2).sum()
            min_loss = min_loss.to(loss.device)
            if loss < min_loss:
                min_loss = loss
                final_scale = s
        final_scale = min(final_scale, 12)
        self.counter[final_scale + 7] += 1
        final_scale = self.counter.index(max(self.counter)) - 7
        scale = scale.data * 0 + 1 / (2 ** final_scale)
        zero_point = torch.zeros_like(zero_point)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point

    def set_quant_type(self, qtype):
        self.quant_type = qtype


class EMAQuantileObserver(ObserverBase):
    """Moving average quantile among batches.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, ema_ratio=0.9,
                 threshold=0.99999, bins=2048, factory_kwargs=None):
        super(EMAQuantileObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                  ch_axis, pot_scale, factory_kwargs)
        assert self.ch_axis == -1, "Quantile observer only support in per-tensor scheme."
        self.ema_ratio = ema_ratio
        self.threshold = threshold
        self.bins = bins

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(torch.abs(x), bins=self.bins, min=0., max=max_hist_range)
        cur_total = 0
        clip_value = max_hist_range
        for i, cnt in enumerate(hist):
            if cur_total + cnt >= self.threshold * x.numel():
                clip_value = (i + 0.5) * (max_hist_range / self.bins)
                break
            cur_total += cnt

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = max(min_val_cur, -clip_value)
            self.max_val = min(max_val_cur, clip_value)
        else:
            self.min_val = self.min_val * self.ema_ratio + max(min_val_cur, -clip_value) * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + min(max_val_cur, clip_value) * (1.0 - self.ema_ratio)
        return x


class ClipStdObserver(ObserverBase):
    """Clip std.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, std_scale=2.6,
                 factory_kwargs=None):
        super(ClipStdObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                              ch_axis, pot_scale, factory_kwargs=None)
        self.std_scale = std_scale

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            mean = x.mean()
            std = x.std()
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            mean = y.mean(1)
            std = y.std(1)

        # using statistics to clip min and max
        min_val = torch.minimum(mean - self.std_scale * std, min_val_cur)
        max_val = torch.maximum(mean + self.std_scale * std, max_val_cur)

        self.min_val = min_val
        self.max_val = max_val

        return x


class LSQObserver(ObserverBase):
    '''
    LSQ observer.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, factory_kwargs=None):
        super(LSQObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                          ch_axis, pot_scale, factory_kwargs)
        self.tensor_norm = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.tensor_norm = x.abs().mean()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.tensor_norm = y.abs().mean(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self):
        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        if self.pot_scale:
            scale = pot_quantization(scale)
        zero_point = torch.zeros_like(self.tensor_norm)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point


class LSQPlusObserver(ObserverBase):
    '''
    LSQ+ observer.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, factory_kwargs=None):

        super(LSQPlusObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                              ch_axis, pot_scale, factory_kwargs)
        self.mean = None
        self.std = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.mean = x.mean()
            self.std = x.std()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.mean = y.mean(1)
            self.std = y.std(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self):
        scale = torch.maximum((self.mean - 3 * self.std).abs(),
                              (self.mean + 3 * self.std).abs()) / (self.quant_max - self.quant_min + 1)
        if self.pot_scale:
            scale = pot_quantization(scale)
        zero_point = torch.zeros_like(self.mean)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point


class MSEObserver(ObserverBase):
    '''
    Calculate mseobserver of whole calibration dataset.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, p=2.0,
                 factory_kwargs=None):
        super(MSEObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                          ch_axis, pot_scale, factory_kwargs)
        self.p = p

    def lp_loss(self, pred, tgt):
        """
        loss function measured in L_p Norm
        """
        return (pred - tgt).abs().pow(self.p).mean()

    def mse(self, x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, iter=80):
        best_score = 1e+10
        best_min, best_max = torch.tensor([1.0], dtype=torch.float), torch.tensor([1.0], dtype=torch.float)
        best_min.copy_(x_min)
        best_max.copy_(x_max)
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()),
                self.quant_min, self.quant_max)
            score = self.lp_loss(x_q, x)
            if score < best_score:
                best_score = score
                best_min, best_max = new_min, new_max
        return best_min, best_max

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val_cur, max_val_cur = self.mse(x, min_val_cur, max_val_cur, iter=95)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            for ch, val in enumerate(min_val_cur):
                min_val_cur[ch], max_val_cur[ch] = self.mse(x_channel[ch], min_val_cur[ch], max_val_cur[ch], iter=80)

        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)
        return x


class EMAMSEObserver(ObserverBase):
    '''
    Calculate mseobserver of whole calibration dataset.
    '''
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False,
                 p=2.0, ema_ratio=0.9, factory_kwargs=None):
        super(EMAMSEObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                             ch_axis, pot_scale, factory_kwargs)
        self.ema_ratio = ema_ratio
        self.p = p

    def lp_loss(self, pred, tgt):
        """
        loss function measured in L_p Norm
        """
        return (pred - tgt).abs().pow(self.p).mean()

    def mse(self, x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, iter=80):
        best_score = 1e+10
        best_min, best_max = torch.tensor([1.0], dtype=torch.float), torch.tensor([1.0], dtype=torch.float)
        best_min.copy_(x_min)
        best_max.copy_(x_max)
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()),
                self.quant_min, self.quant_max)
            score = self.lp_loss(x_q, x)
            if score < best_score:
                best_score = score
                best_min, best_max = new_min, new_max
        return best_min, best_max

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val_cur, max_val_cur = self.mse(x, min_val_cur, max_val_cur, iter=95)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            for ch, val in enumerate(min_val_cur):
                min_val_cur[ch], max_val_cur[ch] = self.mse(x_channel[ch], min_val_cur[ch],
                                                            max_val_cur[ch], iter=80)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (1.0 - self.ema_ratio)
        return x
