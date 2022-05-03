from torch.onnx import register_custom_op_symbolic

# Register symbolic op for torch.quantize_function op.

def _fake_quantize_learnable_per_tensor_affine(g, x, scale, zero_point, quant_min, quant_max, grad_factor):
    return g.op("::LearnablePerTensorAffine", x, scale, zero_point, quant_min, quant_max)


register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine', _fake_quantize_learnable_per_tensor_affine, 11)


def fake_quantize_per_channel_affine(g, x, scale, zero_point, ch_axis, quant_min, quant_max):
    return g.op("::FixedPerChannelAffine", x, scale, zero_point, ch_axis, quant_min, quant_max)


register_custom_op_symbolic('::fake_quantize_per_channel_affine', fake_quantize_per_channel_affine, 11)


def fake_quantize_per_tensor_affine(g, x, scale, zero_point, quant_min, quant_max):
    return g.op("::FixedPerTensorAffine", x, scale, zero_point, quant_min, quant_max)


register_custom_op_symbolic('::fake_quantize_per_tensor_affine', fake_quantize_per_tensor_affine, 11)