import numpy as np
import onnx
import os

from mqbench.utils.logger import logger
from onnx import numpy_helper
from .deploy_onnx_qnn import ONNXQNNPass, FAKE_QUANTIZE_OP
from .common import parse_attrs, prepare_data, prepare_initializer



class ONNXQLinearPass(ONNXQNNPass):
    def __init__(self, onnx_model_path):
        super(ONNXQLinearPass, self).__init__(onnx_model_path)
        self.onnx_model_path = onnx_model_path

    def parse_qparams(self, node, name2data):
        tensor_name, scale, zero_point = node.input[:3]
        scale, zero_point = name2data[scale], name2data[zero_point]
        if len(node.input) > 3:
            qmin, qmax = node.input[-2:]
            qmin, qmax = name2data[qmin], name2data[qmax]
        elif len(node.attribute) > 0:
            qparams = parse_attrs(node.attribute)
            qmin = qparams['quant_min']
            qmax = qparams['quant_max']
        else:
            logger.info(f'qmin and qmax are not found for <{node.name}>!')
        return tensor_name, scale, zero_point, qmin, qmax

    def clip_weight(self, node, name2data, named_initializer):
        tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
        data = name2data[tensor_name]
        clip_range_min = (qmin - zero_point) * scale
        clip_range_max = (qmax - zero_point) * scale
        if scale.shape[0] > 1:
            new_data = []
            next_node = self.onnx_model.get_tensor_consumer(node.output[0])[0]
            if next_node.op_type == 'ConvTranspose':
                for c in range(data.shape[1]):
                    new_data.append(np.clip(data[:, c], clip_range_min[c], clip_range_max[c]))
            else:
                for c in range(data.shape[0]):
                    new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
            new_data = np.array(new_data)
            logger.info(f'Clip weights <{tensor_name}> to per-channel ranges.')
        else:
            new_data = np.clip(data, clip_range_min, clip_range_max)
            logger.info(f'Clip weights <{tensor_name}> to range [{clip_range_min}, {clip_range_max}].')
        new_data = numpy_helper.from_array(new_data)
        named_initializer[tensor_name].raw_data = new_data.raw_data

    def wrap_onnx_constant(self, data):
        """warp onnx constant data to iterable numpy object

        Args:
            data (float or list): data from onnx.get_constant

        Returns:
            ndarray: iterable numpy array
        """
        if type(data) != list:
            return np.array([data])
        else:
            return np.array(data)

    def format_qlinear_dtype_pass(self):
        name2data = prepare_data(self.onnx_model.graph)
        named_initializer = prepare_initializer(self.onnx_model.graph)
        for node in self.onnx_model.graph.node:
            if node.op_type in FAKE_QUANTIZE_OP:
                if node.op_type == 'FakeQuantizeLearnablePerchannelAffine':
                    scale, zero_point = node.input[1], node.input[2]
                    assert node.attribute[0].name == 'quant_max' and node.attribute[1].name == 'quant_min'
                    qmax = node.attribute[0].i
                    qmin = node.attribute[1].i
                else:
                    scale, zero_point, qmin, qmax = node.input[-4:]
                    qmin = self.onnx_model.get_constant(qmin)
                    qmax = self.onnx_model.get_constant(qmax)
                assert qmax - qmin in (2 ** 8 - 1, 2 ** 8 - 2), "Only 8 bit quantization support deployment to ONNX."
                # In onnx, quantize linear node values are in [-128, 127], this step is to remove inconsistency
                if qmax - qmin == 2 ** 8 - 2:
                    self.clip_weight(node, name2data, named_initializer)
                # ? for model mixed constant and initializer
                # scale
                try:
                    scale_proto = self.onnx_model.initializer[scale][0]
                    if scale_proto.raw_data != b'':
                        scale_data = self.onnx_model.get_initializer(scale)
                        self.onnx_model.set_initializer(scale, scale_data.astype(np.float32), raw=False)
                except KeyError:
                    scale_data = self.wrap_onnx_constant(self.onnx_model.get_constant(scale))
                    self.onnx_model.set_initializer(scale, scale_data.astype(np.float32), raw=False)
                # zero_point
                try:
                    zero_point_data = self.onnx_model.get_initializer(zero_point)
                except KeyError:
                    zero_point_data = self.wrap_onnx_constant(self.onnx_model.get_constant(zero_point))
                assert not np.any(zero_point_data != 0), "Asymmetric quantization is not supported for TensorRT Backend."
                if qmin == 0:
                    self.onnx_model.set_initializer(zero_point, zero_point_data.astype(np.uint8), raw=False)
                else:
                    self.onnx_model.set_initializer(zero_point, zero_point_data.astype(np.int8), raw=False)

    def replace_qlinear_layer_pass(self):
        for node in self.onnx_model.graph.node:
            if node.op_type in FAKE_QUANTIZE_OP:
                kwargs = {}
                if node.op_type == 'FakeQuantizeLearnablePerchannelAffine':
                    next_node = self.onnx_model.get_tensor_consumer(node.output[0])[0]
                    if next_node.op_type == 'ConvTranspose':
                        kwargs['axis'] = 1
                    else:
                        kwargs['axis'] = 0
                quantize_linear_node = onnx.helper.make_node("QuantizeLinear", node.input[:3],
                                                             [node.name + '_quantized_out'], node.name + '_quantized', **kwargs)
                dequantize_linear_node = onnx.helper.make_node("DequantizeLinear",
                                                               [node.name + '_quantized_out'] +
                                                               quantize_linear_node.input[1:3],
                                                               node.output,
                                                               node.name + '_dequantized', **kwargs)
                self.onnx_model.insert_node_purely(quantize_linear_node)
                self.onnx_model.insert_node_purely(dequantize_linear_node)
                self.onnx_model.remove_node_purely(node)
                self.onnx_model.topologize_graph()

    def run(self):
        self.onnx_model.topologize_graph()
        self.format_qlinear_dtype_pass()
        self.replace_qlinear_layer_pass()
        self.onnx_model.optimize_model()
        # Per-channel QuantizeLinear and DequantizeLinear is supported since opset 13
        self.onnx_model.set_opset_version('', 13)
        # This gives error with `axis` in QuantizeLinear node
        try:
            onnx.checker.check_model(self.onnx_model.model)
        except onnx.checker.ValidationError as e:
            logger.critical('The model is invalid: %s' % e)
        output_dir = os.path.dirname(self.onnx_model_path)
        self.onnx_model.save_onnx_model(os.path.join(output_dir, 'onnx_quantized_model.onnx'))
