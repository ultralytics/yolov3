import argparse

import onnx
import numpy as np

from onnx import numpy_helper

from collections import namedtuple
from typing import Any, Dict, Optional
from functools import partial

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph.base_tensor import Tensor
from nndct_shared.utils import AddXopError
from nndct_shared.compile.xgraph import XGraph
from nndct_shared.compile.xop_creator import _Converter, _get_xir_attr_from_node, _pack 
from pytorch_nndct.parse.op_dispatcher import OpCreator

class ONNX_OP(object):
    CONV2d = 'Conv'
    RELU = 'Relu'
    MAXPOOL = 'MaxPool'
    ADD = 'Add'
    GEMM = 'Gemm'
    ADPTIVEAVGPOOL2D = 'GlobalAveragePool'
    FLATTEN = 'Flatten'
    INPUT = 'Input'
    RESIZE = 'Resize'
    CONCAT = 'Concat'


class ONNX_PARAM(object):
    WEIGHT = 'weight'
    BIAS = 'bias'
    ZEROPOINT = 'zero_point'
    SCALE = 'scale'
    ALL = [WEIGHT, BIAS, ZEROPOINT, SCALE]


ONNX2NNDCT_CONVERTOR = {
    ONNX_OP.CONV2d: NNDCT_OP.CONV2D,
    ONNX_OP.RELU: NNDCT_OP.RELU,
    ONNX_OP.MAXPOOL: NNDCT_OP.MAX_POOL,
    ONNX_OP.ADD: NNDCT_OP.ADD,
    ONNX_OP.GEMM: NNDCT_OP.DENSE,
    ONNX_OP.ADPTIVEAVGPOOL2D: NNDCT_OP.ADAPTIVEAVGPOOL2D,
    ONNX_OP.FLATTEN: NNDCT_OP.FLATTEN,
    ONNX_OP.INPUT: NNDCT_OP.INPUT,
    ONNX_OP.RESIZE: NNDCT_OP.RESIZE,
    ONNX_OP.CONCAT: NNDCT_OP.CONCAT,
}

perchannel_fakequantizer = [
    'FakeQuantizeLearnablePerchannelAffine', 'FixedPerChannelAffine',
    'FakeQuantizeDSQPerchannel', 
]
pertensor_fakequantizer = [
    'LearnablePerTensorAffine', 'FixedPerTensorAffine',
    'FakeQuantizeDSQPertensor', 'FakeQuantizeTqtAffine'
]
output_fakequantizer = [
    'LearnablePerTensorAffine', 'FakeQuantizeTqtAffine'
]
all_fakequantizer = perchannel_fakequantizer + pertensor_fakequantizer

_filed = [
    'name', 'shape', 'op_type', 'in_tensors', 'in_tensors_dim',
    'in_tensors_layout', 'out_tensors_shape', 'out_name', 'op', 'params',
    'attrs'
]
FakeNode = namedtuple(
    'FakeNode',
    _filed,
)
FakeNode.__new__.__defaults__ = (None, ) * len(_filed)


def data_onnx_op(xgraph: XGraph, node, quant_config):
    shape = node.out_tensors_shape

    out_tensor = np.zeros(shape, dtype=np.float32)
    attrs: Dict[str, Any] = {}
    attrs["shape"] = shape
    attrs["data_type"] = _Converter.to_xir_dtype(out_tensor.dtype)
    xgraph.create_fixed_normal_op(node.name,
                                  "data",
                                  quant_config,
                                  tensor=out_tensor,
                                  attrs=attrs)


def onnx_to_xir(onnx_op_type):
    return partial(default_onnx_to_xop, onnx_op_type)


def default_onnx_to_xop(onnx_op_type, xgraph, node, quant_config):
    attrs = _get_xir_attr_from_node(node)

    input_ops = {}
    if node.attrs['has_bound_params']:
        for param_name, param_tensor in node.params:
            param = xgraph.get_op_by_name(param_name)
            head = param_name.split('.')[-1].lower()[0]
            if param:
                input_ops['weights' if head == 'w' else 'bias'] = [param]

    input_list = []
    for input_name in node.in_tensors:
        if node.attrs['has_bound_params'] and is_param_tensor(input_name):
            continue
        elif is_param_tensor(input_name):
            input_op = xgraph.get_op_by_name(input_name)
        else:
            input_op = xgraph.get_op_by_name(input_name)
        input_list.append(input_op)

    input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name,
                                                     quant_config)

    xgraph.create_fixed_normal_op(node.out_name,
                                  onnx_op_type,
                                  quant_config,
                                  attrs=attrs,
                                  input_ops=input_ops)



def resize(xgraph, node, quant_config):
    attrs: Dict[str, Any] = {}
    attrs["scale"] = node.attrs['scale']
    attrs["align_corners"] = node.attrs['align_corners']
    attrs["half_pixel_centers"] = node.attrs['half_pixel_centers']
    attrs["mode"] = node.attrs['mode']
    attrs["mode"] = {'nearest': "NEAREST"}.get(attrs["mode"].s.decode())
    size = node.attrs['size']
    if size[0] == 0 and size[1] == 0:
        input_ops = {}
        input_list = []
        for input in node.in_tensors:
            input_op = xgraph.get_op_by_name(input)
            input_list.append(input_op)
        input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
        xgraph.create_fixed_normal_op(
            node.name, "resize", quant_config, attrs=attrs, input_ops=input_ops)
    else:
        sub_pack_op, pack_list = _pack(xgraph, node, "size", size, quant_config)
        input_ops = {}
        input_ops["size"] = [sub_pack_op]
        input_list = [xgraph.get_op_by_name(node.in_tensors[0])]
        input_ops["input"] = input_list
        input_ops["input"] = [
            op for op in input_ops["input"]
            if op and op.get_name() not in [i.get_name() for i in pack_list]
        ]
        input_ops["input"] = xgraph.create_input_fix_ops(input_ops["input"], node.name, quant_config)
        xgraph.create_fixed_normal_op(
            node.out_name, "resize", quant_config, attrs=attrs, input_ops=input_ops)
        node_need_to_be_clear = node.attrs['to_remove'] 
        for n in node_need_to_be_clear:
            xgraph.graph.remove_op(xgraph.get_op_by_name(n))   


def avgpool(xgraph: XGraph, node, quant_config):
    needScale = False
    scale = 1.0
    if node.attrs['kernel'] == [3, 3]:
        needScale = True
        scale = 9.0 * 7.0 / 64.0
    elif node.attrs['kernel'] == [5, 5]:
        needScale = True
        scale = 25.0 * 10.0 / 256.0
    elif node.attrs['kernel'] in [[6, 6], [3, 6], [6, 3]]:
        needScale = True
        scale = 36.0 * 7.0 / 256.0
    elif node.attrs['kernel'] == [7, 7]:
        needScale = True
        scale = 49.0 * 21.0 / 1024.0
    elif node.attrs['kernel'] == [14, 14]:
        needScale = True
        scale = 196.0 * 21.0 / 4096.0

    if needScale:
        attrs = node.attrs
        input_ops = {}
        input_ops["input"] = [xgraph.get_op_by_name(node.in_tensors[0])]
        input_ops["input"] = xgraph.create_input_fix_ops(
            input_ops["input"], node.name, quant_config)
        xgraph.create_fixed_normal_op(node.name + '_pool',
                                      "avgpool2d",
                                      quant_config,
                                      attrs=attrs,
                                      input_ops=input_ops)

        scale = [scale]
        xgraph.create_fixed_const_op(name=node.name + "_scale",
                                     data=np.array(scale, dtype=np.float32),
                                     quant_info=quant_config)

        input_ops = {}
        input_ops["input"] = [
            xgraph.get_op_by_name(node.name + '_pool'),
            xgraph.get_op_by_name(node.name + "_scale")
        ]
        xgraph.create_fixed_normal_op(node.out_name,
                                      "mul",
                                      quant_config,
                                      input_ops=input_ops)
    else:
        onnx_to_xir("avgpool2d")(xgraph, node, quant_config)


def flatten(xgraph: XGraph, node, quant_config):

    if node.in_tensors_dim[0] != 4 or node.in_tensors_layout[
            0] == Tensor.Layout.NHWC:
        onnx_to_xir("flatten")(xgraph, node, quant_config)
    else:
        attrs: Dict[str, Any] = {}
        # NHWC -> NCHW
        attrs["order"] = [0, 3, 1, 2]
        input_ops = {}
        input_ops["input"] = [xgraph.get_op_by_name(node.in_tensors[0])]
        xgraph.create_fixed_normal_op(node.name + "_i0",
                                      "transpose",
                                      quant_config,
                                      attrs=attrs,
                                      input_ops=input_ops)

        attrs = node.attrs

        input_ops = {}
        input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]

        xgraph.create_fixed_normal_op(node.out_name,
                                      "flatten",
                                      quant_config,
                                      attrs=attrs,
                                      input_ops=input_ops)


def dense(xgraph: XGraph, node, quant_config):
    input_ops = {}
    for param_name, param_tensor in node.params:
        param = xgraph.get_op_by_name(param_name)
        head = param_name.split('.')[-1].lower()[0]
        if head == 'b':
            input_ops['bias'] = [param]
        else:
            weights = xgraph.get_op_by_name(param_name)

    input_list = []
    for input in node.in_tensors:
        input_op = xgraph.get_op_by_name(input)
        input_list.append(input_op)
    input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name,
                                                     quant_config)
    input_ops["input"].append(weights)

    attrs: Dict[str, Any] = {}
    attrs["transpose_a"] = False
    attrs["transpose_b"] = True

    xgraph.create_fixed_normal_op(node.out_name,
                                  "matmul",
                                  quant_config,
                                  attrs=attrs,
                                  input_ops=input_ops)


def is_param_tensor(name):
    # judge a instance is a parameter tensor or not
    return name.split('.')[-1] in ONNX_PARAM.ALL if name else False


ONNX2XIR_CONVERTOR = {
    ONNX_OP.INPUT: data_onnx_op,
    ONNX_OP.CONV2d: onnx_to_xir('conv2d'),
    ONNX_OP.ADPTIVEAVGPOOL2D: avgpool,
    ONNX_OP.MAXPOOL: onnx_to_xir('maxpool2d'),
    ONNX_OP.RELU: onnx_to_xir('relu'),
    ONNX_OP.ADD: onnx_to_xir('add'),
    ONNX_OP.GEMM: dense,
    ONNX_OP.FLATTEN: flatten,
    ONNX_OP.RESIZE: resize,
    ONNX_OP.CONCAT: onnx_to_xir('concat')
}


def update_inp2node_out2node(graph):
    out2node = {}
    inp2node = {}
    for node in graph.node:
        for out in node.output:
            # suppose each node only has one output
            out2node[out] = node
        for idx, inp in enumerate(node.input):
            # one node may have multiple inputs
            if inp not in inp2node:
                inp2node[inp] = []
            inp2node[inp].append([node, idx])
    return out2node, inp2node

def prepare_data(graph):
    params = {}
    for init in graph.initializer:
        params[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    params[node.output[0]] = numpy_helper.to_array(attr.t)
    return params

def prepare_initializer(graph):
    named_initializer = {}
    for init in graph.initializer:
        named_initializer[init.name] = init
    return named_initializer

def parse_attrs(node_attrs):
    attrs = {}
    for attr in node_attrs:
        if attr.type == onnx.AttributeProto.AttributeType.INTS:
            attrs[attr.name] = tuple(attr.ints)
        elif attr.type == onnx.AttributeProto.AttributeType.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
            attrs[attr.name] = tuple(attr.floats)
        elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
            attrs[attr.name] = numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.AttributeType.STRING:
            attrs[attr.name] = str(attr.s)
        elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
            attrs[attr.name] = tuple([str(x) for x in attr.strings])
        else:
            raise Exception("ATTR Type [{}] Not Supported!".format(attr.type))
    return attrs



def get_constant_inputs(node, out2node):
    node_list = []
    for inp in node.input:
        if inp in out2node and out2node[inp].op_type == 'Constant':
            node_list.append(out2node[inp])
    return node_list


class XIR_process(object):

    bias_fix_point, bias_bitwidth = 0, 8

    def create_normal_nodes_from_onnx_graph(self, onnx_graph):
        normal_nodes = []
        for node in onnx_graph.node:
            if node.op_type in ['Constant']:
                continue
            elif node.op_type == ONNX_OP.CONV2d:
                node_attr = {'has_bound_params': True}
                node_op = self.get_xop_of_conv2d(node)
                input_weight_bias = self.get_unquant_input_weight_bias_of_node(
                    node)
                params = [(input_weight_bias['weight'],
                           self.name2data[input_weight_bias['weight']])]
                if input_weight_bias['bias']:
                    params.append((input_weight_bias['bias'],
                                   self.name2data[input_weight_bias['bias']]))

                new_node = FakeNode(
                    name=node.name,
                    shape=self.name2shape[input_weight_bias['input']],
                    op_type=node.op_type,
                    out_tensors_shape=self.name2shape[node.name],
                    op=node_op,
                    params=params,
                    in_tensors=[
                        input_weight_bias[k] for k in input_weight_bias.keys() if input_weight_bias[k]
                    ],
                    attrs=node_attr,
                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.GEMM:
                node_attr = {'has_bound_params': True}
                input_weight_bias = self.get_unquant_input_weight_bias_of_node(
                    node)
                params = [(input_weight_bias['weight'],
                           self.name2data[input_weight_bias['weight']])]
                if input_weight_bias['bias']:
                    params.append((input_weight_bias['bias'],
                                   self.name2data[input_weight_bias['bias']]))

                inputs = [input_weight_bias['input']]

                new_node = FakeNode(name=node.name,
                                    op_type=node.op_type,
                                    params=params,
                                    in_tensors=inputs,
                                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.MAXPOOL:
                node_attr = {'has_bound_params': False}
                node_op = self.get_xop_of_max_pool(node)
                inputs = self.get_unquant_inputs(node)
                input_weight_bias = self.get_unquant_input_weight_bias_of_node(
                    node)
                new_node = FakeNode(
                    name=node.name,
                    shape=self.name2shape[input_weight_bias['input']],
                    op_type=node.op_type,
                    out_tensors_shape=self.name2shape[node.name],
                    op=node_op,
                    in_tensors=inputs,
                    attrs=node_attr,
                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.RELU:
                node_attr = {'has_bound_params': False}
                node_op = self.get_xop_of_relu(node)
                inputs = self.get_unquant_inputs(node)

                new_node = FakeNode(
                    name=node.name,
                    op_type=node.op_type,
                    out_tensors_shape=self.name2shape[node.name],
                    op=node_op,
                    in_tensors=inputs,
                    attrs=node_attr,
                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.ADD:
                node_attr = {'has_bound_params': False}
                inputs = self.get_unquant_inputs(node)
                node_op = self.get_xop_of_add(node, inputs)

                new_node = FakeNode(
                    name=node.name,
                    shape=self.name2shape[input_weight_bias['input']],
                    op_type=node.op_type,
                    out_tensors_shape=self.name2shape[node.name],
                    op=node_op,
                    in_tensors=inputs,
                    attrs=node_attr,
                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.ADPTIVEAVGPOOL2D:
                node_op = self.get_xop_of_adaptive_avg_pool2d(node)
                inputs = self.get_unquant_inputs(node)
                attrs = {}
                attrs['kernel'] = self.name2shape[inputs[0]][1:3]
                attrs['stride'] = self.name2shape[inputs[0]][1:3]
                attrs['count_include_pad'] = True
                attrs['global'] = True
                attrs['pad'] = [0, 0, 0, 0]
                attrs['pad_mode'] = 'FLOOR'
                attrs['has_bound_params'] = False
                new_node = FakeNode(
                    name=node.name,
                    op_type=node.op_type,
                    op=node_op,
                    in_tensors=inputs,
                    out_tensors_shape=self.name2shape[node.name],
                    attrs=attrs,
                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.FLATTEN:
                inputs = self.get_unquant_inputs(node)
                node_op = self.get_xop_of_flatten(node)
                inputs_dim = [len(self.name2shape[i]) for i in inputs]
                # IT IS A BUG OF THE XIR COMPILER for vitis-ai 1.4.1.978
                # inputs_layout = [
                #     Tensor.Layout.NHWC
                #     if self.name2shape[i][-1] > 1 else Tensor.Layout.NCHW
                #     for i in inputs
                # ]
                inputs_layout = [Tensor.Layout.NCHW]
                start_axis = node.attribute[0].i
                end_axis = inputs_dim[0] - 1
                attrs = {}
                attrs['start_axis'] = start_axis
                attrs['end_axis'] = end_axis
                attrs['has_bound_params'] = False
                new_node = FakeNode(name=node.name,
                                    op_type=node.op_type,
                                    op=node_op,
                                    in_tensors=inputs,
                                    in_tensors_dim=inputs_dim,
                                    in_tensors_layout=inputs_layout,
                                    attrs=attrs,
                                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.RESIZE:
                node_op = self.get_xop_of_interpolate(node) 
                inputs = self.get_unquant_inputs(node)
                inputs_layout = [Tensor.Layout.NCHW]
                size = numpy_helper.to_array(self.name2data[node.input[3]]) if len(node.input) == 4 else None 
                scale = self.name2data[node.input[2]] if not size else None 
                assert scale is None or (scale - scale.astype('int') == 0).all(), f'Only integer scales is supportted! The given scale is {scale}'
                if not size:
                    size = self.name2shape[node.name] 
                    scale = np.array([1, 1]) 
                attrs_dict = {} 
                for a in node.attribute:
                    attrs_dict[a.name] = a 
                mode = attrs_dict['mode'] 
                align_corners = True if attrs_dict['coordinate_transformation_mode'] == "align_corners" else False
                half_pixel_centers = True if attrs_dict['coordinate_transformation_mode'] == 'pytorch_half_pixel' else False
                if size and len(size) == 4:
                    size = size[1:-1] 
                attrs = {} 
                attrs['scale'] = scale 
                attrs['align_corners'] = align_corners 
                attrs['half_pixel_centers'] = half_pixel_centers 
                attrs['mode'] = mode 
                attrs['size'] = size 
                attrs['has_bound_params'] = False
                if scale is not None:
                    attrs['to_remove'] = [node.input[2]]
                new_node = FakeNode(name=node.name,
                                    op_type=node.op_type,
                                    op=node_op,
                                    in_tensors=inputs,
                                    in_tensors_layout=inputs_layout,
                                    out_tensors_shape=size,
                                    attrs=attrs,
                                    out_name=node.output[0])
                normal_nodes.append(new_node)
            elif node.op_type == ONNX_OP.CONCAT:
                node_op = self.get_xop_of_concat(node)
                inputs = self.get_unquant_inputs(node)
                inputs_layout = [Tensor.Layout.NCHW]
                size = self.name2shape[node.name] 
                axis = node.attribute[0].i 
                dim = axis
                attrs = {'axis': dim}
                attrs['has_bound_params'] = False
                new_node = FakeNode(name=node.name,
                                    op_type=node.op_type,
                                    op=node_op,
                                    in_tensors=inputs,
                                    in_tensors_layout=inputs_layout,
                                    out_tensors_shape=size,
                                    attrs=attrs,
                                    out_name=node.output[0])
                normal_nodes.append(new_node)
        return normal_nodes

    def create_input_nodes_from_onnx_graph(self, onnx_graph, reshape=True):
        input_nodes = []
        for input_message in onnx_graph.input:
            shape = self.get_dim_from_tensor_shape_message(input_message)
            if reshape and len(shape) == 4:
                shape = list(np.zeros(shape=shape).transpose(0, 2, 3, 1).shape)
            node = FakeNode(
                name=input_message.name,
                shape=self.get_dim_from_tensor_shape_message(input_message),
                op_type='Input',
                out_tensors_shape=shape)

            input_nodes.append(node)
        return input_nodes

    def get_xop_of_concat(self, node):
        _n = namedtuple('_n', ['name', 'shape'])
        inputs = self.get_unquant_inputs(node)
        ts = [_n(i, self.name2shape[i]) for i in inputs]
        axis = node.attribute[0].i 
        dim = axis
        return OpCreator(None).cat(ts, dim)

    def get_xop_of_interpolate(self, node):
        input = self.out2node[node.input[0]]
        _n = namedtuple('_n', ['name', 'shape']) 
        inputs = self.get_unquant_inputs(node)
        input_node = Tensor(name=inputs[0], shape=self.name2shape[inputs[0]])
        # self.graph._graph.node(get_full_name(self.graph.name, node.input[0])).out_tensors[0]
        size = numpy_helper.to_array(node.input[3]) if len(node.input) == 4 else None 
        scale_factor = self.name2data[node.input[2]].tolist() if not size else [1, 1]
        if len(scale_factor) == 1:
            scale_factor += scale_factor
        elif len(scale_factor) == 4:
            scale_factor = scale_factor[-2:]
        if size and len(size) == 4:
            size = size[1:-1]
        attrs = node.attribute
        attrs_dict = {} 
        for a in attrs:
            attrs_dict[a.name] = a 
        mode = f"'{attrs_dict['mode'].s.decode()}'"
        assert mode == "'nearest'", f'the interpolate {mode} is not supported' 
        align_corners = True if attrs_dict['coordinate_transformation_mode'].s.decode() == "align_corners" else None 
        recompute_scale_factor = None 
        return OpCreator(None)._interpolate(input_node, size, scale_factor, mode, align_corners, recompute_scale_factor)

    def get_xop_of_conv2d(self, node):
        pre_op = self.out2node.get(node.input[0], None)
        if pre_op and pre_op.op_type in all_fakequantizer:
            if pre_op.input[0] in self.out2node:
                pre_op = self.out2node[pre_op.input[0]]
            elif pre_op.input[0] in self.input_nodes:
                pre_op = FakeNode(name=pre_op.input[0])
        input_shape = self.name2shape[pre_op.name] if pre_op else self.name2shape[node.input[0]]
        input = np.zeros(input_shape)
        weight = self.name2data[self.out2node[node.input[1]].input[0]]
        if len(node.input) >= 3:
            bias = node.input[2]
        else:
            bias = np.zeros(weight.shape[-1])
        stride = list(node.attribute[4].ints)
        padding = list(node.attribute[3].ints)
        dilation = list(node.attribute[0].ints)
        transposed = False
        output_padding = None
        groups = int(node.attribute[1].i)
        return OpCreator(None)._convolution(input, weight, bias, stride,
                                            padding, dilation, transposed,
                                            output_padding, groups, None, None,
                                            None)

    def get_xop_of_adaptive_avg_pool2d(self, node):
        output_shape = self.name2shape[node.name]
        pre_op = self.out2node[node.input[0]]
        if pre_op.op_type in all_fakequantizer:
            if pre_op.input[0] in self.out2node:
                pre_op = self.out2node[pre_op.input[0]]
            elif pre_op.input[0] in self.input_nodes:
                pre_op = FakeNode(name=pre_op.input[0])
        input_shape = self.name2shape[pre_op.name]
        return OpCreator(None).adaptive_avg_pool2d(np.zeros(input_shape),
                                                   output_shape[1:3])

    def get_xop_of_max_pool(self, node):
        kernel_size = list(node.attribute[1].ints)
        stride = list(node.attribute[3].ints)
        padding = list(node.attribute[2].ints)
        dilation = [1]
        ceil_mode = int(node.attribute[0].i)
        return OpCreator(None).max_pool2d(None, kernel_size, stride, padding,
                                          dilation, ceil_mode)

    def get_xop_of_relu(self, node):
        return OpCreator(None).relu(None)

    def get_xop_of_add(self, node, inputs):
        input_tensor = Tensor(name=inputs[0],
                              shape=self.name2shape[inputs[0]],
                              dtype=np.dtype('int8'))
        other_tensor = Tensor(name=inputs[1],
                              shape=self.name2shape[inputs[1]],
                              dtype=np.dtype('int8'))
        return OpCreator(None).add(input_tensor, other_tensor)

    def get_xop_of_flatten(self, node):
        inputs = self.get_unquant_inputs(node)
        inputs_dim = [len(self.name2shape[i]) for i in inputs]
        start_axis = node.attribute[0].i
        end_axis = inputs_dim[0] - 1
        return OpCreator(None).flatten(inputs[0], start_axis, end_axis)

    def shape_patch(self, graph, graph_without_quant, reshape=True):
        inferred_model = onnx.shape_inference.infer_shapes(graph_without_quant)
        value_info = inferred_model.graph.value_info
        self.name2shape = {}
        self.input_nodes = []
        for out in value_info:
            shape = self.get_dim_from_tensor_shape_message(out)
            if reshape and len(shape) == 4:
                shape = list(np.zeros(shape=shape).transpose(0, 2, 3, 1).shape)
            self.name2shape[self.out2node[out.name].name] = shape
            self.name2shape[out.name] = shape
        for input_message in graph.graph.input:
            shape = self.get_dim_from_tensor_shape_message(input_message)
            if reshape and len(shape) == 4:
                shape = list(np.zeros(shape=shape).transpose(0, 2, 3, 1).shape)
            self.name2shape[input_message.name] = shape
            self.input_nodes.append(input_message.name)
        for output_message in graph.graph.output:
            shape = self.get_dim_from_tensor_shape_message(output_message)
            if reshape and len(shape) == 4:
                shape = list(np.zeros(shape=shape).transpose(0, 2, 3, 1).shape)
            if output_message.name in self.out2node:
                pre_node = self.out2node[output_message.name]
                if pre_node.op_type in all_fakequantizer:
                    unquant_node = self.out2node[pre_node.input[0]]
                    input = unquant_node.name
                    input = unquant_node.output[0]
                else:
                    input = pre_node.name
                    input = pre_node.output[0]
                self.name2shape[input] = shape 
                self.name2shape[self.out2node[input].name] = shape 
            self.name2shape[output_message.name] = shape


    def get_dim_from_tensor_shape_message(self, message):
        return [v.dim_value for v in message.type.tensor_type.shape.dim]

    def get_unquant_inputs(self, node):
        inputs = []
        for input in node.input:
            if input in self.name2data:
                inputs.append(input)
            elif input in self.out2node and not self.out2node[
                    input].op_type in all_fakequantizer:
                inputs.append(self.out2node[input].output[0])
            elif input in self.out2node and not self.out2node[
                    input].op_type in output_fakequantizer and self.out2node[
                        input].op_type in all_fakequantizer:
                continue
            elif input in self.out2node and self.out2node[
                    input].op_type in output_fakequantizer:
                input = self.out2node[input].input[0]
                if input in self.out2node:
                    pre_node = self.out2node[input]
                    if pre_node.op_type in all_fakequantizer:
                        unquant_node = self.out2node[pre_node.input[0]]
                        input = unquant_node.name
                        input = unquant_node.output[0]
                    else:
                        input = pre_node.name
                        input = pre_node.output[0]
                inputs.append(input)
            elif input in self.inp2node and input not in self.out2node:
                inputs.append(input)
        return inputs

    def get_unquant_input_weight_bias_of_node(self, node):
        input_weight_and_bias = {'input': None, 'weight': None, 'bias': None}
        for input in node.input:
            type = None
            if input in self.name2data:
                type = input.split('.')[-1]
            elif input in self.out2node and not self.out2node[
                    input].op_type in all_fakequantizer:
                type = 'input'
                if input in self.out2node:
                    pre_node = self.out2node[input]
                    if pre_node.op_type in all_fakequantizer:
                        unquant_node = self.out2node[pre_node.input[0]]
                        input = unquant_node.name
                        input = unquant_node.output[0]
                    else:
                        input = pre_node.name
                        input = pre_node.output[0]
            elif input in self.out2node and self.out2node[
                    input].op_type in all_fakequantizer:
                pre_node = self.out2node[input]
                real_in = pre_node.input[0]
                if real_in in self.name2data:
                    type = real_in.split('.')[-1]
                    input = real_in
                elif real_in in self.inp2node:
                    type = 'input'
                    if real_in in self.out2node:
                        input = self.out2node[real_in].name
                        input = self.out2node[real_in].output[0]
                    else:
                        input = real_in
            elif input in self.inp2node and input not in self.out2node:
                type = 'input' 

            if type in [ONNX_PARAM.BIAS, ONNX_PARAM.WEIGHT, 'input']:
                input_weight_and_bias[type] = input
        return input_weight_and_bias

    def parse_bias_qparams_from_data(self, node, name2data):
        tensor_name = node.name 
        bit_width = 8
        max_val = max(-1 * self.name2data[tensor_name].flatten().min(), self.name2data[tensor_name].flatten().max()) 
        if max_val < 2 ** -10:
            sign_shift = -1
            val_shift = -bit_width
        elif max_val == -1 * self.name2data[tensor_name].flatten().min():
            sign_shift = 0
            val_shift = -np.round(np.log2(max_val))
        else:
            sign_shift = 1
            val_shift = -np.round(np.log2(max_val))
        sym_fix = int(bit_width - sign_shift + val_shift - 1)
        return tensor_name, sym_fix, bit_width

    def get_quant_weight_loss(self, unquant_weight, amp, val_max):
        real_weight = unquant_weight * amp 
        real_weight = np.round(np.clip(real_weight, -val_max, val_max - 1))
        real_weight /= amp 
        return float(((real_weight - unquant_weight)**2).sum())

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
            print(f'qmin and qmax are not found for <{node.name}>!')
            raise ValueError('The name2data is corrputed.')

        if abs(zero_point - 0).all() < 1e-5:
            bit_width = np.log2(qmax - qmin + 1).astype(int)
            final_scale = int(np.floor(np.log2(1 / scale)))
        else:
            print(f'<{node.name}> is a asym quant node, which is not support by Vitis Backend')
        return tensor_name, final_scale, bit_width

    def get_wbi_of_onnx_calib(self, graph, calib_graph):
        self.biasname2wi = {}
        for node in calib_graph.node:
            if node.op_type in [ONNX_OP.CONV2d, ONNX_OP.GEMM]:
                input_pre_fix = node.input[0]
                weight_pre_fix = node.input[1]
                if len(node.input) > 2:
                    bias_fix = node.input[2]
                    self.biasname2wi[bias_fix] = [weight_pre_fix, input_pre_fix]
                else: 
                    bias_fix = None 

    def get_quant_config_of_onnx(self, graph, name):
        qconfig = {}

        output_config = {}
        for node in graph.node:
            if node.op_type in output_fakequantizer:
                output_name, fix_point, bitwidth = self.parse_qparams(
                    node, self.name2data)
                output_config[output_name] = [bitwidth, fix_point]

        param_config = {}
        for param in graph.initializer:
            ptype = param.name[param.name.rfind('.') + 1:]
            data_type = param.data_type
            if ptype in [ONNX_PARAM.WEIGHT
                         ] or (ptype in [ONNX_PARAM.BIAS]
                               and self.inp2node[param.name][0][0].op_type
                               in all_fakequantizer):
                _, fix_point, bitwidth = self.parse_qparams(
                    self.inp2node[param.name][0][0], self.name2data)
                param_config[param.name] = [bitwidth, fix_point]
            elif ptype in [ONNX_PARAM.BIAS] and not self.inp2node[
                    param.name][0][0].op_type in all_fakequantizer:
                bitwidth = self.bias_bitwidth
                _, fix_point, bitwidth = self.parse_bias_qparams_from_data(param, self.name2data)
                param_config[param.name] = [bitwidth, fix_point]
            elif ptype in [ONNX_PARAM.ZEROPOINT, ONNX_PARAM.SCALE]:
                continue
            else:
                if param.name in self.inp2node:
                    print(f'A data array named {param.name} with {numpy_helper.to_array(param)} has been used.')

        output_config = {}
        for node in graph.node:
            if node.op_type in output_fakequantizer:
                output_name, fix_point, bitwidth = self.parse_qparams(
                    node, self.name2data)
                output_config[output_name] = [bitwidth, fix_point]

        input_config = {}

        qconfig['param'] = param_config
        qconfig['output'] = output_config
        qconfig['input'] = input_config

        return qconfig

    def do_compile(self,
                   onnx_graph_file,
                   onnx_graph_file_cali,
                   name=None,
                   graph_attr_kwargs: Optional[Dict[str, Any]] = None,
                   reshape=True) -> None:
        # get the onnx graph
        onnx_graph = onnx_graph_file.graph
        # create a xgraph like do_compile
        xgraph = XGraph(name if name else onnx_graph.name)
        self.graph = xgraph

        # build the inter-node via the in/out infomation
        self.out2node, self.inp2node = update_inp2node_out2node(onnx_graph)

        # add the extra args
        if graph_attr_kwargs is not None:
            for name, attr in graph_attr_kwargs.items():
                xgraph.graph.set_attr(name, attr)

        # get the weight/bias in the initializer
        self.name2data = prepare_data(onnx_graph)
        # get the input/output
        self.out2node, self.inp2node = update_inp2node_out2node(onnx_graph)
        # patch the wbi pairs 
        self.get_wbi_of_onnx_calib(onnx_graph, onnx_graph_file_cali.graph)
        # create weight/bias node <fix> via FakeQuantizeLearnablePerchannelAffine
        quant_config_info = self.get_quant_config_of_onnx(onnx_graph, name)
        # patch the shape
        self.shape_patch(onnx_graph_file,
                         onnx_graph_file_cali,
                         reshape=reshape)

        op_to_xir_op = ONNX2NNDCT_CONVERTOR

        def get_op_of_initializer(name):
            if get_type_of_initializer(name) in [
                    ONNX_PARAM.ZEROPOINT, ONNX_PARAM.SCALE
            ]:
                return None
            if name not in self.inp2node:
                return None
            node = self.inp2node[name][0][0]
            if node.op_type in all_fakequantizer:
                node = self.inp2node[node.output[0]][0][0]
            ret_type = op_to_xir_op[
                node.op_type] if node.op_type in op_to_xir_op.keys() else node.op_type
            return ret_type

        def get_type_of_initializer(name):
            name = name[name.rfind('.') + 1:]
            if name in ONNX_PARAM.ALL:
                return name
            else:
                return None

        implemented_op = [
            ONNX2NNDCT_CONVERTOR[i] for i in ONNX2NNDCT_CONVERTOR.keys()
        ]
        implemented_op += all_fakequantizer


        print('Essential data has been collected by the Vitis backend compiler.')
        for node in onnx_graph.initializer:
            op_type = get_op_of_initializer(node.name)
            if op_type is None:
                continue
            param_type = get_type_of_initializer(node.name)
            if op_type in [
                    NNDCT_OP.BATCH_NORM, NNDCT_OP.BATCH_NORM1D,
                    NNDCT_OP.BATCH_NORM3D
            ]:
                print('BN should have been merged in previous precess!')
                raise Exception()
            if xgraph.get_op_by_name(node.name):
                continue
            data = self.name2data[node.name]
            if op_type == NNDCT_OP.CONV2D and param_type == ONNX_PARAM.WEIGHT:
                # do weight reshape from oikk to okki
                data = data.transpose(0, 2, 3, 1)
                data = np.ascontiguousarray(data)
            if op_type in implemented_op is False:
                raise NotImplementedError(
                    f'{op_type} has not been implemented.')

            try:
                xgraph.create_fixed_const_op(name=node.name,
                                             data=data,
                                             quant_info=quant_config_info)
            except Exception as e:
                raise AddXopError(node.name, 'const', str(e))

        unknown_op_types = {
            f"{node.op_type}({node.name})"
            for node in onnx_graph.node
            if node.op_type not in ONNX2NNDCT_CONVERTOR
            and node.op_type not in all_fakequantizer + ['Constant']
        }
        if not unknown_op_types:
            input_nodes = self.create_input_nodes_from_onnx_graph(
                onnx_graph, reshape=reshape)
            for node in input_nodes:
                try:
                    print(f'Trying to insert {node.name}<{node.op_type}> into the Vitis Call-Graph.')
                    ONNX2XIR_CONVERTOR.get(node.op_type,
                                           None)(xgraph, node,
                                                 quant_config_info)
                except Exception as e:
                    raise AddXopError(node.name, node.op_type, str(e))
            # before that need to insert input
            normal_node = self.create_normal_nodes_from_onnx_graph(onnx_graph)
            for node in normal_node:
                if node.op_type in all_fakequantizer + ['Constant']:
                    continue
                try:
                    ONNX2XIR_CONVERTOR.get(node.op_type,
                                           None)(xgraph, node,
                                                 quant_config_info)
                except Exception as e:
                    raise AddXopError(node.name, node.op_type, str(e))
        else:
            raise AddXopError(unknown_op_types)

        return_ops = []
        print('Trying to append <Output Nodes> into the Vitis Call-Graph.')
        for out in onnx_graph.output:
            if out.name in self.out2node:
                pre_node = self.get_unquant_inputs(self.out2node[out.name])[0]
                if xgraph.get_op_by_name(pre_node + '_fix'):
                    return_ops.append(pre_node + '_fix')
                else:
                    return_ops.append(pre_node)

        if return_ops:
            xgraph.graph.set_attr("return_ops", return_ops)

        if name:
            if quant_config_info is None:
                name += '_float'
            else:
                name += '_int'

            xgraph.export_to_xmodel(name)
            print('Finished exporting the Vitis Xmodel.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-Q", "--qmodel", help="onnx model with fake quantize nodes.", required=True)
    parser.add_argument("-C", "--cmodel", help="onnx model without fake quantize nodes, or deploy model.", required=True)
    parser.add_argument("-N", "--name", help="model name", required=True)
    args = parser.parse_args()

    xir_compiler = XIR_process()
    qmodel = onnx.load(args.qmodel)
    cmodel = onnx.load(args.cmodel)
    xir_compiler.do_compile(qmodel, cmodel, args.name)
    print('MQBench has converted the model into xmodel.')
