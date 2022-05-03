import json
import os


import onnx
import numpy as np
from onnx import numpy_helper

from mqbench.utils.logger import logger
from mqbench.deploy.common import (
    update_inp2node_out2node,
    prepare_initializer,
    prepare_data,
    OnnxPreprocess,
    get_constant_inputs,
    parse_attrs
)


PERCHANNEL_FAKEQUANTIZER = ['FakeQuantizeLearnablePerchannelAffine', 
                            'FixedPerChannelAffine',
                            'FakeQuantizeDSQPerchannel']
PERTENSOR_FAKEQUANTIZER = ['LearnablePerTensorAffine', 
                           'FixedPerTensorAffine',
                           'FakeQuantizeDSQPertensor',
                           'FakeQuantizeTqtAffine']
ALL_FAKEQUANTIZER = PERCHANNEL_FAKEQUANTIZER + PERTENSOR_FAKEQUANTIZER


class LinearQuantizer_process(object):
    # some method like dorefa need pre-compute weights
    def weight_preprocess(self, target_tensor, out2node, inp2node, named_initializer):
        def find_weight(tensor):
            if tensor not in named_initializer:
                _node = out2node[tensor]
                for inp in _node.input:
                    return find_weight(inp)
            return tensor
        weight = find_weight(target_tensor)

        # TODO need more general method, like onnxruntime infer
        data = numpy_helper.to_array(named_initializer[weight])
        data = np.tanh(data)
        data = data / (np.max(np.abs(data)) + 1e-5)
        data = numpy_helper.from_array(data)
        named_initializer[weight].raw_data = data.raw_data

        redundant_nodes = []

        def find_redundant_nodes(tensor):
            if tensor == target_tensor:
                return
            nodes = inp2node[tensor]
            for node, idx in nodes:
                if node not in redundant_nodes:
                    redundant_nodes.append(node)
                    redundant_nodes.extend(get_constant_inputs(node, out2node))
                find_redundant_nodes(node.output[0])
        find_redundant_nodes(weight)
        return weight, redundant_nodes

    def deal_with_weight_fakequant(self, node, out2node, inp2node, named_initializer):
        next_nodes = inp2node[node.output[0]]
        assert len(next_nodes) == 1
        next_node, idx = next_nodes[0]
        assert next_node.op_type in ['Conv', 'Gemm', 'ConvTranspose']
        redundant_nodes = []
        if node.input[0] not in named_initializer:
            node.input[0], redundant_nodes = \
                self.weight_preprocess(node.input[0], out2node, inp2node, named_initializer)
        next_node.input[idx] = node.input[0]
        return redundant_nodes

    def deal_with_activation_fakequant(self, node, inp2node):
        next_nodes = inp2node[node.output[0]]
        for next_node, idx in next_nodes:
            next_node.input[idx] = node.input[0]
        return

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

    def clip_weight(self, node, name2data, inp2node, named_initializer):
        tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
        data = name2data[tensor_name]
        clip_range_min = ((qmin - zero_point) * scale).astype(data.dtype)
        clip_range_max = ((qmax - zero_point) * scale).astype(data.dtype)
        if scale.shape[0] > 1:
            new_data = []
            transposed = False
            next_node = inp2node[node.output[0]]
            if len(next_node) == 1 and next_node[0][0].op_type == 'ConvTranspose':
                transposed = True
                data = data.transpose(1, 0, 2, 3)
            for c in range(data.shape[0]):
                new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
            new_data = np.array(new_data)
            if transposed:
                new_data = new_data.transpose(1, 0, 2, 3)
            logger.info(f'Clip weights <{tensor_name}> to per-channel ranges.')
        else:
            new_data = np.clip(data, clip_range_min, clip_range_max)
            logger.info(f'Clip weights <{tensor_name}> to range [{clip_range_min}, {clip_range_max}].')
        new_data = numpy_helper.from_array(new_data)
        named_initializer[tensor_name].raw_data = new_data.raw_data

    def post_process_clip_ranges(self, clip_ranges, graph, inp2node):
        def find_the_closest_clip_range(node):
            if node.input[0] in clip_ranges:
                return node.input[0]
            elif node.op_type in ['Flatten', 'Resize'] and node.output[0] in inp2node:
                return find_the_closest_clip_range(inp2node[node.output[0]][0][0])
            else:
                return None

        for node in graph.node:
            if node.op_type in ['Flatten', 'Resize']:
                tensor_name = find_the_closest_clip_range(node)
                if tensor_name:
                    clip_ranges[node.input[0]] = clip_ranges[tensor_name]
                    logger.info(f'Pass <{tensor_name}> clip range to <{node.name}> input <{node.input[0]}>.')
        return clip_ranges

    def remove_fakequantize_and_collect_params(self, onnx_path, model_name, backend):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)

        clip_ranges = {}
        nodes_to_be_removed = []
        for node in graph.node:
            if node.op_type in ALL_FAKEQUANTIZER:
                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))

            if node.op_type in PERCHANNEL_FAKEQUANTIZER:
                # fake quantize for weights, suppose per-channel quantize only for weight
                redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                nodes_to_be_removed.extend(redundant_nodes)
                self.clip_weight(node, name2data, inp2node, named_initializer)
                if backend == 'ppl':
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    clip_ranges[tensor_name] = {'step': [float(x) for x in scale],
                                                'zero_point': [int(x) for x in zero_point],
                                                'min': [float(x) for x in scale * (qmin - zero_point)],
                                                'max': [float(x) for x in scale * (qmax - zero_point)],
                                                'bit': int(np.log2(qmax - qmin + 1)),
                                                'type': "biased",
                                                }
                elif backend == 'vitis':
                    logger.info("Vitis-DPU does not support per-channel quatization.")
                    raise NotImplementedError("Vitis-DPU does not support per-channel quatization.")


            elif node.op_type in PERTENSOR_FAKEQUANTIZER:
                if node.output[0] not in inp2node:
                    assert node.output[0] in [l.name for l in graph.output]
                    inp2node[node.output[0]] = []
                next_nodes = inp2node[node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, name2data, inp2node, named_initializer)
                elif len(next_nodes) == 1 and next_nodes[0][1] == 2 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for bias 
                    assert backend == 'vitis'
                    redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, name2data, inp2node, named_initializer)
                else:
                    # fake quantize for activations
                    self.deal_with_activation_fakequant(node, inp2node)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    for out in graph.output:
                        if out.name == node.output[0]:
                            out.name = tensor_name

                    if backend == 'tensorrt':
                        clip_ranges[tensor_name] = float(scale * max(-qmin, qmax))
                    elif backend == 'snpe':
                        clip_ranges[tensor_name] = [
                            {'bitwidth': int(np.log2(qmax - qmin + 1)),
                             'min': float(scale * (qmin - zero_point)),
                             'max': float(scale * (qmax - zero_point))}
                        ]
                if backend == 'ppl':
                    clip_ranges[tensor_name] = {'step': float(scale),
                                                'zero_point': int(zero_point),
                                                'min': float(scale * (qmin - zero_point)),
                                                'max': float(scale * (qmax - zero_point)),
                                                'bit': int(np.log2(qmax - qmin + 1)),
                                                'type': "biased",
                                                }
                elif backend == 'vitis':
                    clip_ranges[tensor_name] = {'scale': float(scale)}
                elif backend == 'ppl-cuda':
                    clip_ranges[tensor_name] = float(max(-scale * (qmin - zero_point), scale * (qmax - zero_point)))

        for node in nodes_to_be_removed:
            graph.node.remove(node)
        # delete initializer
        out2node, inp2node = update_inp2node_out2node(graph)
        named_initializer = prepare_initializer(graph)
        for name, initial_data in named_initializer.items():
            if name in (out2node.keys() | inp2node.keys()):
                continue
            graph.initializer.remove(initial_data)

        clip_ranges = self.post_process_clip_ranges(clip_ranges, graph, inp2node)
        if backend == 'tensorrt':
            context = {"tensorrt": {"blob_range": clip_ranges}}
        elif backend == 'snpe':
            context = {'activation_encodings': clip_ranges, 'param_encodings': {}}
        elif backend == 'ppl':
            context = {"ppl": clip_ranges}
        elif backend == 'vitis':
            context = {'vitis': clip_ranges}
        elif backend == 'ppl-cuda':
            context = {'ppl-cuda': clip_ranges}
        output_path = os.path.dirname(onnx_path)
        context_filename = os.path.join(output_path, '{}_clip_ranges.json'.format(model_name))
        with open(context_filename, 'w') as f:
            json.dump(context, f, indent=4)
        onnx_filename = os.path.join(output_path, '{}_deploy_model.onnx'.format(model_name))
        onnx.save(model, onnx_filename)
        if backend == 'ppl-cuda':
            with open(context_filename, 'w') as f:
                for k, v in clip_ranges.items():
                    f.write('{}: {}\n'.format(k, v))
        if backend == 'vitis':
            logger.info(f"To finish xmodel converting process, call \
                $ mqbench.deploy.convert_xir -Q {onnx_filename} -C {onnx_path} -N <name> \
                    in the mqbench docker built from Dockerfile")
        logger.info("Finish deploy process.")


remove_fakequantize_and_collect_params = LinearQuantizer_process().remove_fakequantize_and_collect_params
