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
    get_constant_inputs
)


class NNIE_process(object):
    def gen_gfpq_param_file(self, graph, clip_val):
        nnie_exclude_layer_type = ['Flatten', 'Relu', 'PRelu', 'Sigmoid', 'Reshape',
                                   'Softmax', 'CaffeSoftmax', 'Clip', 'GlobalAveragePool', 'Mul']
        interp_layer_cnt = 0
        gfpq_param_dict = {}
        for idx, node in enumerate(graph.node):
            # We can not support NNIE group conv.
            # Group conv need group-size input params.
            if node.op_type == 'Conv' and node.attribute[1].i != 1:
                continue

            layer_input_tensor = []
            for in_tensor in node.input:
                if in_tensor in clip_val:
                    clip_value = clip_val[in_tensor]
                    layer_input_tensor.append(float(clip_value))
                # Upsample layer only reserve one input.
                if node.op_type in ['Upsample', 'DynamicUpsample']:
                    break

            if node.op_type not in nnie_exclude_layer_type and len(layer_input_tensor) > 0:
                gfpq_param_dict[node.name] = layer_input_tensor

            # Upsample ---> Upsample + Permute in NNIE.
            if node.op_type in ['Upsample', 'DynamicUpsample']:
                interp_layer_name = node.name
                gfpq_param_dict[interp_layer_name + '_permute_' + str(interp_layer_cnt)] = gfpq_param_dict[interp_layer_name]
                interp_layer_cnt += 1
        return gfpq_param_dict

    def remove_fakequantize_and_collect_params(self, onnx_path, model_name):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.replace_resize_op_with_upsample(graph, out2node)
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)

        nodes_to_be_removed = []
        clip_ranges = {}
        for node in graph.node:
            if node.op_type == 'NNIEQuantize':
                next_nodes = inp2node[node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    next_node, idx = next_nodes[0]
                    next_node.input[idx] = node.input[0]
                    # clip weights
                    tensor_name = node.input[0]
                    data = name2data[tensor_name]
                    clip_range = name2data[node.input[1]]
                    new_data = np.clip(data, -clip_range, clip_range)
                    new_data = numpy_helper.from_array(new_data)
                    named_initializer[tensor_name].raw_data = new_data.raw_data
                    logger.info(f'Clip weights {tensor_name} to range [{-clip_range}, {clip_range}].')
                else:
                    # fake quantize for activations
                    clip_ranges[node.input[0]] = name2data[node.input[1]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = node.input[0]

                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))

        for node in nodes_to_be_removed:
            graph.node.remove(node)

        gfpq_param_dict = self.gen_gfpq_param_file(graph, clip_ranges)

        output_path = os.path.dirname(onnx_path)
        gfpq_param_file = os.path.join(output_path, '{}_gfpq_param_dict.json'.format(model_name))
        with open(gfpq_param_file, 'w') as f:
            json.dump({"nnie": {"gfpq_param_dict": gfpq_param_dict}}, f, indent=4)
        onnx_filename = os.path.join(output_path, '{}_deploy_model.onnx'.format(model_name))
        onnx.save(model, onnx_filename)
        logger.info("Finish deploy process.")


remove_fakequantize_and_collect_params_nnie = NNIE_process().remove_fakequantize_and_collect_params