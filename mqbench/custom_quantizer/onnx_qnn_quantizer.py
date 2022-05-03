import operator
from typing import Dict, Callable, List

import torch
from torch.fx import GraphModule
from torch.quantization.quantization_mappings import get_default_qat_module_mappings
from torch.quantization.utils import get_combined_dict


import mqbench.nn as qnn 
import mqbench.nn.intrinsic as qnni 
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.ONNX_QNN)
class ONNXQNNQuantizer(ModelQuantizer):
    """Quantize model according to TVM ONNX frontend.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)

    @property
    def _relu_module_type(self):
        return (torch.nn.ReLU, torch.nn.ReLU6)

    @property
    def _relu_function_type(self):
        return (torch.nn.functional.relu, torch.nn.functional.relu6)

    def _find_act_quants(self, model: GraphModule) -> List:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = super()._find_act_quants(model)
        for node in nodes:
            if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or \
                    node.name in self.exclude_node_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input):
                # Add current node if not merge relu.
                for next_node in node.users:
                    if not ((next_node.op == 'call_function' and next_node.target in self._relu_function_type) or (
                            next_node.op == 'call_module' and isinstance(modules[next_node.target], self._relu_module_type))):
                        node_need_to_quantize_output.append(node)
                    else:
                        node_need_to_quantize_output.append(next_node)
        return node_need_to_quantize_output

    def _qat_swap_modules(self, root: GraphModule, additional_qat_module_mapping: Dict[Callable, Callable]):
        all_mappings = get_combined_dict(
            get_default_qat_module_mappings(), additional_qat_module_mapping)
        # There is no QLinearFC in ONNX for now.
        del(all_mappings[torch.nn.modules.linear.Linear])
        del(all_mappings[torch.nn.modules.linear._LinearWithBias])
        del(all_mappings[torch.nn.intrinsic.modules.fused.LinearReLU])
        del(all_mappings[qnni.modules.fused.LinearBn1d])
        root = self._convert(root, all_mappings, inplace=True)
        return root

    @property
    def function_type_to_quant_input(self) -> list:
        return [
            operator.add,
            # TODO operator.mul,
            # TODO torch.cat,
            torch.nn.functional.adaptive_avg_pool2d
            # sigmoid
            # TODO torch.nn.functional.sigmoid
        ]

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # Conv
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
            torch.nn.qat.Conv2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
            qnn.intrinsic.qat.LinearBn1d,
            # Pooling
            torch.nn.modules.pooling.AvgPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            # Prelu
            # TODO torch.nn.PReLU,
        )

    @property
    def implicit_merge_patterns(self) -> list:
        # Layers which do not need quantize among them.
        # In reversed order!
        return [
            (torch.nn.ReLU, operator.add)
        ]