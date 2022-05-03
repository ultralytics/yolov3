import torch
from torch.fx import GraphModule

from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.PPLCUDA)
@register_model_quantizer(BackendType.SNPE)
@register_model_quantizer(BackendType.PPLW8A16)
class TotalINTQuantizer(ModelQuantizer):
    """There is only INT8 calculations in the model.
    We quantize the input tensors and output tensors of all layers,
    except those in _passed_func_type and _passed_module_type.
    For example add + relu pattern, there is no need to insert fake
    quantize node between them.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)

    @property
    def _passed_func_type(self):
        return (
            torch.nn.functional.relu, 
            torch.nn.functional.relu6,
            torch.flatten
        )

    @property
    def _passed_module_type(self):
        return (
            torch.nn.ReLU,
            torch.nn.ReLU6
        )

    def _find_act_quants(self, model: GraphModule) -> list:
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
                for next_node in node.users:
                    if not ((next_node.op == 'call_function' and next_node.target in self._passed_func_type) or
                            (next_node.op == 'call_module' and isinstance(modules[next_node.target], self._passed_module_type))):
                        node_need_to_quantize_output.append(node)
                    else:
                        node_need_to_quantize_output.append(next_node)
        return node_need_to_quantize_output