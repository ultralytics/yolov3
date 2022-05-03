import torch
from torch.fx import GraphModule

from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.Tengine_u8)
class TengineQuantizer(ModelQuantizer):
    """
    Tengine needs de-quantization parameters for output.

    Parameters
    ----------
    ModelQuantizer : _type_
        _description_
    """
    @property
    def _passed_func_type(self):
        return (
            torch.flatten,
        )

    @property
    def _passed_module_type(self):
        # TODO: softmax
        return ()

    @property
    def implicit_merge_patterns(self) -> list:
        # Layers which do not need quantize among them.
        # In reversed order!
        return []

    @property
    def function_type_to_quant_input(self) -> list:
        return [
            torch.cat,
            torch.nn.functional.hardswish,
            torch.nn.functional.sigmoid
        ] + super().function_type_to_quant_input

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            torch.nn.Hardswish,
            torch.nn.Sigmoid,
        ) + super().module_type_to_quant_input

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
            elif node.op == "output":
                for _arg in node.args:
                    if isinstance(_arg, torch.fx.node.Node):
                        node_need_to_quantize_output.append(_arg)
        return node_need_to_quantize_output