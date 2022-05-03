import copy
from collections import OrderedDict
from typing import List


import torch
from torch.fx import GraphModule
from torch.quantization import propagate_qconfig_
from torch.quantization.fx.qconfig_utils import get_flattened_qconfig_dict

from mqbench.utils import is_symmetric_quant, getitem2node
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.Academic)
class AcademicQuantizer(ModelQuantizer):
    """Academic setting mostly do not merge BN and leave the first and last layer to higher bits.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.io_module = {}
        self.post_act_8bit_node_name = []

    def prepare(self, model: GraphModule, qconfig):
        self._get_io_module(model)
        self._get_post_act_8bit_node_name(model)
        model = self._weight_quant(model, qconfig)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig)
        return model

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")
        wqconfig_8bit = copy.deepcopy(qconfig)
        wq_symmetry = True if is_symmetric_quant(qconfig.weight.p.keywords['qscheme']) else False
        wqconfig_8bit.weight.p.keywords['quant_min'] = -2 ** (8 - 1) if wq_symmetry else 0
        wqconfig_8bit.weight.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if wq_symmetry else 2 ** 8 - 1
        for name, module in model.named_modules():
            if name in self.io_module.keys():
                logger.info("Set layer {} to 8 bit.".format(name))
                module.qconfig = wqconfig_8bit
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model

    @property
    def function_type_to_quant_input(self) -> list:
        return self.additional_function_type

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # Conv
            torch.nn.qat.modules.conv.Conv2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
        ) + self.additional_module_type

    def _get_post_act_8bit_node_name(self, model):
        for node in self.io_module.values():
            for _arg in node.args:
                if isinstance(_arg, torch.fx.node.Node):
                    self.post_act_8bit_node_name.append(_arg.name)

    def _get_io_module(self, model):
        total_args = []
        nodes = list(model.graph.nodes)
        for node in nodes:
            the_first_layer = False
            for _arg in node.args:
                if isinstance(_arg, torch.fx.node.Node):
                    if _arg.op == 'placeholder':
                        the_first_layer = True
                    total_args.append(_arg.name)
            if the_first_layer:
                self.io_module[node.target] = node
            if node.op == 'output':
                for _arg in node.args:
                    if isinstance(_arg, torch.fx.node.Node):
                        self.io_module[_arg.target] = _arg

    def _find_act_quants(self, model: GraphModule) -> List:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        g2node = getitem2node(model)
        for node in nodes:
            if ((node.op == "call_module" and node.target in self.exclude_module_name) or
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or
                    node.name in self.exclude_node_name) and node.name not in self.additional_node_name:
                logger.info("Exclude skip: {}".format(node.name))
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input) or node.name in self.additional_node_name:
                input_node_list = self._flatten_args(node.args)
                # Means this is not Tensor + Tensor.
                if not all([isinstance(_node, torch.fx.node.Node) for _node in input_node_list]):
                    continue
                for _node in input_node_list:
                    if self._is_implicit_merge(modules, (node, _node)):
                        logger.info("Implicit merge: {} + {}".format(_node.name, node.name))
                        continue
                    if _node in g2node:
                        _node = g2node[_node]
                    node_need_to_quantize_output.append(_node)
        return node_need_to_quantize_output

    def _insert_fake_quantize_for_act_quant(self, model: GraphModule, qconfig):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()

        aqconfig_8bit = copy.deepcopy(qconfig.activation)
        aq_symmetry = True if is_symmetric_quant(qconfig.activation.p.keywords['qscheme']) else False
        aqconfig_8bit.p.keywords['quant_min'] = -2 ** (8 - 1) if aq_symmetry else 0
        aqconfig_8bit.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if aq_symmetry else 2 ** 8 - 1
        for node in node_to_quantize_output:
            if node.name in self.post_act_8bit_node_name:
                logger.info("Set {} post act quantize to 8 bit.".format(node.name))
                fake_quantizer = aqconfig_8bit()
            else:
                fake_quantizer = qconfig.activation()
            quantizer_name = node.name + quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        model.recompile()
        model.graph.lint()
        return model