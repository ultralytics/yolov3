import copy
import operator
from collections import OrderedDict
from typing import Any

import torch
from torch.fx import GraphModule
from torch.quantization import propagate_qconfig_
from torch.quantization.fx.qconfig_utils import get_flattened_qconfig_dict
from torch.quantization.quantize_fx import _fuse_fx

from mqbench.utils import is_symmetric_quant
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.OPENVINO)
class OPENVINOQuantizer(ModelQuantizer):
    """OPENVINO type, activation is scaled to [0, 255] when qscheme is symmetric
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.academic_mode = extra_quantizer_dict.get('academic_mode', False)

    @property
    def _passed_func_type(self):
        academic_pass_type = (operator.getitem, getattr)
        if self.academic_mode:
            return academic_pass_type
        else:
            return academic_pass_type + (torch.cat, )

    @property
    def _passed_module_type(self):
        return tuple()

    @property
    def _linear_module_node(self) -> tuple:
        return (
            torch.nn.qat.modules.conv.Conv2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn1d,
            torch.nn.qat.modules.conv.Conv2d,
            torch.nn.qat.modules.linear.Linear,
        )

    @property
    def _propagated_pattern(self) -> tuple:
        prev_nodes_pattern = {
            'func_type': (torch.nn.functional.max_pool2d, torch.flatten),
            'module_type': (torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.Flatten)
        }

        cur_nodes_pattern = {
            'func_type': (torch.nn.functional.conv2d, torch.nn.functional.conv1d, torch.nn.functional.conv3d, torch.matmul),
            'module_type': self._linear_module_node,
        }

        return (prev_nodes_pattern, cur_nodes_pattern)

    @property
    def function_type_to_quant_input(self) -> list:
        return [
            operator.add,
            torch.nn.functional.adaptive_avg_pool2d,
            torch.nn.functional.max_pool2d,
            torch.nn.functional.avg_pool2d,
            torch.flatten,
            'mean',
            'sum',
            torch.nn.functional.interpolate,
        ]

    @property
    def module_type_to_quant_input(self) -> tuple:
        if self.academic_mode:
            return (
                # Conv
                torch.nn.qat.modules.conv.Conv2d,
                # Linear
                torch.nn.qat.modules.linear.Linear,
                # Pooling
                torch.nn.modules.pooling.AvgPool2d,
                torch.nn.modules.pooling.AdaptiveAvgPool2d,
                torch.nn.modules.pooling.MaxPool2d,
                # Prelu
                # TODO torch.nn.PReLU,
                torch.nn.modules.Upsample,
            ) + self.additional_module_type
        else:
            return (
                # Conv
                torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
                torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d,
                torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d,
                torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
                torch.nn.intrinsic.qat.modules.conv_fused.ConvBn1d,
                torch.nn.qat.modules.conv.Conv2d,
                # Linear
                torch.nn.qat.modules.linear.Linear,
                # Pooling
                torch.nn.modules.pooling.AvgPool2d,
                torch.nn.modules.pooling.AdaptiveAvgPool2d,
                torch.nn.modules.pooling.MaxPool2d,
                # Prelu
                # TODO torch.nn.PReLU,
                torch.nn.modules.Upsample,
            ) + self.additional_module_type

    @property
    def module_type_to_quant_unsigned(self) -> tuple:
        if self.academic_mode:
            return (torch.nn.modules.ReLU, )
        else:
            return (
                torch.nn.modules.ReLU,
                torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
                torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d,
                torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d,
            )

    @property
    def function_type_maybe_unsigned(self) -> tuple:
        return self.function_type_to_quant_input

    @property
    def function_type_to_quant_unsigned(self) -> tuple:
        return (torch.nn.functional.relu, )

    @property
    def module_type_maybe_unsigned(self) -> tuple:
        return (torch.nn.Upsample, torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.pooling.AvgPool2d, torch.nn.modules.pooling.AdaptiveAvgPool2d)

    def prepare(self, model: GraphModule, qconfig):
        if not self.academic_mode:
            model = _fuse_fx(model, self.extra_fuse_dict)
        model = self._weight_quant(model, qconfig)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig)
        return model

    def _find_act_quants(self, model: GraphModule) -> list:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []

        def quanlified_node(node):
            return (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and node.target in self.function_type_to_quant_input) or node.op == 'placeholder'

        def passed_node(node):
            return (node.op == 'call_function' and node.target in self._passed_func_type) or \
                (node.op == 'call_module' and isinstance(modules[node.target], self._passed_module_type))

        prev_nodes_pattern, cur_nodes_pattern = self._propagated_pattern

        def node_in_pattern(node, pattern):
            return ((node.op == 'call_function' or node.op == 'call_method') and node.target in pattern['func_type']) or \
                (node.op == "call_module" and isinstance(modules[node.target], pattern['module_type'])) 

        def propagated_pattern(prev_node, cur_node):
            return node_in_pattern(prev_node, prev_nodes_pattern) and node_in_pattern(cur_node, cur_nodes_pattern)

        for node in nodes:
            if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or \
                    node.name in self.exclude_node_name:
                continue
            if passed_node(node):
                continue
            if node.op == 'placeholder':
                node_need_to_quantize_output.append(node)
                continue
            is_output = False
            # last layer do not quantize
            for next_node in node.users:
                if next_node.op == 'output':
                    is_output = True
                    break
            if is_output:
                continue
            # check propagated pattern
            is_propagated_pattern = False
            for next_node in node.users:
                if propagated_pattern(node, next_node):
                    is_propagated_pattern = True
                    break
            if is_propagated_pattern:
                continue
            for next_node in node.users:
                if quanlified_node(next_node):
                    node_need_to_quantize_output.append(node)
                    break
        return node_need_to_quantize_output

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")
        wqconfig_8bit = copy.deepcopy(qconfig)
        wq_symmetry = True if is_symmetric_quant(qconfig.weight.p.keywords['qscheme']) else False
        numbits = 8
        logger.info('Now all weight quantizers will effectively use only 7 bits out of 8 bits. This resolves the overflow issue problem on AVX2 and AVX-512 machines.')
        wqconfig_8bit.weight.p.keywords['quant_min'] = -2 ** (numbits - 2) if wq_symmetry else 0
        wqconfig_8bit.weight.p.keywords['quant_max'] = 2 ** (numbits - 2) - 1 if wq_symmetry else 2 ** (numbits - 1) - 1
        wqconfig_8bit.weight.p.keywords['factory_kwargs'] = {'not_calc_quant_min_max': True}
        if self.academic_mode and wq_symmetry:
            wqconfig_8bit.weight.p.keywords['quant_min'] = -2 ** (numbits - 2) + 1
            wqconfig_8bit.weight.p.keywords['quant_max'] = 2 ** (numbits - 2) - 1
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': wqconfig_8bit})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model


    def _insert_fake_quantize_for_act_quant(
            self,
            model: GraphModule,
            qconfig: Any):
        graph = model.graph
        modules = dict(model.named_modules())
        nodes = list(model.graph.nodes)

        quantizer_postfix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()

        aqconfig_8bit = copy.deepcopy(qconfig.activation)
        aq_symmetry = True if is_symmetric_quant(qconfig.activation.p.keywords['qscheme']) else False
        aqconfig_8bit.p.keywords['quant_min'] = 0
        aqconfig_8bit.p.keywords['quant_max'] = 2 ** 8 - 1
        aqconfig_8bit.p.keywords['factory_kwargs'] = {'not_calc_quant_min_max': True}

        def maybe_unsigned(node):
            return ((node.op == 'call_function' or node.op == 'call_method') and node.target in self.function_type_maybe_unsigned) or \
                (node.op == "call_module" and isinstance(modules[node.target], self.module_type_maybe_unsigned))

        def real_unsigned(node):
            return ((node.op == 'call_function' or node.op == 'call_method') and node.target in self.function_type_to_quant_unsigned) or \
                (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_unsigned))

        for node in node_to_quantize_output:
            if aq_symmetry:
                if real_unsigned(node):
                    logger.info("Set {} post act quantize to 8 bit unsigned type.".format(node.name))
                    fake_quantizer = aqconfig_8bit()
                elif maybe_unsigned(node):
                    is_unsigned = False
                    # bfs to determin1e whether it should be set unsigned activation
                    queue = [(node, -1)]
                    bfs_result = dict()
                    while len(queue) > 0:
                        cur_node, level = queue.pop(0)
                        for input_node in cur_node.args:
                            if isinstance(input_node, torch.fx.node.Node):
                                queue.append((input_node, level + 1))
                        cur_node_is_unsigned = None
                        if isinstance(cur_node.target, str) and cur_node.target.endswith(quantizer_postfix):
                            last_fakequantize = getattr(model, cur_node.target)
                            cur_node_is_unsigned = last_fakequantize.quant_min == 0
                        elif real_unsigned(node):
                            cur_node_is_unsigned = True

                        if cur_node_is_unsigned is not None:
                            if level not in bfs_result:
                                if len(bfs_result) > 0:
                                    break
                                else:
                                    bfs_result[level] = cur_node_is_unsigned
                            else:
                                bfs_result[level] = bfs_result[level] and cur_node_is_unsigned
                    queue.clear()
                    for key in bfs_result:
                        is_unsigned = bfs_result[key]
                        break
                    fake_quantizer = aqconfig_8bit() if is_unsigned else qconfig.activation()
                    if is_unsigned:
                        logger.info("Set {} post act quantize to 8 bit unsigned type.".format(node.name))
                else:
                    fake_quantizer = qconfig.activation()
            else:
                fake_quantizer = qconfig.activation()
            quantizer_name = node.name + quantizer_postfix
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        model.recompile()
        model.graph.lint()
        return model