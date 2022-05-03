import torch
from torch.fx.graph_module import GraphModule
import torch.nn.intrinsic.qat as nniqat
import torch.nn.qat.modules.conv as qatconv
import torch.nn as nn

from mqbench.utils.registry import register_weight_equalization_function, WEIGHT_EQUALIZATION_FUNCTION
from mqbench.fake_quantize.tqt import TqtFakeQuantize

from mqbench.utils.logger import logger

COLLECT_TYPES = [nniqat.ConvBnReLU2d, nniqat.ConvBn2d, qatconv.Conv2d]
ACT_TYPES = [nn.ReLU]
POOL_TYPES = [torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d]
MATH_TYPE = [type(len)]
ALL_OP_TYPE = COLLECT_TYPES + ACT_TYPES + POOL_TYPES + MATH_TYPE
FAKE_QUANT_TYPE = [TqtFakeQuantize]


@register_weight_equalization_function(qatconv.Conv2d, qatconv.Conv2d)
def weight_equalize_conv_pair(modules, weq_pair):
    node1, node2 = tuple(weq_pair)
    weight1 = modules[node1.target].weight.data.clone()
    if modules[node1.target].bias is None:
        bias1 = None
    else:
        bias1 = modules[node1.target].bias.data.clone() 
    weight2 = modules[node2.target].weight.data.clone() 
    weight1, bias1, weight2, s = dfq_weight_equalization(weight1, bias1, weight2) 
    modules[node1.target].weight.data, modules[node2.target].weight.data = weight1, weight2
    if bias1 is not None:
        modules[node1.target].bias.data = bias1 
    logger.info(f'Weight equalizing {node1.name} and {node2.name}.')


def _get_name2node(nodes):
    name2node = {}
    for node in nodes:
        name2node[node.name] = node 
    return name2node


def _get_name2type(nodes, modules):
    name2type = {}
    for node in nodes:
        if node.target in modules:
            name2type[node.name] = type(modules[node.target])
        elif type(node.target) in MATH_TYPE:
            name2type[node.name] = type(node.target)
        else:
            name2type[node.name] = None
    return name2type


def _get_name2fanout(nodes, name2type):
    name2fanout = {} 
    for node in nodes:
        if name2type[node.name] in ALL_OP_TYPE:
            cnt = 0
            node_users = list(node.users)
            for u in node_users:
                if name2type[u.name] in ALL_OP_TYPE: 
                    cnt += 1 
                elif name2type[u.name] in FAKE_QUANT_TYPE:
                    for f in u.users:
                        if name2type[f.name] in ALL_OP_TYPE:
                            cnt += 1
            name2fanout[node.name] = cnt
        else:
            name2fanout[node.name] = 0
    return name2fanout


def get_weight_equalization_groups(model: GraphModule, **kwargs):
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    name2node = _get_name2node(nodes)
    name2type = _get_name2type(nodes, modules)
    name2fanout = _get_name2fanout(nodes, name2type)
    input = [name2node[node_name] for node_name in kwargs['input_shape_dict']]
    layer_groups = []
    for node in input:
        collect_layer_group(node, modules, layer_groups, name2fanout)
    print([[n.name for n in i] for i in layer_groups])
    convert_equalization_groups(modules, layer_groups, name2type)


def collect_layer_group(node, modules, groups, name2fanout, visited=None, group=None):
    def _end_collect(grp):
        if len(grp) > 1 and grp not in groups:
            groups.append(grp)
        return []

    visited = [] if not visited else visited
    group = [] if not group else group 

    if node in visited:
        return 
    visited.append(node)

    if node.target not in modules or type(modules[node.target]) in FAKE_QUANT_TYPE:
        pass 
    elif type(modules[node.target]) in COLLECT_TYPES:
        group.append(node)
        if name2fanout[node.name] > 1:
            group = _end_collect(group)
    elif type(modules[node.target]) in POOL_TYPES:
        if name2fanout[node.name] > 1:
            group = _end_collect(group)
    elif type(modules[node.target]) in ACT_TYPES:
        if name2fanout[node.name] > 1:
            group = _end_collect(group)
    else:
        group = _end_collect(group)

    for child in node.users:
        collect_layer_group(child, modules, groups, name2fanout, visited, group)
    _end_collect(group)


def convert_equalization_groups(modules, layer_groups, name2type):
    eq_groups = [] 
    for grp in layer_groups:
        assert len(grp) == 2, 'Multi-layers weight equalization not support.'
        type_list = [name2type[x.name] for x in grp]
        WEIGHT_EQUALIZATION_FUNCTION[type_list[0]][type_list[1]](modules, grp)


def dfq_weight_equalization(weight_1, bias_1, weight_2, s_min=1e-6, s_max=1e6, eps=0):
    groups = weight_1.shape[0] // weight_2.shape[1] 
    w1_ch, w2_ch = weight_1.shape[0] // groups, weight_2.shape[1] // groups 
    scale = torch.zeros([weight_1.shape[0]]) 
    for grp in range(groups):
        w1_ch_start, w1_ch_end = w1_ch * grp, w1_ch * (grp + 1)
        w2_ch_start, w2_ch_end = w2_ch * grp, w2_ch * (grp + 1)
        w1_ch_part = weight_1[w1_ch_start:w1_ch_end]
        w2_ch_part = weight_2[w2_ch_start:w2_ch_end]

        w1_dims = (1, 2, 3) if len(weight_1.shape) == 4 else (1)
        w2_dims = (0, 2, 3) if len(weight_2.shape) == 4 else (0)
        w1_range = w1_ch_part.abs().amax(dim=w1_dims)
        w2_range = w2_ch_part.abs().amax(dim=w2_dims)
        assert w1_range.shape == w2_range.shape, "The equalization pair's weight shape does not match!"
        s = (w1_range * w2_range + eps).sqrt() / (w2_range + eps) 
        s = torch.clip(s, s_min, s_max)
        s = torch.where((w1_range + w2_range) < 0.5, torch.ones_like(s), s)
        scale[w1_ch_start:w1_ch_end] = s 
        if bias_1 is not None:
            bias_1[w1_ch_start:w1_ch_end].mul_(1 / s)
        w1s_shape = [1] * len(weight_1.shape)
        w1s_shape[0] = -1
        weight_1[w1_ch_start:w1_ch_end].mul_(1 / s.reshape(w1s_shape))
        w2s_shape = [1] * len(weight_2.shape)
        w2s_shape[1] = -1
        weight_2[:, w2_ch_start:w2_ch_end].mul_(s.reshape(w2s_shape))

    return weight_1, bias_1, weight_2, scale 
