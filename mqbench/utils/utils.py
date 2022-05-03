import copy

import torch
import torch.fx
from torch.fx import GraphModule
from torch.nn import Module

USE_LINK = False
USE_DDP = False

try:
    import spring.linklink as link
    assert link.is_initialized()
    USE_LINK = True
except (ModuleNotFoundError, AssertionError):
    import torch.distributed as dist
    if torch.distributed.is_initialized():
        USE_DDP = True


def sync_tensor(tensor):
    global USE_LINK
    global USE_DDP
    if USE_LINK:
        if tensor.is_cuda is True:
            tensor.data = tensor.data / link.get_world_size()
            link.allreduce(tensor.data)
    elif USE_DDP:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


def pot_quantization(tensor: torch.Tensor, mode='round'):
    log2t = torch.log2(tensor)
    if mode == 'round':
        log2t = (torch.round(log2t) - log2t).detach() + log2t
    else:
        assert mode == 'floor' 
        log2t = (torch.floor(log2t) - log2t).detach() + log2t
    return 2 ** log2t



def is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


class no_jit_trace:
    def __enter__(self):
        # pylint: disable=protected-access
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def is_tracing_state():
    return torch._C._get_tracing_state()


def deepcopy_graphmodule(gm: GraphModule):
    """Rewrite the deepcopy of GraphModule. (Copy its 'graph'.)

    Args:
        gm (GraphModule): 

    Returns:
        GraphModule: A deepcopied gm.
    """
    copied_gm = copy.deepcopy(gm)
    copied_gm.graph = copy.deepcopy(gm.graph)
    return copied_gm


def deepcopy_mixedmodule(mm: Module, module_list: list):
    """Support for `module_list` which splits modules' nn part and post precess.

    Args:
        mm (nn.Module)
        module_list (list): the children of the mm who are a GraphModule.

    Returns:
        nn.Module
    """
    copied_mm = copy.deepcopy(mm)
    for mname in module_list:
        mod = getattr(mm, mname)
        child_graph = copy.deepcopy(mod.graph)
        copied_child = getattr(copied_mm, mname)
        setattr(copied_child, 'graph', child_graph)
    return copied_mm


def getitem2node(model: GraphModule) -> dict:
    def _update_getitem_path(getitem_args_dict):
        for node in getitem_args_dict:
            args_list = getitem_args_dict[node]
            while args_list[0] in getitem_args_dict:
                args_list = getitem_args_dict[args_list[0]] + args_list[1:]
            getitem_args_dict[node] = args_list
        return getitem_args_dict

    def _getitem_from_args(args, original_args_dict):
        ret = original_args_dict
        for a in args:
            try:
                ret = ret[a]
            except (IndexError, KeyError):
                return {}
        return ret 
    import operator
    nodes = list(model.graph.nodes)
    # the getitem's call graph
    getitem_args_dict = {}
    # the dict used in the model 
    original_key_dict = {}
    getitem2node = {}
    for node in nodes:
        # update the getitems
        if node.target == operator.getitem:
            getitem_args_dict[node] = list(node.args)
            getitem_args_dict = _update_getitem_path(getitem_args_dict)
            for _node in getitem_args_dict:
                if _node in getitem2node:
                    continue
                val = _getitem_from_args(getitem_args_dict[_node], original_key_dict)
                if isinstance(val, torch.fx.node.Node):
                    getitem2node[_node] = val
        elif node.target == 'update':
            if node.args[0] not in original_key_dict:
                original_key_dict[node.args[0]] = {}
            original_key_dict[node.args[0]].update(node.args[1])
    return getitem2node


def _fix_succ_recursivly(args, target_node, inserted_node):
    # List / Tuple
    if isinstance(args, (list, tuple)):
        _tmp = list(args)
        for _i, _arg in enumerate(args):
            if _arg == target_node:
                _tmp[_i] = inserted_node
            elif isinstance(_arg, tuple):
                _tmp[_i] = _fix_succ_recursivly(_arg, target_node, inserted_node)
            elif isinstance(_arg, list):
                _tmp[_i] = list(_fix_succ_recursivly(_arg, target_node, inserted_node))
            elif isinstance(_arg, dict):
                _tmp[_i] = _fix_succ_recursivly(_arg, target_node, inserted_node)
        return tuple(_tmp)
    # Dict
    elif isinstance(args, dict):
        _tmp = {}
        for k, v in args.items():
            if v == target_node:
                _tmp[k] = inserted_node
            elif not isinstance(v, torch.fx.node.Node):
                _tmp[k] = _fix_succ_recursivly(v, target_node, inserted_node)
            else:
                _tmp[k] = v
        return _tmp
    else:
        raise NotImplementedError('{} can not be handled now.'.format(type(args)))


def topology_order(model):
    node2idx = {}
    for idx, node in enumerate(model.graph.nodes):
        node2idx[node] = idx 
    return node2idx