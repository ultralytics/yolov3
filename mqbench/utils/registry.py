from collections import OrderedDict


DEFAULT_MODEL_QUANTIZER = OrderedDict()


def register_model_quantizer(backend_type):
    def insert(quantizer_cls):
        DEFAULT_MODEL_QUANTIZER[backend_type] = quantizer_cls
        return quantizer_cls
    return insert

BACKEND_DEPLOY_FUNCTION = OrderedDict()


def register_deploy_function(backend_type):
    def insert(func):
        if backend_type in BACKEND_DEPLOY_FUNCTION:
            BACKEND_DEPLOY_FUNCTION[backend_type].append(func)
        else:
            BACKEND_DEPLOY_FUNCTION[backend_type] = [func]
        return func
    return insert


FUSED_MODULE_CONVERT_FUNCTION = OrderedDict()


def register_convert_function(module_type):
    def insert(func):
        FUSED_MODULE_CONVERT_FUNCTION[module_type] = func
        return func
    return insert


WEIGHT_EQUALIZATION_FUNCTION = OrderedDict() 


def register_weight_equalization_function(layer1, layer2):
    def insert(func):
        WEIGHT_EQUALIZATION_FUNCTION[layer1] = {layer2: func}
        return func
    return insert