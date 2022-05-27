import torch

from mqbench.utils.logger import logger


def enable_calibration(model):
    logger.info('Enable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug(f'Enable observer and Disable quant: {name}')
            submodule.enable_observer()
            submodule.disable_fake_quant()

def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info(f'Enable observer and Disable quantize for {quantizer_type}')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug(f'Enable observer and Disable quant: {name}')
            submodule.enable_observer()
            submodule.disable_fake_quant()

def enable_calibration_quantization(model, quantizer_type='fake_quant'):
    logger.info(f'Enable observer and Enable quantize for {quantizer_type}')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug(f'Enable observer and Enable quant: {name}')
            submodule.enable_observer()
            submodule.enable_fake_quant()

def enable_quantization(model):
    logger.info('Disable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug(f'Disable observer and Enable quant: {name}')
            submodule.disable_observer()
            submodule.enable_fake_quant()


def disable_all(model):
    logger.info('Disable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug(f'Disable observer and Disable quantize: {name}')
            submodule.disable_observer()
            submodule.disable_fake_quant()
