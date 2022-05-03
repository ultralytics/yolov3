import torch

from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.Academic_NLP)
class AcademicNLPQuantizer(ModelQuantizer):
    """
    NLP model quantizer for Academic settings. Should not de 8bit for
    first / last layer.
    We should uantize Linear / Embedding weights.
    Linear / Matmul layer inputs(activations).
    """
    @property
    def function_type_to_quant_input(self) -> list:
        return [
            # Matmul in MSA
            torch.matmul
        ] + self.additional_function_type

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # Linear
            torch.nn.qat.modules.linear.Linear,
        ) + self.additional_module_type