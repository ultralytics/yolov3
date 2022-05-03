from .model_quantizer import ModelQuantizer
from .academic_quantizer import AcademicQuantizer
from .openvino_quantizer import OPENVINOQuantizer
from .vitis_quantizer import VitisQuantizer
from .total_int_quantizer import TotalINTQuantizer
from .tensorrt_quantizer import TRTModelQuantizer, TensorrtNLPQuantizer
from .tengine_u8_quantizer import TengineQuantizer
from .onnx_qnn_quantizer import ONNXQNNQuantizer
from .nlp_quantizer import AcademicNLPQuantizer