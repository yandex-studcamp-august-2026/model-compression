from experiments._shared import cityscapes as _cityscapes

CLASS_AXIS = _cityscapes.CLASS_AXIS
INPUT_NAME = _cityscapes.INPUT_NAME
INPUT_SHAPE = _cityscapes.INPUT_SHAPE
ONNX_OPSET = _cityscapes.ONNX_OPSET
OUTPUT_NAME = _cityscapes.OUTPUT_NAME
PARITY_ATOL = _cityscapes.PARITY_ATOL
PARITY_RTOL = _cityscapes.PARITY_RTOL
PARITY_SAMPLES = _cityscapes.PARITY_SAMPLES
QUALITY_ATOL = _cityscapes.QUALITY_ATOL
SEMANTIC_OUTPUT_NAME = _cityscapes.SEMANTIC_OUTPUT_NAME
TENSORRT_FP16_ATOL = _cityscapes.TENSORRT_FP16_ATOL
TENSORRT_FP16_RTOL = _cityscapes.TENSORRT_FP16_RTOL
TENSORRT_FP32_ATOL = _cityscapes.TENSORRT_FP32_ATOL
TENSORRT_FP32_RTOL = _cityscapes.TENSORRT_FP32_RTOL
make_inputs = _cityscapes.make_inputs
evaluate_quality = _cityscapes.evaluate_cityscapes_quality


class Model(_cityscapes.CityscapesSegFormerB1):
    pass
