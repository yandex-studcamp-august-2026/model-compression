from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_SHAPE = (1, 3, 32, 32)
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "logits"
ONNX_OPSET = 18
PARITY_SAMPLES = 3
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4
TENSORRT_FP32_ATOL = 1e-4
TENSORRT_FP32_RTOL = 1e-4
TENSORRT_FP16_ATOL = 1e-3
TENSORRT_FP16_RTOL = 1e-2


class TinyForward(nn.Module):
    def __init__(self, weights_path: str | Path):
        super().__init__()
        self.network = nn.Conv2d(3, 19, kernel_size=1)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=True)
        self.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.network(pixel_values)


def evaluate_quality(predict, dataset_root: Path) -> dict[str, float]:
    if not (dataset_root / "sample.ready").is_file():
        raise ValueError("validation fixture is missing")
    logits = np.asarray(predict(torch.zeros(INPUT_SHAPE))[0])
    prediction = logits.argmax(axis=1)
    target = np.zeros(prediction.shape, dtype=np.int64)
    intersection = float(np.count_nonzero((prediction == 0) & (target == 0)))
    union = float(np.count_nonzero((prediction == 0) | (target == 0)))
    return {"mIoU": intersection / union}
