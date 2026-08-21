"""SegFormer adapters and fixed-subset Cityscapes evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

INPUT_SHAPE = (1, 3, 512, 1024)
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "logits"
SEMANTIC_OUTPUT_NAME = OUTPUT_NAME
CLASS_AXIS = 1
ONNX_OPSET = 18
PARITY_SAMPLES = 3
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4
QUALITY_ATOL = 1e-4
TENSORRT_FP32_ATOL = 1e-4
TENSORRT_FP32_RTOL = 1e-4
TENSORRT_FP16_ATOL = 1e-3
TENSORRT_FP16_RTOL = 1e-2

_NUM_CLASSES = 19
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class _CityscapesSegFormer(SegformerForSemanticSegmentation):
    """Cityscapes SegFormer with strict loading of a clean state dict."""

    depths: tuple[int, int, int, int]
    hidden_sizes: tuple[int, int, int, int]
    decoder_hidden_size: int

    def __init__(self, weights_path: Path) -> None:
        config = SegformerConfig(
            num_labels=_NUM_CLASSES,
            id2label={index: str(index) for index in range(_NUM_CLASSES)},
            label2id={str(index): index for index in range(_NUM_CLASSES)},
            semantic_loss_ignore_index=255,
            depths=self.depths,
            hidden_sizes=self.hidden_sizes,
            decoder_hidden_size=self.decoder_hidden_size,
            num_attention_heads=(1, 2, 5, 8),
        )
        super().__init__(config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise TypeError("checkpoint must contain a non-empty state dict")
        if any(not isinstance(name, str) for name in state):
            raise TypeError("checkpoint state-dict keys must be strings")
        if any(not isinstance(value, Tensor) for value in state.values()):
            raise TypeError("checkpoint state-dict values must be tensors")
        wrapped = [name.startswith("network.") for name in state]
        if any(wrapped) and not all(wrapped):
            raise ValueError("checkpoint mixes wrapped and direct parameter names")
        if all(wrapped):
            state = {
                name.removeprefix("network."): value for name, value in state.items()
            }
        self.load_state_dict(state, strict=True)

    def forward(self, pixel_values: Tensor) -> Tensor:
        logits = super().forward(pixel_values=pixel_values, return_dict=False)[0]
        return F.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class CityscapesSegFormerB0(_CityscapesSegFormer):
    """SegFormer-B0 student architecture."""

    depths = (2, 2, 2, 2)
    hidden_sizes = (32, 64, 160, 256)
    decoder_hidden_size = 256


class CityscapesSegFormerB1(_CityscapesSegFormer):
    """SegFormer-B1 reference-teacher architecture."""

    depths = (2, 2, 2, 2)
    hidden_sizes = (64, 128, 320, 512)
    decoder_hidden_size = 256


class CityscapesSegFormerB2(_CityscapesSegFormer):
    """SegFormer-B2 reference-teacher architecture."""

    depths = (3, 4, 6, 3)
    hidden_sizes = (64, 128, 320, 512)
    decoder_hidden_size = 768


def make_inputs(seed: int) -> Tensor:
    """Generate deterministic image-like normalized input for conversion checks."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.rand(INPUT_SHAPE, dtype=torch.float32, generator=generator)
    mean = torch.from_numpy(_MEAN).view(1, 3, 1, 1)
    std = torch.from_numpy(_STD).view(1, 3, 1, 1)
    return (image - mean) / std


def evaluate_cityscapes_quality(
    predict: Any,
    dataset_root: Path,
) -> dict[str, float]:
    """Compute mIoU on the immutable, preprocessed Cityscapes subset."""
    sample_paths = sorted(dataset_root.glob("samples/*.npz"))
    if not sample_paths:
        raise ValueError("Cityscapes validation subset contains no .npz samples")

    confusion = np.zeros((_NUM_CLASSES, _NUM_CLASSES), dtype=np.int64)
    for sample_path in sample_paths:
        with np.load(sample_path, allow_pickle=False) as sample:
            image = np.asarray(sample["image"], dtype=np.uint8)
            target = np.asarray(sample["label"], dtype=np.uint8)
        if image.shape != (512, 1024, 3) or target.shape != (512, 1024):
            raise ValueError(f"invalid validation sample shape: {sample_path}")

        normalized = image.astype(np.float32) / 255.0
        normalized = (normalized - _MEAN) / _STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        outputs = predict(tensor.contiguous())
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
            raise ValueError("predict must return one logits output")
        logits = np.asarray(outputs[0])
        if logits.shape != (1, _NUM_CLASSES, 512, 1024):
            raise ValueError(f"invalid logits shape: {logits.shape}")

        prediction = np.argmax(logits, axis=1)[0]
        valid = target != 255
        encoded = _NUM_CLASSES * target[valid].astype(np.int64) + prediction[
            valid
        ].astype(np.int64)
        confusion += np.bincount(
            encoded,
            minlength=_NUM_CLASSES * _NUM_CLASSES,
        ).reshape(_NUM_CLASSES, _NUM_CLASSES)

    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    present = union > 0
    if not np.any(present):
        raise ValueError("Cityscapes validation subset has no labelled classes")
    return {"mIoU": float(np.mean(intersection[present] / union[present]))}
