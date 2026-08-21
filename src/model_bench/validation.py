from __future__ import annotations

import math
from typing import Any

MIN_SEGMENTATION_PIXEL_AGREEMENT = 0.9999


def compare_outputs(
    reference: tuple[Any, ...],
    candidate: list[Any] | tuple[Any, ...],
    names: tuple[str, ...],
    atol: float,
    rtol: float,
    *,
    task: str | None = None,
    semantic_output_name: str | None = None,
    class_axis: int = 1,
) -> dict[str, Any]:
    import numpy as np

    if len(reference) != len(candidate) or len(reference) != len(names):
        raise ValueError(
            "Reference, candidate, and output names must have equal lengths"
        )

    outputs: dict[str, Any] = {}
    passed = True
    for name, expected_value, actual_value in zip(
        names, reference, candidate, strict=True
    ):
        expected = _as_numpy(expected_value)
        actual = _as_numpy(actual_value)
        if expected.shape != actual.shape:
            outputs[name] = {
                "passed": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
            }
            passed = False
            continue
        finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
        difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
        max_absolute = float(difference.max(initial=0.0))
        max_relative = float((difference / denominator).max(initial=0.0))
        expected_flat = expected.astype(np.float64).ravel()
        actual_flat = actual.astype(np.float64).ravel()
        norm = float(np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat))
        cosine = 1.0 if norm == 0.0 and np.array_equal(expected, actual) else 0.0
        if norm > 0.0:
            cosine = float(np.dot(expected_flat, actual_flat) / norm)
        close = bool(finite and np.allclose(expected, actual, atol=atol, rtol=rtol))
        outputs[name] = {
            "passed": close,
            "shape": list(expected.shape),
            "max_absolute_error": max_absolute,
            "max_relative_error": max_relative,
            "cosine_similarity": cosine if math.isfinite(cosine) else None,
        }
        if task == "segmentation" and name == semantic_output_name:
            semantic = _segmentation_agreement(expected, actual, class_axis)
            outputs[name]["segmentation_agreement"] = semantic
            outputs[name]["passed"] = bool(
                outputs[name]["passed"]
                and semantic["pixel_agreement"] >= MIN_SEGMENTATION_PIXEL_AGREEMENT
            )
            close = outputs[name]["passed"]
        passed = passed and close
    return {"passed": passed, "atol": atol, "rtol": rtol, "outputs": outputs}


def _as_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _segmentation_agreement(
    reference: Any, candidate: Any, class_axis: int
) -> dict[str, float]:
    """Compare segmentation label maps without treating agreement as accuracy."""
    import numpy as np

    if reference.ndim < 3:
        raise ValueError("Segmentation logits must have at least three dimensions")
    if not -reference.ndim <= class_axis < reference.ndim:
        raise ValueError("CLASS_AXIS is outside the segmentation output rank")
    axis = class_axis % reference.ndim
    if reference.shape[axis] < 2:
        raise ValueError("Segmentation class axis must contain at least two classes")
    expected = np.argmax(reference, axis=axis)
    actual = np.argmax(candidate, axis=axis)
    pixel_agreement = float(np.mean(expected == actual))
    classes = np.union1d(expected, actual)
    ious = []
    for class_id in classes:
        expected_mask = expected == class_id
        actual_mask = actual == class_id
        union = np.logical_or(expected_mask, actual_mask).sum()
        if union:
            ious.append(float(np.logical_and(expected_mask, actual_mask).sum() / union))
    return {
        "pixel_agreement": pixel_agreement,
        "minimum_pixel_agreement": MIN_SEGMENTATION_PIXEL_AGREEMENT,
        "mean_iou_between_predictions": float(np.mean(ious)) if ious else 1.0,
    }


def compare_segmentation_raw_tensors(
    reference_path: Any,
    reference_dtype: str,
    candidate_path: Any,
    candidate_dtype: str,
    shape: list[int],
    class_axis: int,
) -> dict[str, float]:
    """Compare complete raw segmentation label maps using memory-mapped logits."""
    import numpy as np

    reference = np.memmap(reference_path, mode="r", dtype=np.dtype(reference_dtype))
    candidate = np.memmap(candidate_path, mode="r", dtype=np.dtype(candidate_dtype))
    elements = math.prod(shape)
    if reference.size != elements or candidate.size != elements:
        raise ValueError("Raw segmentation tensor size does not match its shape")
    expected = reference.reshape(shape)
    actual = candidate.reshape(shape)
    return _segmentation_agreement(expected, actual, class_axis)


def compare_raw_tensors(
    reference_path: Any,
    reference_dtype: str,
    candidate_path: Any,
    candidate_dtype: str,
    elements: int,
    atol: float,
    rtol: float,
    chunk_elements: int = 1_000_000,
) -> dict[str, Any]:
    """Compare complete raw tensors without loading both files into RAM."""
    import numpy as np

    if elements < 0 or chunk_elements <= 0:
        raise ValueError("elements must be non-negative and chunk_elements positive")
    reference_type = np.dtype(reference_dtype)
    candidate_type = np.dtype(candidate_dtype)
    reference = np.memmap(reference_path, mode="r", dtype=reference_type)
    candidate = np.memmap(candidate_path, mode="r", dtype=candidate_type)
    if reference.size != elements or candidate.size != elements:
        raise ValueError(
            "Raw tensor size mismatch: "
            f"expected {elements}, reference has {reference.size}, "
            f"candidate has {candidate.size}"
        )

    passed = True
    finite = True
    max_absolute = 0.0
    max_relative = 0.0
    dot = 0.0
    reference_norm = 0.0
    candidate_norm = 0.0
    for start in range(0, elements, chunk_elements):
        stop = min(start + chunk_elements, elements)
        expected = np.asarray(reference[start:stop], dtype=np.float64)
        actual = np.asarray(candidate[start:stop], dtype=np.float64)
        chunk_finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
        finite = finite and chunk_finite
        if not chunk_finite:
            passed = False
            continue
        difference = np.abs(expected - actual)
        if difference.size:
            max_absolute = max(max_absolute, float(difference.max()))
            denominator = np.maximum(np.abs(expected), 1e-12)
            max_relative = max(max_relative, float(np.max(difference / denominator)))
        passed = passed and bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
        dot += float(np.dot(expected, actual))
        reference_norm += float(np.dot(expected, expected))
        candidate_norm += float(np.dot(actual, actual))

    norm = math.sqrt(reference_norm) * math.sqrt(candidate_norm)
    cosine = 1.0 if norm == 0.0 and max_absolute == 0.0 else 0.0
    if norm > 0.0:
        cosine = dot / norm
    return {
        "passed": bool(passed and finite),
        "atol": atol,
        "rtol": rtol,
        "elements": elements,
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
        "cosine_similarity": cosine if math.isfinite(cosine) else None,
    }
