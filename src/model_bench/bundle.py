from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from model_bench.quality import normalize_task_metrics

EXPERIMENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
TENSOR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
MAX_METADATA_BYTES = 1_000_000
MAX_INPUT_ELEMENTS = 100_000_000
SUPPORTED_SCHEMA_VERSION = 6
SUPPORTED_DATASET_FORMATS = {"cityscapes_segmentation_npz_v1"}
RAW_DTYPE_SIZES = {
    "bool": 1,
    "float16": 2,
    "float32": 4,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(
    bundle_dir: Path, *, expected_backend: str | None = None
) -> dict[str, Any]:
    root = bundle_dir.resolve()
    bundle_path = root / "bundle.json"
    onnx_path = root / "model.onnx"
    inputs_path = root / "benchmark_inputs.npz"
    if (
        not onnx_path.is_file()
        or onnx_path.is_symlink()
        or not inputs_path.is_file()
        or inputs_path.is_symlink()
    ):
        raise ValueError(f"Invalid benchmark bundle: {bundle_dir}")
    bundle = _load_json_object(bundle_path)
    if bundle.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("Unsupported bundle schema version")
    experiment = bundle.get("experiment")
    if not isinstance(experiment, str) or EXPERIMENT_NAME.fullmatch(experiment) is None:
        raise ValueError("Invalid experiment name in bundle")
    backends = bundle.get("backends")
    if (
        not isinstance(backends, list)
        or not backends
        or len(backends) != len(set(backends))
        or any(backend not in {"cpu", "gpu"} for backend in backends)
    ):
        raise ValueError("Invalid backends in bundle")
    if expected_backend is not None and expected_backend not in backends:
        raise ValueError(
            f"Bundle does not target {expected_backend}; targets: {', '.join(backends)}"
        )

    names = _tensor_names(bundle.get("input_names"), "input_names")
    outputs = _tensor_names(bundle.get("output_names"), "output_names")
    shapes = bundle.get("input_shapes")
    if not isinstance(shapes, list) or len(shapes) != len(names):
        raise ValueError("input_names and input_shapes must have equal lengths")
    total_elements = 0
    for shape in shapes:
        if not isinstance(shape, list) or not shape:
            raise ValueError("Every input shape must be a non-empty list")
        if any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("Input dimensions must be positive integers")
        total_elements += math.prod(shape)
    if total_elements > MAX_INPUT_ELEMENTS:
        raise ValueError("Bundle inputs exceed the safety limit")
    if not outputs:
        raise ValueError("At least one output is required")

    training_metrics = bundle.get("training_metrics")
    if not isinstance(training_metrics, dict) or training_metrics.get("task") not in {
        "segmentation",
        "depth",
    }:
        raise ValueError("Bundle training metrics must declare a supported task")
    _validate_conversion_quality(bundle.get("conversion_quality"), training_metrics)

    onnx_files = bundle.get("onnx_files")
    if not isinstance(onnx_files, dict) or "model.onnx" not in onnx_files:
        raise ValueError("Bundle must list model.onnx and its external data files")
    for file_name, expected_hash in onnx_files.items():
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError("Invalid ONNX artifact name")
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ):
            raise ValueError("Invalid ONNX artifact checksum")
        artifact = root / file_name
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"Missing ONNX artifact: {file_name}")
        if sha256_file(artifact) != expected_hash:
            raise ValueError(f"ONNX artifact checksum mismatch: {file_name}")
    if bundle.get("onnx_sha256") != onnx_files["model.onnx"]:
        raise ValueError("Primary ONNX checksum does not match onnx_files")
    inputs_hash = bundle.get("benchmark_inputs_sha256")
    if not isinstance(inputs_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", inputs_hash
    ):
        raise ValueError("Invalid benchmark input checksum")
    if sha256_file(inputs_path) != inputs_hash:
        raise ValueError("Benchmark input checksum mismatch")
    return bundle


def _validate_conversion_quality(value: Any, training_metrics: dict[str, Any]) -> None:
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise ValueError("Bundle must contain passing post-conversion quality")
    task = training_metrics["task"]
    if value.get("task") != task:
        raise ValueError("Conversion quality task does not match training metrics")
    source = normalize_task_metrics(
        value.get("source_metrics"), task, "source quality metrics"
    )
    converted = normalize_task_metrics(
        value.get("post_conversion_metrics"), task, "post-conversion quality metrics"
    )
    if set(source) != set(converted):
        raise ValueError("Source and post-conversion quality metrics must match")
    tolerance = value.get("tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or not 0 <= tolerance <= 1e-3
    ):
        raise ValueError("Invalid post-conversion quality tolerance")
    deltas = value.get("absolute_deltas")
    if not isinstance(deltas, dict) or set(deltas) != set(source):
        raise ValueError("Invalid post-conversion quality deltas")
    for name in source:
        expected = abs(converted[name] - source[name])
        if (
            isinstance(deltas[name], bool)
            or not isinstance(deltas[name], (int, float))
            or not math.isclose(
                float(deltas[name]), expected, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError("Post-conversion quality delta does not match metrics")
        if expected > tolerance:
            raise ValueError("Post-conversion quality exceeds tolerance")
    dataset = value.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Post-conversion quality dataset provenance is missing")
    if (
        not isinstance(dataset.get("uri"), str)
        or not isinstance(dataset.get("include"), list)
        or dataset.get("format") not in SUPPORTED_DATASET_FORMATS
        or type(dataset.get("object_count")) is not int
        or dataset["object_count"] <= 0
        or type(dataset.get("total_bytes")) is not int
        or dataset["total_bytes"] <= 0
        or not isinstance(dataset.get("listing_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset["listing_sha256"]) is None
    ):
        raise ValueError("Invalid post-conversion dataset provenance")


def load_parity_manifest(parity_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_json_object(parity_dir / "manifest.json")
    if manifest.get("reference_runtime") != "onnxruntime-cpu":
        raise ValueError("Unsupported TensorRT parity reference runtime")
    inputs = _tensor_mapping(manifest.get("inputs"), "inputs")
    outputs = _tensor_mapping(manifest.get("outputs"), "outputs")
    if list(inputs) != bundle["input_names"] or list(outputs) != bundle["output_names"]:
        raise ValueError("Parity tensor names do not match bundle metadata")
    for metadata in (*inputs.values(), *outputs.values()):
        _validate_tensor_file(parity_dir, metadata)
    additional = manifest.get("additional_samples", [])
    if not isinstance(additional, list):
        raise ValueError("additional_samples must be a list")
    seen_ids = {0}
    for sample in additional:
        if not isinstance(sample, dict) or type(sample.get("id")) is not int:
            raise ValueError("Invalid additional parity sample")
        if sample["id"] in seen_ids:
            raise ValueError("Parity sample ids must be unique")
        seen_ids.add(sample["id"])
        sample_inputs = _tensor_mapping(sample.get("inputs"), "inputs")
        sample_outputs = _tensor_mapping(sample.get("outputs"), "outputs")
        if (
            list(sample_inputs) != bundle["input_names"]
            or list(sample_outputs) != bundle["output_names"]
        ):
            raise ValueError("Additional parity tensor names do not match bundle")
        for metadata in (*sample_inputs.values(), *sample_outputs.values()):
            _validate_tensor_file(parity_dir, metadata)
    tolerances = manifest.get("tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError("Parity tolerances are missing")
    for precision, limits in {"fp32": (1e-4, 1e-4), "fp16": (1e-3, 1e-2)}.items():
        configured = tolerances.get(precision)
        if not isinstance(configured, dict):
            raise ValueError(f"Parity tolerance for {precision} is missing")
        for name, maximum in zip(("atol", "rtol"), limits, strict=True):
            value = configured.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {precision} {name}")
            if value < 0 or value > maximum:
                raise ValueError(f"{precision} {name} exceeds policy maximum")
    semantic = manifest.get("semantic_validation")
    if not isinstance(semantic, dict) or semantic.get("task") not in {
        "segmentation",
        "depth",
    }:
        raise ValueError("Semantic validation metadata is missing")
    if semantic["task"] != bundle["training_metrics"]["task"]:
        raise ValueError("Semantic validation task does not match training metrics")
    if semantic["task"] == "segmentation":
        if semantic.get("output_name") not in bundle["output_names"]:
            raise ValueError("Semantic output does not match bundle outputs")
        if type(semantic.get("class_axis")) is not int:
            raise ValueError("Segmentation class_axis must be an integer")
        minimum_agreement = semantic.get("minimum_pixel_agreement")
        if (
            isinstance(minimum_agreement, bool)
            or not isinstance(minimum_agreement, (int, float))
            or not math.isclose(float(minimum_agreement), 0.9999)
        ):
            raise ValueError("Invalid segmentation label-map agreement policy")
    return manifest


def safe_result_directory(results_root: Path, experiment: str) -> Path:
    if EXPERIMENT_NAME.fullmatch(experiment) is None:
        raise ValueError("Invalid experiment name")
    root = results_root.resolve()
    output = (root / experiment).resolve()
    if output.parent != root:
        raise ValueError("Result path escapes results directory")
    return output


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing metadata: {path}")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError(f"Metadata is too large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return value


def _tensor_names(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(
        not isinstance(name, str) or TENSOR_NAME.fullmatch(name) is None
        for name in value
    ):
        raise ValueError(f"{field} contains an invalid tensor name")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must contain unique names")
    return value


def _tensor_mapping(value: Any, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Parity {field} must be a non-empty object")
    _tensor_names(list(value), field)
    if any(not isinstance(metadata, dict) for metadata in value.values()):
        raise ValueError(f"Parity {field} metadata must be objects")
    return value


def _validate_tensor_file(root: Path, metadata: dict[str, Any]) -> None:
    file_name = metadata.get("file")
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise ValueError("Parity manifest contains an invalid file name")
    path = root / file_name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Invalid parity tensor file: {path}")
    elements = metadata.get("elements")
    shape = metadata.get("shape")
    if type(elements) is not int or elements < 0:
        raise ValueError("Invalid parity tensor element count")
    if not isinstance(shape, list) or any(
        type(value) is not int or value < 0 for value in shape
    ):
        raise ValueError("Invalid parity tensor shape")
    if math.prod(shape) != elements:
        raise ValueError("Parity tensor shape does not match element count")
    dtype = metadata.get("dtype")
    if dtype not in RAW_DTYPE_SIZES:
        raise ValueError(f"Unsupported parity tensor dtype: {dtype!r}")
    if path.stat().st_size != elements * RAW_DTYPE_SIZES[dtype]:
        raise ValueError(
            f"Parity tensor byte size does not match metadata: {file_name}"
        )
    expected_hash = metadata.get("sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise ValueError("Invalid parity tensor checksum")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Parity tensor checksum mismatch: {file_name}")
