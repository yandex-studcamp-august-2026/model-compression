from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from model_bench.adapter import (
    evaluate_quality,
    find_model_class,
    flatten_outputs,
    input_names,
    input_shapes,
    instantiate_model,
    load_forward_module,
    make_inputs,
    output_names,
)
from model_bench.bundle import MAX_INPUT_ELEMENTS, sha256_file
from model_bench.discovery import Candidate
from model_bench.quality import load_metrics
from model_bench.storage import resolve_weights
from model_bench.validation import compare_outputs


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or None


def _write_raw_tensor(path: Path, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value)
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        array = array.astype(array.dtype.newbyteorder("<"))
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {
        "file": path.name,
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "elements": int(array.size),
        "sha256": sha256_file(path),
    }


def export_candidate(
    candidate: Candidate,
    bundles_root: Path,
    weights_path: Path | None = None,
    dataset_root: Path | None = None,
) -> Path:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("Install project with the [export] extra") from exc

    bundle_dir = bundles_root / candidate.name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    owns_weights = weights_path is None
    weights = weights_path or resolve_weights(
        candidate.weights_url, bundle_dir / "inputs"
    )
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    expected_weights_sha256 = candidate.weights_sha256.read_text(
        encoding="ascii"
    ).strip()
    actual_weights_sha256 = sha256_file(weights)
    if actual_weights_sha256 != expected_weights_sha256:
        raise ValueError(
            "Checkpoint SHA-256 does not match weights.sha256: "
            f"expected {expected_weights_sha256}, got {actual_weights_sha256}"
        )
    if dataset_root is None or not dataset_root.is_dir():
        raise ValueError("A downloaded validation dataset directory is required")
    dataset_provenance = _dataset_provenance(dataset_root)

    module = load_forward_module(candidate.forward)
    model = instantiate_model(find_model_class(module), weights).eval().cpu()
    names = input_names(module)
    outputs = output_names(module)
    shapes = input_shapes(module)
    if len(names) != len(shapes):
        raise ValueError("INPUT_NAMES and INPUT_SHAPES must have equal lengths")
    opset = int(getattr(module, "ONNX_OPSET", 18))
    parity_samples = int(getattr(module, "PARITY_SAMPLES", 3))
    if not 3 <= parity_samples <= 20:
        raise ValueError("PARITY_SAMPLES must be between 3 and 20")
    atol = _bounded_tolerance(module, "PARITY_ATOL", 1e-4)
    rtol = _bounded_tolerance(module, "PARITY_RTOL", 1e-4)
    training_metrics = load_metrics(candidate.metrics, candidate.name)
    task = training_metrics["task"]
    semantic_output_name = None
    class_axis = 1
    if task == "segmentation":
        semantic_output_name = str(getattr(module, "SEMANTIC_OUTPUT_NAME", outputs[0]))
        if semantic_output_name not in outputs:
            raise ValueError("SEMANTIC_OUTPUT_NAME must name one declared output")
        class_axis = int(getattr(module, "CLASS_AXIS", 1))
    if sum(math.prod(shape) for shape in shapes) > MAX_INPUT_ELEMENTS:
        raise ValueError("Model inputs exceed the safety limit")

    example = make_inputs(module, names, shapes, seed=0)
    onnx_path = bundle_dir / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            onnx_path,
            input_names=list(names),
            output_names=list(outputs),
            opset_version=opset,
            dynamo=True,
        )
    onnx.checker.check_model(str(onnx_path), full_check=True)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample_reports = []
    parity_dir = bundle_dir / "tensorrt_parity"
    parity_entries = []
    for seed in range(parity_samples):
        values = make_inputs(module, names, shapes, seed=seed)
        feed = {
            name: value.detach().cpu().numpy()
            for name, value in zip(names, values, strict=True)
        }
        with torch.inference_mode():
            expected = flatten_outputs(model(*values), outputs)
        actual = session.run(list(outputs), feed)
        parity_entries.append(
            {
                "id": seed,
                "inputs": {
                    name: _write_raw_tensor(
                        parity_dir / f"sample-{seed}-input-{index}.raw", value
                    )
                    for index, (name, value) in enumerate(feed.items())
                },
                "outputs": {
                    name: _write_raw_tensor(
                        parity_dir / f"sample-{seed}-output-{index}.raw", value
                    )
                    for index, (name, value) in enumerate(
                        zip(outputs, actual, strict=True)
                    )
                },
            }
        )
        report = compare_outputs(
            expected,
            actual,
            outputs,
            atol=atol,
            rtol=rtol,
            task=task,
            semantic_output_name=semantic_output_name,
            class_axis=class_axis,
        )
        report["seed"] = seed
        sample_reports.append(report)
    conversion_report: dict[str, Any] = {
        "passed": all(report["passed"] for report in sample_reports),
        "samples": sample_reports,
    }
    conversion_path = bundle_dir / "conversion_report.json"
    conversion_path.write_text(
        json.dumps(conversion_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not conversion_report["passed"]:
        raise RuntimeError(f"PyTorch and ONNX outputs differ; see {conversion_path}")

    def normalize_evaluation_inputs(values: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(values) != len(names):
            raise ValueError(f"Quality evaluator expected {len(names)} inputs")
        normalized = []
        for name, shape, value in zip(names, shapes, values, strict=True):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Quality input {name!r} must be a torch.Tensor")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Quality input {name!r} has shape {tuple(value.shape)}, "
                    f"expected {shape}"
                )
            normalized.append(value.detach().cpu().contiguous())
        return tuple(normalized)

    def source_predict(*values: Any) -> tuple[Any, ...]:
        tensors = normalize_evaluation_inputs(values)
        with torch.inference_mode():
            predictions = flatten_outputs(model(*tensors), outputs)
        return tuple(value.detach().cpu().numpy() for value in predictions)

    def onnx_predict(*values: Any) -> tuple[Any, ...]:
        tensors = normalize_evaluation_inputs(values)
        feed = {name: value.numpy() for name, value in zip(names, tensors, strict=True)}
        return tuple(session.run(list(outputs), feed))

    source_quality = evaluate_quality(module, source_predict, dataset_root, task)
    converted_quality = evaluate_quality(module, onnx_predict, dataset_root, task)
    quality_tolerance = _bounded_tolerance(module, "QUALITY_ATOL", 1e-3, default=1e-4)
    quality_deltas = {
        name: abs(converted_quality[name] - source_quality[name])
        for name in source_quality
    }
    quality_validation = {
        "passed": all(delta <= quality_tolerance for delta in quality_deltas.values()),
        "task": task,
        "source_metrics": source_quality,
        "post_conversion_metrics": converted_quality,
        "absolute_deltas": quality_deltas,
        "tolerance": quality_tolerance,
        "dataset": dataset_provenance,
    }
    conversion_report["task_quality_validation"] = quality_validation
    conversion_path.write_text(
        json.dumps(conversion_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not quality_validation["passed"]:
        raise RuntimeError(
            f"Post-conversion task quality differs from PyTorch; see {conversion_path}"
        )

    if not parity_entries:
        raise RuntimeError("No TensorRT parity sample was generated")
    parity_manifest = {
        "reference_runtime": "onnxruntime-cpu",
        "inputs": parity_entries[0]["inputs"],
        "outputs": parity_entries[0]["outputs"],
        "additional_samples": parity_entries[1:],
        "tolerances": {
            "fp32": {
                "atol": _bounded_tolerance(
                    module, "TENSORRT_FP32_ATOL", 1e-4, default=atol
                ),
                "rtol": _bounded_tolerance(
                    module, "TENSORRT_FP32_RTOL", 1e-4, default=rtol
                ),
            },
            "fp16": {
                "atol": _bounded_tolerance(module, "TENSORRT_FP16_ATOL", 1e-3),
                "rtol": _bounded_tolerance(module, "TENSORRT_FP16_RTOL", 1e-2),
            },
        },
        "semantic_validation": {
            "task": task,
            "output_name": semantic_output_name,
            "class_axis": class_axis if task == "segmentation" else None,
            "require_exact_label_map": task == "segmentation",
        },
    }
    (parity_dir / "manifest.json").write_text(
        json.dumps(parity_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    np.savez(
        bundle_dir / "benchmark_inputs.npz",
        **{
            name: value.detach().cpu().numpy()
            for name, value in zip(names, example, strict=True)
        },
    )
    onnx_files = {
        path.name: sha256_file(path)
        for path in sorted(bundle_dir.glob("model.onnx*"))
        if path.is_file() and not path.is_symlink()
    }
    if "model.onnx" not in onnx_files:
        raise RuntimeError("ONNX export did not create model.onnx")
    benchmark_inputs_path = bundle_dir / "benchmark_inputs.npz"
    bundle = {
        "schema_version": 4,
        "experiment": candidate.name,
        "backend": candidate.backend.value,
        "git_sha": _git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
        "input_names": list(names),
        "input_shapes": [list(shape) for shape in shapes],
        "output_names": list(outputs),
        "onnx_opset": opset,
        "export_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "onnxscript": version("onnxscript"),
            "transformers": version("transformers"),
        },
        "weights_sha256": actual_weights_sha256,
        "onnx_sha256": onnx_files["model.onnx"],
        "onnx_files": onnx_files,
        "benchmark_inputs_sha256": sha256_file(benchmark_inputs_path),
        "conversion_validation": conversion_report,
        "training_metrics": training_metrics,
        "conversion_quality": quality_validation,
    }
    bundle_path = bundle_dir / "bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if owns_weights:
        weights.unlink(missing_ok=True)
    return bundle_path


def _dataset_provenance(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "_dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Downloaded dataset is missing _dataset_manifest.json")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    return value


def _bounded_tolerance(
    module: Any, name: str, maximum: float, *, default: float | None = None
) -> float:
    value = float(getattr(module, name, maximum if default is None else default))
    if not math.isfinite(value) or value < 0 or value > maximum:
        raise ValueError(f"{name} must be finite and between 0 and {maximum}")
    return value
