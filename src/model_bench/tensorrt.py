from __future__ import annotations

import json
import math
import platform
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_bench.bundle import (
    load_bundle,
    load_parity_manifest,
    safe_result_directory,
)
from model_bench.reporting import (
    export_latency_artifacts,
    quality_result,
    summarize_latency,
)
from model_bench.validation import (
    MIN_SEGMENTATION_PIXEL_AGREEMENT,
    compare_raw_tensors,
    compare_segmentation_raw_tensors,
)

GPU_QUERY_FIELDS = ",".join(
    (
        "name",
        "uuid",
        "driver_version",
        "memory.total",
        "pstate",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
        "power.limit",
    )
)

TRT_DTYPES = {
    "BOOL": "bool",
    "FP16": "float16",
    "FP32": "float32",
    "INT8": "int8",
    "INT32": "int32",
    "INT64": "int64",
    "UINT8": "uint8",
}

CITYSCAPES_FORMAT = "cityscapes_segmentation_npz_v1"
CITYSCAPES_CLASSES = 19
CITYSCAPES_IMAGE_SHAPE = (512, 1024, 3)
CITYSCAPES_TARGET_SHAPE = (512, 1024)
CITYSCAPES_IGNORE_LABEL = 255
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)


def parse_trtexec_version(output: str) -> str:
    semantic = re.search(
        r"TensorRT(?: version)?:?\s+v?([0-9]+(?:\.[0-9]+){2,3})",
        output,
        re.IGNORECASE,
    )
    if semantic is not None:
        return semantic.group(1)

    encoded = re.search(r"TensorRT\s+v([0-9]{6})\b", output, re.IGNORECASE)
    if encoded is None:
        raise ValueError("Cannot parse TensorRT version")
    digits = encoded.group(1)
    return f"{int(digits[:-4])}.{int(digits[-4:-2])}.{int(digits[-2:])}"


def ensure_trtexec() -> str:
    binary = shutil.which("trtexec")
    if binary is None:
        raise RuntimeError("trtexec is not available")
    return binary


def run_trtexec(
    arguments: list[str], cwd: Path | None = None, log_path: Path | None = None
) -> str:
    command = [ensure_trtexec(), *arguments]
    result = subprocess.run(
        command, text=True, capture_output=True, check=False, cwd=cwd
    )
    output = result.stdout + result.stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec failed with exit code {result.returncode}\n"
            f"command: {' '.join(command)}\n{output}"
        )
    return output


def trtexec_runtime_version() -> str:
    """Read the TensorRT version from the supported trtexec help output."""
    return parse_trtexec_version(run_trtexec(["--help"]))


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    shapes: str,
    precision: str,
    log_path: Path | None = None,
) -> str:
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    arguments = [
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--shapes={shapes}",
        "--builderOptimizationLevel=5",
        "--noTF32",
        "--skipInference",
    ]
    if precision == "fp16":
        arguments.append("--fp16")
    return run_trtexec(arguments, log_path=log_path)


def benchmark_engine(
    engine_path: Path,
    shapes: str,
    warmup_ms: int,
    iterations: int,
    times_path: Path,
    input_spec: str | None = None,
    inf_streams: int = 1,
) -> tuple[float, str]:
    if inf_streams <= 0:
        raise ValueError("inf_streams must be positive")
    arguments = [
        f"--loadEngine={engine_path}",
        f"--shapes={shapes}",
        f"--infStreams={inf_streams}",
        f"--warmUp={warmup_ms}",
        "--duration=0",
        f"--iterations={iterations}",
        "--noDataTransfers",
        "--useCudaGraph",
        "--useSpinWait",
        f"--exportTimes={times_path}",
    ]
    if input_spec is not None:
        arguments.append(f"--loadInputs={input_spec}")
    output = run_trtexec(arguments)
    return parse_throughput(output), output


def parse_throughput(output: str) -> float:
    match = re.search(
        r"\bThroughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps\b", output, re.IGNORECASE
    )
    if match is None:
        raise RuntimeError("TensorRT throughput was not found in trtexec output")
    return float(match.group(1))


def load_gpu_compute_samples(times_path: Path) -> list[float]:
    payload = json.loads(times_path.read_text(encoding="utf-8"))
    records = (
        payload.get("times", payload.get("iterations", []))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(records, list):
        raise RuntimeError(f"Unsupported trtexec timing format: {times_path}")
    samples = [
        float(record["computeMs"])
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("computeMs"), (int, float))
        and math.isfinite(record["computeMs"])
        and record["computeMs"] >= 0
    ]
    if not samples:
        raise RuntimeError(f"No GPU computeMs samples found in {times_path}")
    return samples


def _safe_tensor_name(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-," else "_"
        for character in name
    )


def _candidate_raw_output(
    directory: Path, tensor_name: str
) -> tuple[Path, str, list[int]]:
    matches = sorted(directory.glob(f"{_safe_tensor_name(tensor_name)}.output.*.raw"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one raw TensorRT output for {tensor_name!r}, got {matches}"
        )
    prefix = f"{_safe_tensor_name(tensor_name)}.output."
    payload = matches[0].name.removeprefix(prefix).removesuffix(".raw")
    dimensions, dtype_token = payload.rsplit(".", maxsplit=1)
    shape = [] if not dimensions else [int(value) for value in dimensions.split(".")]
    try:
        return matches[0], TRT_DTYPES[dtype_token], shape
    except KeyError as exc:
        raise RuntimeError(f"Unsupported TensorRT output dtype: {dtype_token}") from exc


def _parity_file(parity_dir: Path, file_name: str) -> Path:
    if Path(file_name).name != file_name:
        raise ValueError(f"Parity manifest contains an invalid file name: {file_name}")
    path = parity_dir / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Parity tensor is missing: {path}")
    return path


def _input_spec(parity_dir: Path, inputs: dict[str, Any]) -> str:
    return ",".join(
        f"{name}:{_parity_file(parity_dir, metadata['file']).resolve()}"
        for name, metadata in inputs.items()
    )


def validate_engine(
    engine_path: Path,
    shapes: str,
    parity_dir: Path,
    work_dir: Path,
    precision: str,
) -> tuple[dict[str, Any], str]:
    manifest = json.loads((parity_dir / "manifest.json").read_text(encoding="utf-8"))
    tolerance = manifest["tolerances"][precision]
    semantic = manifest["semantic_validation"]
    samples = [
        {"id": 0, "inputs": manifest["inputs"], "outputs": manifest["outputs"]},
        *manifest.get("additional_samples", []),
    ]
    sample_results = []
    logs = []
    passed = True
    for sample in samples:
        sample_work_dir = work_dir / f"sample-{sample['id']}"
        sample_work_dir.mkdir(parents=True, exist_ok=True)
        input_spec = _input_spec(parity_dir, sample["inputs"])
        log = run_trtexec(
            [
                f"--loadEngine={engine_path.resolve()}",
                f"--shapes={shapes}",
                f"--loadInputs={input_spec}",
                "--iterations=1",
                "--warmUp=0",
                "--duration=0",
                "--dumpRawBindingsToFile",
            ],
            cwd=sample_work_dir,
        )
        logs.append(f"===== sample {sample['id']} =====\n{log}")
        outputs: dict[str, Any] = {}
        sample_passed = True
        for name, metadata in sample["outputs"].items():
            candidate_path, candidate_dtype, candidate_shape = _candidate_raw_output(
                sample_work_dir, name
            )
            result = compare_raw_tensors(
                _parity_file(parity_dir, metadata["file"]),
                metadata["dtype"],
                candidate_path,
                candidate_dtype,
                int(metadata["elements"]),
                atol=float(tolerance["atol"]),
                rtol=float(tolerance["rtol"]),
            )
            result["shape"] = metadata["shape"]
            result["candidate_shape"] = candidate_shape
            result["reference_dtype"] = metadata["dtype"]
            result["candidate_dtype"] = candidate_dtype
            result["passed"] = bool(
                result["passed"] and candidate_shape == metadata["shape"]
            )
            if (
                semantic["task"] == "segmentation"
                and name == semantic["output_name"]
                and candidate_shape == metadata["shape"]
            ):
                agreement = compare_segmentation_raw_tensors(
                    _parity_file(parity_dir, metadata["file"]),
                    metadata["dtype"],
                    candidate_path,
                    candidate_dtype,
                    metadata["shape"],
                    semantic["class_axis"],
                )
                result["segmentation_agreement"] = agreement
                result["passed"] = bool(
                    result["passed"]
                    and agreement["pixel_agreement"] >= MIN_SEGMENTATION_PIXEL_AGREEMENT
                )
            outputs[name] = result
            sample_passed = sample_passed and result["passed"]
        sample_results.append(
            {"id": sample["id"], "passed": sample_passed, "outputs": outputs}
        )
        passed = passed and sample_passed
    return {
        "passed": passed,
        "reference_runtime": manifest["reference_runtime"],
        "atol": float(tolerance["atol"]),
        "rtol": float(tolerance["rtol"]),
        "samples": sample_results,
    }, "\n".join(logs)


def evaluate_engine_quality(
    engine_path: Path,
    shapes: str,
    dataset_root: Path,
    work_dir: Path,
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Measure TensorRT task quality on the same immutable validation dataset."""
    import numpy as np

    expected_dataset = bundle["conversion_quality"]["dataset"]
    manifest_path = dataset_root / "_dataset_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("TensorRT validation dataset manifest is missing")
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance_fields = {
        "uri",
        "include",
        "format",
        "object_count",
        "total_bytes",
        "listing_sha256",
    }
    if any(
        dataset.get(name) != expected_dataset.get(name) for name in provenance_fields
    ):
        raise ValueError("TensorRT validation dataset provenance does not match bundle")
    if dataset.get("format") != CITYSCAPES_FORMAT:
        raise ValueError(
            f"Unsupported TensorRT quality dataset: {dataset.get('format')!r}"
        )
    if bundle["training_metrics"]["task"] != "segmentation":
        raise ValueError("Cityscapes TensorRT quality requires a segmentation model")
    if len(bundle["input_names"]) != 1 or len(bundle["output_names"]) != 1:
        raise ValueError(
            "Cityscapes TensorRT quality requires one input and one output"
        )
    required_input_shape = [1, 3, *CITYSCAPES_IMAGE_SHAPE[:2]]
    if bundle["input_shapes"] != [required_input_shape]:
        raise ValueError(
            "Cityscapes TensorRT quality requires input shape 1x3x512x1024"
        )

    sample_paths = sorted(dataset_root.glob("samples/*.npz"))
    if not sample_paths or len(sample_paths) > 512:
        raise ValueError("Cityscapes validation subset must contain 1..512 samples")
    if any(path.is_symlink() or not path.is_file() for path in sample_paths):
        raise ValueError("Cityscapes validation samples must be regular files")

    confusion = np.zeros((CITYSCAPES_CLASSES, CITYSCAPES_CLASSES), dtype=np.int64)
    mean = np.asarray(CITYSCAPES_MEAN, dtype=np.float32)
    std = np.asarray(CITYSCAPES_STD, dtype=np.float32)
    input_name = bundle["input_names"][0]
    output_name = bundle["output_names"][0]
    logs = []
    for index, sample_path in enumerate(sample_paths):
        sample_dir = work_dir / f"sample-{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            with np.load(sample_path, allow_pickle=False) as sample:
                if set(sample.files) != {"image", "label"}:
                    raise ValueError(f"Invalid Cityscapes sample keys: {sample_path}")
                image = np.asarray(sample["image"], dtype=np.uint8)
                target = np.asarray(sample["label"], dtype=np.uint8)
            if image.shape != CITYSCAPES_IMAGE_SHAPE:
                raise ValueError(f"Invalid Cityscapes image shape: {sample_path}")
            if target.shape != CITYSCAPES_TARGET_SHAPE:
                raise ValueError(f"Invalid Cityscapes target shape: {sample_path}")
            valid_target = target[target != CITYSCAPES_IGNORE_LABEL]
            if valid_target.size and int(valid_target.max()) >= CITYSCAPES_CLASSES:
                raise ValueError(f"Invalid Cityscapes target class: {sample_path}")

            normalized = image.astype(np.float32) / 255.0
            normalized = (normalized - mean) / std
            model_input = np.ascontiguousarray(
                normalized.transpose(2, 0, 1)[np.newaxis]
            )
            input_path = sample_dir / "input.raw"
            model_input.tofile(input_path)
            log = run_trtexec(
                [
                    f"--loadEngine={engine_path.resolve()}",
                    f"--shapes={shapes}",
                    f"--loadInputs={input_name}:{input_path.resolve()}",
                    "--iterations=1",
                    "--warmUp=0",
                    "--duration=0",
                    "--dumpRawBindingsToFile",
                ],
                cwd=sample_dir,
            )
            logs.append(f"===== {sample_path.name} =====\n{log}")
            output_path, output_dtype, output_shape = _candidate_raw_output(
                sample_dir, output_name
            )
            expected_shape = [1, CITYSCAPES_CLASSES, *CITYSCAPES_TARGET_SHAPE]
            if output_shape != expected_shape:
                raise ValueError(
                    f"Invalid TensorRT Cityscapes output shape: {output_shape}"
                )
            if output_dtype not in {"float16", "float32"}:
                raise ValueError(
                    f"Invalid TensorRT Cityscapes output dtype: {output_dtype}"
                )
            logits = np.memmap(
                output_path,
                mode="r",
                dtype=np.dtype(output_dtype),
                shape=tuple(expected_shape),
            )
            prediction = np.argmax(logits, axis=1)[0]
            valid = target != CITYSCAPES_IGNORE_LABEL
            encoded = CITYSCAPES_CLASSES * target[valid].astype(np.int64) + prediction[
                valid
            ].astype(np.int64)
            confusion += np.bincount(
                encoded,
                minlength=CITYSCAPES_CLASSES * CITYSCAPES_CLASSES,
            ).reshape(CITYSCAPES_CLASSES, CITYSCAPES_CLASSES)
            del logits
        finally:
            shutil.rmtree(sample_dir, ignore_errors=True)

    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    present = union > 0
    if not np.any(present):
        raise ValueError("Cityscapes validation subset has no labelled classes")
    measured = {"mIoU": float(np.mean(intersection[present] / union[present]))}
    source = bundle["conversion_quality"]["source_metrics"]
    tolerance = float(bundle["conversion_quality"]["tolerance"])
    deltas = {name: abs(measured[name] - float(source[name])) for name in measured}
    return {
        "passed": all(delta <= tolerance for delta in deltas.values()),
        "task": "segmentation",
        "source_metrics": source,
        "post_conversion_metrics": measured,
        "absolute_deltas": deltas,
        "tolerance": tolerance,
        "dataset": dataset,
        "sample_count": len(sample_paths),
        "runtime": "tensorrt",
    }, "\n".join(logs)


def _performance_result(
    engine_path: Path,
    shapes: str,
    output_dir: Path,
    precision: str,
    warmup_ms: int,
    iterations: int,
    input_spec: str,
    throughput_streams: int,
) -> dict[str, Any]:
    suffix = "" if precision == "fp32" else "_fp16"
    times_path = output_dir / f"trtexec{suffix}-times.json"
    _, benchmark_log = benchmark_engine(
        engine_path, shapes, warmup_ms, iterations, times_path, input_spec
    )
    (output_dir / f"trtexec{suffix}.log").write_text(benchmark_log, encoding="utf-8")
    samples = load_gpu_compute_samples(times_path)
    if len(samples) != iterations:
        raise RuntimeError(f"Expected {iterations} iterations, got {len(samples)}")
    latency_csv = f"latency{suffix}_ms.csv"
    histogram = f"latency{suffix}_histogram.svg"
    export_latency_artifacts(
        samples,
        output_dir / latency_csv,
        output_dir / histogram,
        f"TensorRT {precision.upper()} GPU compute latency",
    )
    throughput_sweep = []
    throughput_artifacts = {}
    for stream_count in _sweep_values(throughput_streams):
        run_name = f"trtexec{suffix}-throughput-streams-{stream_count}"
        throughput_times = output_dir / f"{run_name}-times.json"
        throughput, throughput_log = benchmark_engine(
            engine_path,
            shapes,
            warmup_ms,
            iterations,
            throughput_times,
            input_spec,
            inf_streams=stream_count,
        )
        throughput_log_name = f"{run_name}.log"
        (output_dir / throughput_log_name).write_text(throughput_log, encoding="utf-8")
        throughput_sweep.append({"streams": stream_count, "throughput_qps": throughput})
        throughput_artifacts[str(stream_count)] = {
            "log": throughput_log_name,
            "timings": throughput_times.name,
        }
    best_throughput = max(throughput_sweep, key=lambda result: result["throughput_qps"])
    return {
        "latency_kind": "gpu_compute_time",
        "latency_unit": "ms",
        "latency": summarize_latency(samples),
        "throughput_kind": "best_multi_stream_sweep",
        "throughput_qps": best_throughput["throughput_qps"],
        "throughput_streams": best_throughput["streams"],
        "throughput_sweep": throughput_sweep,
        "artifacts": {
            "latency_samples": latency_csv,
            "latency_histogram": histogram,
            "benchmark_log": f"trtexec{suffix}.log",
            "timings": times_path.name,
            "throughput_runs": throughput_artifacts,
        },
    }


def _sweep_values(maximum: int) -> list[int]:
    values = []
    current = 1
    while current <= maximum:
        values.append(current)
        current *= 2
    if values[-1] != maximum:
        values.append(maximum)
    return values


def benchmark_gpu_bundle(
    bundle_dir: Path,
    results_root: Path,
    dataset_root: Path,
    warmup_ms: int,
    iterations: int,
    throughput_streams: int = 4,
) -> Path:
    if warmup_ms < 0 or iterations <= 0 or throughput_streams <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    onnx_path = bundle_dir / "model.onnx"
    parity_dir = bundle_dir / "tensorrt_parity"
    bundle = load_bundle(bundle_dir, expected_backend="gpu")
    parity_manifest = load_parity_manifest(parity_dir, bundle)
    benchmark_input_spec = _input_spec(parity_dir, parity_manifest["inputs"])
    shape_items = [
        f"{name}:" + "x".join(str(value) for value in shape)
        for name, shape in zip(
            bundle["input_names"], bundle["input_shapes"], strict=True
        )
    ]
    shapes = ",".join(shape_items)
    output_dir = safe_result_directory(results_root, str(bundle["experiment"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 2,
        "experiment": bundle["experiment"],
        "git_sha": bundle.get("git_sha"),
        "created_at": datetime.now(UTC).isoformat(),
        "backend": "tensorrt",
        "status": "initializing",
        "conversion_status": "unavailable",
        "precision_policy": "fp32-no-tf32",
        "platform": platform.platform(),
        "gpu": _command_value(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY_FIELDS}", "--format=csv,noheader"]
        ),
        "runtime_version": trtexec_runtime_version(),
        "input_shapes": bundle["input_shapes"],
        "warmup_ms": warmup_ms,
        "measured_iterations": iterations,
        "max_throughput_streams": throughput_streams,
        "conversion_validation": bundle.get("conversion_validation"),
        "training_metrics": bundle.get("training_metrics"),
        "quality": quality_result(bundle.get("conversion_quality")),
        "export_environment": bundle.get("export_environment"),
        "source_artifacts": {
            "weights_sha256": bundle.get("weights_sha256"),
            "onnx_sha256": bundle.get("onnx_sha256"),
            "onnx_files": bundle.get("onnx_files"),
            "benchmark_inputs_sha256": bundle.get("benchmark_inputs_sha256"),
        },
    }
    report_path = output_dir / "report.json"
    engines = {
        "fp32": output_dir / "model.fp32.plan",
        "fp16": output_dir / "model.fp16.plan",
    }
    try:
        fp32_build_path = output_dir / "trtexec-build.log"
        build_engine(
            onnx_path, engines["fp32"], shapes, "fp32", log_path=fp32_build_path
        )
        fp32_validation, fp32_validation_log = validate_engine(
            engines["fp32"], shapes, parity_dir, output_dir / ".parity-fp32", "fp32"
        )
        (output_dir / "trtexec-validation.log").write_text(
            fp32_validation_log, encoding="utf-8"
        )
        report["tensorrt_validation"] = fp32_validation
        if not fp32_validation["passed"]:
            report["conversion_status"] = "rejected"
            report["status"] = "rejected"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raise RuntimeError(f"TensorRT FP32 parity failed; see {report_path}")
        fp32_quality, fp32_quality_log = evaluate_engine_quality(
            engines["fp32"],
            shapes,
            dataset_root,
            output_dir / ".quality-fp32",
            bundle,
        )
        (output_dir / "trtexec-quality.log").write_text(
            fp32_quality_log, encoding="utf-8"
        )
        report["tensorrt_quality"] = fp32_quality
        report["quality"] = quality_result(fp32_quality, runtime="tensorrt")
        if not fp32_quality["passed"]:
            report["conversion_status"] = "rejected"
            report["status"] = "rejected"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raise RuntimeError(f"TensorRT FP32 task quality failed; see {report_path}")
        report["conversion_status"] = "validated"
        report["status"] = "benchmarking"
        report["performance"] = _performance_result(
            engines["fp32"],
            shapes,
            output_dir,
            "fp32",
            warmup_ms,
            iterations,
            benchmark_input_spec,
            throughput_streams,
        )
        report["status"] = "completed"

        fp16_validation: dict[str, Any] | None = None
        try:
            fp16_build_path = output_dir / "trtexec-fp16-build.log"
            build_engine(
                onnx_path,
                engines["fp16"],
                shapes,
                "fp16",
                log_path=fp16_build_path,
            )
            fp16_validation, fp16_validation_log = validate_engine(
                engines["fp16"],
                shapes,
                parity_dir,
                output_dir / ".parity-fp16",
                "fp16",
            )
            (output_dir / "trtexec-fp16-validation.log").write_text(
                fp16_validation_log, encoding="utf-8"
            )
            fp16_result: dict[str, Any] = {
                "conversion_status": (
                    "validated" if fp16_validation["passed"] else "rejected"
                ),
                "status": "benchmarking" if fp16_validation["passed"] else "rejected",
                "precision_policy": "fp16-enabled-no-tf32",
                "validation": fp16_validation,
            }
            if fp16_validation["passed"]:
                fp16_quality, fp16_quality_log = evaluate_engine_quality(
                    engines["fp16"],
                    shapes,
                    dataset_root,
                    output_dir / ".quality-fp16",
                    bundle,
                )
                (output_dir / "trtexec-fp16-quality.log").write_text(
                    fp16_quality_log, encoding="utf-8"
                )
                fp16_result["quality"] = fp16_quality
                if fp16_quality["passed"]:
                    fp16_result["performance"] = _performance_result(
                        engines["fp16"],
                        shapes,
                        output_dir,
                        "fp16",
                        warmup_ms,
                        iterations,
                        benchmark_input_spec,
                        throughput_streams,
                    )
                    fp16_result["status"] = "completed"
                else:
                    fp16_result["conversion_status"] = "rejected"
                    fp16_result["status"] = "rejected"
        except (OSError, RuntimeError, ValueError) as exc:
            fp16_result = {
                "conversion_status": (
                    "validated"
                    if fp16_validation is not None and fp16_validation["passed"]
                    else "unavailable"
                ),
                "status": (
                    "benchmark_failed"
                    if fp16_validation is not None and fp16_validation["passed"]
                    else "unavailable"
                ),
                "precision_policy": "fp16-enabled-no-tf32",
                "error": str(exc)[-4000:],
            }
            if fp16_validation is not None:
                fp16_result["validation"] = fp16_validation
        report["additional_benchmarks"] = {"fp16": fp16_result}
        report["artifacts"] = {
            "fp32_build_log": "trtexec-build.log",
            "fp32_validation_log": "trtexec-validation.log",
            "fp32_quality_log": "trtexec-quality.log",
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report_path
    except (OSError, RuntimeError, ValueError) as exc:
        if report.get("status") == "benchmarking":
            report["status"] = "benchmark_failed"
        elif report.get("status") != "rejected":
            report["status"] = "unavailable"
        report["error"] = str(exc)[-4000:]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise
    finally:
        for engine_path in engines.values():
            engine_path.unlink(missing_ok=True)
        shutil.rmtree(output_dir / ".parity-fp32", ignore_errors=True)
        shutil.rmtree(output_dir / ".parity-fp16", ignore_errors=True)
        shutil.rmtree(output_dir / ".quality-fp32", ignore_errors=True)
        shutil.rmtree(output_dir / ".quality-fp16", ignore_errors=True)


def _command_value(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout + result.stderr).strip() or None
