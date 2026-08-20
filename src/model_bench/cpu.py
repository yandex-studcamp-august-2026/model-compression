from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from model_bench.bundle import load_bundle, safe_result_directory
from model_bench.reporting import (
    export_latency_artifacts,
    quality_result,
    summarize_latency,
)


def benchmark_cpu_bundle(
    bundle_dir: Path,
    results_root: Path,
    warmup_iterations: int,
    iterations: int,
    threads: int = 2,
    throughput_workers: int = 2,
) -> Path:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install project with the [cpu] extra") from exc
    if warmup_iterations < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if threads <= 0 or throughput_workers <= 0:
        raise ValueError("threads and throughput workers must be positive")

    bundle = load_bundle(bundle_dir, expected_backend="cpu")
    onnx_path = bundle_dir / "model.onnx"
    inputs_path = bundle_dir / "benchmark_inputs.npz"
    if not onnx_path.is_file() or not inputs_path.is_file():
        raise ValueError(f"Invalid benchmark bundle: {bundle_dir}")

    with np.load(inputs_path) as archive:
        feed = {name: archive[name] for name in bundle["input_names"]}

    session = _create_session(ort, onnx_path, threads)
    for _ in range(warmup_iterations):
        session.run(None, feed)
    samples = []
    for _ in range(iterations):
        iteration_started = time.perf_counter_ns()
        session.run(None, feed)
        samples.append((time.perf_counter_ns() - iteration_started) / 1_000_000)

    throughput_session = _create_session(ort, onnx_path, 1)
    selected_workers, throughput_qps, throughput_sweep = _benchmark_throughput(
        throughput_session, feed, iterations, throughput_workers
    )

    output_dir = safe_result_directory(results_root, str(bundle["experiment"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    export_latency_artifacts(
        samples,
        output_dir / "latency_ms.csv",
        output_dir / "latency_histogram.svg",
        "ONNX Runtime CPU latency",
    )
    report = {
        "schema_version": 1,
        "experiment": bundle["experiment"],
        "git_sha": bundle.get("git_sha"),
        "created_at": datetime.now(UTC).isoformat(),
        "backend": "onnxruntime-cpu",
        "status": "completed",
        "conversion_status": "validated",
        "runtime_version": ort.__version__,
        "platform": platform.platform(),
        "processor": _processor_name(),
        "threads": threads,
        "logical_cpu_count": os.cpu_count(),
        "max_throughput_workers": throughput_workers,
        "input_shapes": bundle["input_shapes"],
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "performance": {
            "latency_kind": "host_wall_clock",
            "latency_unit": "ms",
            "latency": summarize_latency(samples),
            "throughput_kind": "best_concurrency_sweep",
            "throughput_qps": throughput_qps,
            "throughput_workers": selected_workers,
            "throughput_sweep": throughput_sweep,
        },
        "quality": quality_result(bundle.get("conversion_quality")),
        "conversion_validation": bundle.get("conversion_validation"),
        "training_metrics": bundle.get("training_metrics"),
        "export_environment": bundle.get("export_environment"),
        "source_artifacts": {
            "weights_sha256": bundle.get("weights_sha256"),
            "onnx_sha256": bundle.get("onnx_sha256"),
            "onnx_files": bundle.get("onnx_files"),
            "benchmark_inputs_sha256": bundle.get("benchmark_inputs_sha256"),
        },
        "artifacts": {
            "latency_samples": "latency_ms.csv",
            "latency_histogram": "latency_histogram.svg",
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report_path


def _create_session(ort: object, onnx_path: Path, threads: int):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _run_repeated(session: object, feed: dict[str, object], count: int) -> None:
    for _ in range(count):
        session.run(None, feed)


def _benchmark_throughput(
    session: object,
    feed: dict[str, object],
    iterations: int,
    max_workers: int,
) -> tuple[int, float, list[dict[str, float | int]]]:
    candidates = []
    workers = 1
    while workers <= max_workers:
        candidates.append(workers)
        workers *= 2
    if candidates[-1] != max_workers:
        candidates.append(max_workers)

    sweep = []
    for worker_count in candidates:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            warmups = [
                executor.submit(session.run, None, feed) for _ in range(worker_count)
            ]
            for warmup in warmups:
                warmup.result()
            work = [iterations // worker_count] * worker_count
            for index in range(iterations % worker_count):
                work[index] += 1
            started = time.perf_counter_ns()
            futures = [
                executor.submit(_run_repeated, session, feed, count)
                for count in work
                if count
            ]
            for future in futures:
                future.result()
            elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
        sweep.append(
            {
                "workers": worker_count,
                "throughput_qps": iterations / elapsed_seconds,
            }
        )
    best = max(sweep, key=lambda result: result["throughput_qps"])
    return int(best["workers"]), float(best["throughput_qps"]), sweep


def _processor_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"
