from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

SHORT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def load_metrics(path: Path, experiment_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Benchmark candidate is missing {path.name}")
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError("metrics.json must contain a JSON object")
    required_strings = ("experiment_name", "author", "hypothesis", "task", "recipe")
    for field in required_strings:
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"metrics.json field {field!r} must be a non-empty string")
    if value["experiment_name"] != experiment_name:
        raise ValueError("metrics.json experiment_name must match the directory name")
    if value["task"] not in {"segmentation", "depth"}:
        raise ValueError("metrics.json task must be segmentation or depth")
    if SHORT_NAME.fullmatch(value["recipe"]) is None:
        raise ValueError("metrics.json recipe must use snake_case")
    metrics = normalize_task_metrics(value.get("metrics"), value["task"], "metrics")
    baseline = normalize_task_metrics(
        value.get("baseline_metrics"), value["task"], "baseline_metrics"
    )
    if set(metrics) != set(baseline):
        raise ValueError("metrics and baseline_metrics must contain the same keys")
    value["metrics"] = metrics
    value["baseline_metrics"] = baseline
    return value


def normalize_task_metrics(
    value: Any, task: str, field: str = "quality metrics"
) -> dict[str, float]:
    metrics = _metric_mapping(value, field)
    _validate_task_metrics(task, metrics)
    return metrics


def validate_conclusion(path: Path) -> None:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise ValueError("Benchmark candidate must contain a non-empty conclusion.md")


def _metric_mapping(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"metrics.json {field} must be a non-empty object")
    normalized = {}
    for name, metric in value.items():
        if not isinstance(name, str) or METRIC_NAME.fullmatch(name) is None:
            raise ValueError(f"metrics.json {field} contains an invalid metric name")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"metrics.json {field}.{name} must be a JSON number")
        normalized_metric = float(metric)
        if not math.isfinite(normalized_metric):
            raise ValueError(f"metrics.json {field}.{name} must be finite")
        normalized[name] = normalized_metric
    return normalized


def _reject_constant(token: str) -> None:
    raise ValueError(f"metrics.json contains non-finite value {token}")


def _validate_task_metrics(task: str, metrics: dict[str, float]) -> None:
    if task == "segmentation":
        if "mIoU" not in metrics:
            raise ValueError("segmentation metrics must contain mIoU")
        if not 0.0 <= metrics["mIoU"] <= 1.0:
            raise ValueError("mIoU must be between 0 and 1")
        return
    required = {"abs_rel", "rmse", "delta1"}
    if not required.issubset(metrics):
        raise ValueError("depth metrics must contain abs_rel, rmse, and delta1")
    if metrics["abs_rel"] < 0.0 or metrics["rmse"] < 0.0:
        raise ValueError("abs_rel and rmse must be non-negative")
    if not 0.0 <= metrics["delta1"] <= 1.0:
        raise ValueError("delta1 must be between 0 and 1")
