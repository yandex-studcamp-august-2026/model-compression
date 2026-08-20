#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    paths = sorted(args.reports.glob("*/report.json"))
    if not paths:
        raise FileNotFoundError(f"no reports found under {args.reports}")
    print(
        "| Experiment | Runtime | Precision | Status | Quality | Throughput "
        "| p50 | p90 | p95 | p99 |"
    )
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        precision = report.get("precision_policy", "fp32")
        _write_row(
            report,
            precision,
            report.get("status", "completed"),
            report.get("performance"),
        )
        for name, result in report.get("additional_benchmarks", {}).items():
            _write_row(report, name, result["status"], result.get("performance"))


def _write_row(
    report: dict, precision: str, status: str, performance: dict | None
) -> None:
    quality = _quality_text(report.get("quality", {}))
    if performance is None:
        print(
            f"| {report['experiment']} | {report['backend']} | {precision} "
            f"| {status} | {quality} | — | — | — | — | — |"
        )
        return
    latency = performance["latency"]
    print(
        f"| {report['experiment']} | {report['backend']} | {precision} | {status} "
        f"| {quality} "
        f"| {performance['throughput_qps']:.2f} qps "
        f"| {latency['p50']:.3f} ms "
        f"| {latency['p90']:.3f} ms "
        f"| {latency['p95']:.3f} ms "
        f"| {latency['p99']:.3f} ms |"
    )


def _quality_text(quality: dict) -> str:
    metrics = quality.get("post_conversion_metrics") or quality.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return str(quality.get("status", "not_provided"))
    values = ", ".join(f"{name}={float(value):.6g}" for name, value in metrics.items())
    label = "post-conversion" if quality.get("status") == "measured" else "source"
    return f"{label} ({values})"


if __name__ == "__main__":
    main()
