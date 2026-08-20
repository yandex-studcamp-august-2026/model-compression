from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def quality_result(conversion_quality: object) -> dict[str, object]:
    if (
        not isinstance(conversion_quality, dict)
        or conversion_quality.get("passed") is not True
    ):
        raise ValueError("A passing post-conversion quality result is required")
    return {
        "status": "measured",
        "scope": "source_and_onnxruntime_on_validation_dataset",
        "task": conversion_quality.get("task"),
        "source_metrics": conversion_quality.get("source_metrics"),
        "post_conversion_metrics": conversion_quality.get("post_conversion_metrics"),
        "absolute_deltas": conversion_quality.get("absolute_deltas"),
        "tolerance": conversion_quality.get("tolerance"),
        "passed": True,
        "dataset": conversion_quality.get("dataset"),
        "tensorrt_interpretation": (
            "TensorRT quality is accepted only together with its separate "
            "numerical and task-semantic parity gate against ONNX Runtime."
        ),
    }


def summarize_latency(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("At least one latency sample is required")
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("Latency samples must be finite and non-negative")
    ordered = sorted(samples)

    def percentile(value: float) -> float:
        return ordered[max(0, math.ceil(value * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def export_latency_artifacts(
    samples: list[float], csv_path: Path, svg_path: Path, title: str
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("iteration", "latency_ms"))
        writer.writerows(enumerate(samples, start=1))
    _write_histogram_svg(samples, svg_path, title)


def _write_histogram_svg(
    values: list[float], output: Path, title: str, bins: int | None = None
) -> None:
    if not values:
        raise ValueError("At least one value is required")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency samples must be finite and non-negative")
    if bins is not None and bins <= 0:
        raise ValueError("Histogram bin count must be positive")
    bins = bins or _histogram_bin_count(values)
    low, high = min(values), max(values)
    if math.isclose(low, high):
        high = low + 1.0
    bin_width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        counts[min(int((value - low) / bin_width), bins - 1)] += 1

    chart_width, chart_height, margin = 900, 420, 55
    inner_width = chart_width - 2 * margin
    inner_height = chart_height - 2 * margin
    bar_width = inner_width / bins
    peak = max(counts)
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    root = ET.Element(
        f"{{{namespace}}}svg",
        {
            "width": str(chart_width),
            "height": str(chart_height),
            "viewBox": f"0 0 {chart_width} {chart_height}",
        },
    )
    ET.SubElement(
        root, f"{{{namespace}}}rect", width="100%", height="100%", fill="white"
    )
    heading = ET.SubElement(
        root,
        f"{{{namespace}}}text",
        x=str(chart_width / 2),
        y="28",
        **{"text-anchor": "middle", "font-family": "sans-serif", "font-size": "18"},
    )
    heading.text = title
    for index, count in enumerate(counts):
        height = inner_height * count / peak
        x = margin + index * bar_width
        y = margin + inner_height - height
        ET.SubElement(
            root,
            f"{{{namespace}}}rect",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            width=f"{max(bar_width - 1, 1):.2f}",
            height=f"{height:.2f}",
            fill="#2563eb",
        )
    for attributes in (
        {
            "x1": margin,
            "y1": margin + inner_height,
            "x2": margin + inner_width,
            "y2": margin + inner_height,
        },
        {"x1": margin, "y1": margin, "x2": margin, "y2": margin + inner_height},
    ):
        ET.SubElement(
            root,
            f"{{{namespace}}}line",
            **{key: str(value) for key, value in attributes.items()},
            stroke="#111827",
        )
    _svg_text(root, namespace, margin, chart_height - 14, f"{low:.3f} ms")
    _svg_text(
        root,
        namespace,
        margin + inner_width,
        chart_height - 14,
        f"{high:.3f} ms",
        anchor="end",
    )
    _svg_text(
        root,
        namespace,
        16,
        chart_height / 2,
        "count",
        anchor="middle",
        transform=f"rotate(-90 16 {chart_height / 2})",
    )
    percentiles = summarize_latency(values)
    marker_styles = (
        ("p50", "#16a34a", 46),
        ("p90", "#d97706", 61),
        ("p95", "#dc2626", 76),
        ("p99", "#7c3aed", 91),
    )
    for name, color, label_y in marker_styles:
        value = percentiles[name]
        x = margin + inner_width * (value - low) / (high - low)
        ET.SubElement(
            root,
            f"{{{namespace}}}line",
            x1=f"{x:.2f}",
            y1=str(margin),
            x2=f"{x:.2f}",
            y2=str(margin + inner_height),
            stroke=color,
            **{"stroke-width": "2", "stroke-dasharray": "5 4"},
        )
        _svg_text(
            root,
            namespace,
            min(max(x + 4, margin + 4), margin + inner_width - 4),
            label_y,
            f"{name} {value:.3f} ms",
            anchor="end" if x > margin + inner_width * 0.82 else None,
            fill=color,
        )
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=True)


def _histogram_bin_count(values: list[float]) -> int:
    """Choose a stable bin count using the Freedman-Diaconis rule."""
    if len(values) < 2 or math.isclose(min(values), max(values)):
        return 1
    ordered = sorted(values)

    def interpolated_quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    iqr = interpolated_quantile(0.75) - interpolated_quantile(0.25)
    width = 2 * iqr / math.pow(len(values), 1 / 3)
    if width <= 0:
        count = math.ceil(math.sqrt(len(values)))
    else:
        count = math.ceil((ordered[-1] - ordered[0]) / width)
    return min(max(count, 1), 80)


def _svg_text(
    root: ET.Element,
    namespace: str,
    x: float,
    y: float,
    value: str,
    *,
    anchor: str | None = None,
    transform: str | None = None,
    fill: str | None = None,
) -> None:
    attributes = {
        "x": str(x),
        "y": str(y),
        "font-family": "sans-serif",
        "font-size": "12",
    }
    if anchor:
        attributes["text-anchor"] = anchor
    if transform:
        attributes["transform"] = transform
    if fill:
        attributes["fill"] = fill
    element = ET.SubElement(root, f"{{{namespace}}}text", **attributes)
    element.text = value
