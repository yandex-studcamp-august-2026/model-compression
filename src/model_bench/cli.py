from __future__ import annotations

import argparse
from pathlib import Path

from . import adapter, cpu, export, storage, tensorrt
from .discovery import (
    Backend,
    Candidate,
    candidates_json,
    discover_candidates,
    reject_mixed_infrastructure_candidate_change,
)


def _required_candidate(experiment: Path) -> Candidate:
    candidates = discover_candidates(experiment.parent, explicit=experiment)
    if len(candidates) != 1:
        raise RuntimeError(f"{experiment} is not a benchmark candidate")
    return candidates[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    discover = commands.add_parser("discover", help="find changed benchmark candidates")
    discover.add_argument("--experiments", type=Path, default=Path("experiments"))
    discover.add_argument("--base")
    discover.add_argument("--head")
    discover.add_argument("--experiment", type=Path)
    discover.add_argument("--backend", choices=[item.value for item in Backend])
    discover.add_argument("--reject-mixed-infrastructure", action="store_true")

    fetch = commands.add_parser("fetch", help="download a candidate checkpoint")
    fetch.add_argument("experiment", type=Path)
    fetch.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser(
        "validate", help="validate an experiment contract without downloading weights"
    )
    validate.add_argument("experiment", type=Path)

    export_command = commands.add_parser(
        "export", help="export a candidate and verify numerical parity"
    )
    export_command.add_argument("experiment", type=Path)
    export_command.add_argument("--weights", type=Path)
    export_command.add_argument("--dataset", type=Path)
    export_command.add_argument("--output", type=Path, required=True)

    benchmark_cpu = commands.add_parser(
        "benchmark-cpu", help="benchmark an ONNX bundle with ONNX Runtime"
    )
    benchmark_cpu.add_argument("--bundle", type=Path, required=True)
    benchmark_cpu.add_argument("--results", type=Path, required=True)
    benchmark_cpu.add_argument("--warmup-iterations", type=int, default=20)
    benchmark_cpu.add_argument("--iterations", type=int, default=1000)
    benchmark_cpu.add_argument("--threads", type=int, default=2)
    benchmark_cpu.add_argument(
        "--throughput-workers",
        type=int,
        default=4,
        help="maximum worker count included in the CPU throughput sweep",
    )

    benchmark_gpu = commands.add_parser(
        "benchmark-gpu", help="build and benchmark TensorRT on the local GPU"
    )
    benchmark_gpu.add_argument("--bundle", type=Path, required=True)
    benchmark_gpu.add_argument("--results", type=Path, required=True)
    benchmark_gpu.add_argument("--warmup-ms", type=int, default=5000)
    benchmark_gpu.add_argument("--iterations", type=int, default=1000)
    benchmark_gpu.add_argument(
        "--throughput-streams",
        type=int,
        default=8,
        help="maximum TensorRT stream count included in the throughput sweep",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.command == "discover":
        if args.reject_mixed_infrastructure:
            if args.base is None or args.head is None:
                raise ValueError(
                    "--reject-mixed-infrastructure requires --base and --head"
                )
            reject_mixed_infrastructure_candidate_change(
                args.experiments, args.base, args.head
            )
        backend = Backend(args.backend) if args.backend else None
        candidates = discover_candidates(
            args.experiments,
            base=args.base,
            head=args.head,
            explicit=args.experiment,
            backend=backend,
        )
        print(candidates_json(candidates))
        return

    if args.command == "fetch":
        candidate = _required_candidate(args.experiment)
        storage.download_weights(candidate.weights_url, args.output)
        return

    if args.command == "validate":
        candidate = _required_candidate(args.experiment)
        module = adapter.load_forward_module(candidate.forward)
        adapter.find_model_class(module)
        names = adapter.input_names(module)
        shapes = adapter.input_shapes(module)
        outputs = adapter.output_names(module)
        adapter.quality_evaluator(module)
        if len(names) != len(shapes):
            raise ValueError("INPUT_NAMES and INPUT_SHAPES must have equal lengths")
        print(
            f"valid candidate: {candidate.name} backend={candidate.backend.value} "
            f"inputs={dict(zip(names, shapes, strict=True))} outputs={outputs}"
        )
        return

    if args.command == "export":
        candidate = _required_candidate(args.experiment)
        report = export.export_candidate(
            candidate, args.output, args.weights, args.dataset
        )
    elif args.command == "benchmark-cpu":
        report = cpu.benchmark_cpu_bundle(
            args.bundle,
            args.results,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            threads=args.threads,
            throughput_workers=args.throughput_workers,
        )
    else:
        report = tensorrt.benchmark_gpu_bundle(
            args.bundle,
            args.results,
            warmup_ms=args.warmup_ms,
            iterations=args.iterations,
            throughput_streams=args.throughput_streams,
        )

    print(report)


if __name__ == "__main__":
    main()
