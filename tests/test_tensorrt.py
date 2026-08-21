import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_bench.reporting import export_latency_artifacts, summarize_latency
from model_bench.tensorrt import (
    benchmark_engine,
    build_engine,
    evaluate_engine_quality,
    load_gpu_compute_samples,
    parse_throughput,
    parse_trtexec_version,
    run_trtexec,
    trtexec_runtime_version,
    validate_engine,
)
from model_bench.validation import compare_raw_tensors


class TensorRTMetricsTest(unittest.TestCase):
    def test_parses_v100_compatible_trtexec_version(self):
        self.assertEqual(parse_trtexec_version("TensorRT v100400"), "10.4.0")
        self.assertEqual(
            parse_trtexec_version("TensorRT version: 10.4.0.26"),
            "10.4.0.26",
        )

    def test_rejects_unrecognized_trtexec_version(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            parse_trtexec_version("unknown")

    @patch("model_bench.tensorrt.run_trtexec")
    def test_reads_runtime_version_from_supported_help_output(self, run):
        run.return_value = "TensorRT v100400"

        self.assertEqual(trtexec_runtime_version(), "10.4.0")
        run.assert_called_once_with(["--help"])

    @patch("model_bench.tensorrt.shutil.which", return_value="/usr/bin/trtexec")
    @patch("model_bench.tensorrt.subprocess.run")
    def test_preserves_failed_trtexec_log(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="builder output", stderr="builder error"
        )
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "build.log"

            with self.assertRaisesRegex(RuntimeError, "exit code 1"):
                run_trtexec(["--onnx=model.onnx"], log_path=log)

            self.assertEqual(log.read_text(), "builder outputbuilder error")

    @patch("model_bench.tensorrt.shutil.which", return_value="/usr/bin/trtexec")
    @patch("model_bench.tensorrt.subprocess.run")
    def test_builds_strict_fp32_and_optional_fp16(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        build_engine(Path("model.onnx"), Path("fp32.plan"), "x:1x3", "fp32")
        fp32_command = run.call_args.args[0]
        self.assertIn("--noTF32", fp32_command)
        self.assertNotIn("--fp16", fp32_command)

        build_engine(Path("model.onnx"), Path("fp16.plan"), "x:1x3", "fp16")
        fp16_command = run.call_args.args[0]
        self.assertIn("--noTF32", fp16_command)
        self.assertIn("--fp16", fp16_command)

    @patch("model_bench.tensorrt.shutil.which", return_value="/usr/bin/trtexec")
    @patch("model_bench.tensorrt.subprocess.run")
    def test_benchmark_uses_model_only_latency_flags(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Throughput: 12.5 qps\n", stderr=""
        )

        throughput, _ = benchmark_engine(
            Path("model.plan"),
            "input:1x3x32x32",
            5000,
            1000,
            Path("times.json"),
            "input:/tmp/input.raw",
        )

        command = run.call_args.args[0]
        self.assertEqual(throughput, 12.5)
        self.assertIn("--infStreams=1", command)
        self.assertIn("--warmUp=5000", command)
        self.assertIn("--duration=0", command)
        self.assertIn("--iterations=1000", command)
        self.assertIn("--noDataTransfers", command)
        self.assertIn("--useCudaGraph", command)
        self.assertIn("--useSpinWait", command)
        self.assertIn("--loadInputs=input:/tmp/input.raw", command)

    def test_parses_current_trtexec_throughput(self):
        output = "[I] GPU Compute Time: mean = 13.2 ms\n[I] Throughput: 76.42 qps\n"
        self.assertEqual(parse_throughput(output), 76.42)

    def test_prefers_gpu_compute_time(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "times.json"
            path.write_text(
                json.dumps(
                    [
                        {"computeMs": 1.0, "latencyMs": 9.0},
                        {"computeMs": 2.0, "latencyMs": 10.0},
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_gpu_compute_samples(path), [1.0, 2.0])

    def test_does_not_fall_back_to_end_to_end_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "times.json"
            path.write_text(json.dumps([{"latencyMs": 9.0}]), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "computeMs"):
                load_gpu_compute_samples(path)

    def test_calculates_nearest_rank_percentiles(self):
        metrics = summarize_latency([float(value) for value in range(1, 101)])
        self.assertEqual(metrics["mean"], 50.5)
        self.assertEqual(metrics["p50"], 50.0)
        self.assertEqual(metrics["p90"], 90.0)
        self.assertEqual(metrics["p95"], 95.0)
        self.assertEqual(metrics["p99"], 99.0)

    def test_exports_csv_and_histogram(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_latency_artifacts(
                [1.0, 2.0],
                root / "latency_ms.csv",
                root / "histogram.svg",
                "Latency",
            )
            self.assertIn(
                "latency_ms", (root / "latency_ms.csv").read_text(encoding="utf-8")
            )
            histogram = (root / "histogram.svg").read_text(encoding="utf-8")
            self.assertIn("<svg", histogram)
            for percentile in ("p50", "p90", "p95", "p99"):
                self.assertIn(percentile, histogram)

    def test_compares_complete_raw_tensors_in_chunks(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.raw"
            candidate = root / "candidate.raw"
            np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(reference)
            np.array([1.0, 2.001, 3.0], dtype=np.float16).tofile(candidate)

            report = compare_raw_tensors(
                reference,
                "float32",
                candidate,
                "float16",
                elements=3,
                atol=0.01,
                rtol=0.01,
                chunk_elements=2,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["elements"], 3)
            self.assertGreater(report["max_absolute_error"], 0.0)

    @patch("model_bench.tensorrt.run_trtexec")
    def test_validates_engine_with_fixed_input_and_complete_output(self, run):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parity = root / "parity"
            parity.mkdir()
            np.array([1.0, 2.0], dtype=np.float32).tofile(parity / "input-0.raw")
            np.array([3.0, 4.0], dtype=np.float32).tofile(parity / "output-0.raw")
            np.array([2.0, 1.0], dtype=np.float32).tofile(parity / "input-1.raw")
            np.array([4.0, 3.0], dtype=np.float32).tofile(parity / "output-1.raw")
            (parity / "manifest.json").write_text(
                json.dumps(
                    {
                        "reference_runtime": "onnxruntime-cpu",
                        "inputs": {
                            "pixels": {
                                "file": "input-0.raw",
                                "dtype": "float32",
                                "shape": [1, 2],
                                "elements": 2,
                            }
                        },
                        "outputs": {
                            "logits": {
                                "file": "output-0.raw",
                                "dtype": "float32",
                                "shape": [1, 2],
                                "elements": 2,
                            }
                        },
                        "additional_samples": [
                            {
                                "id": 1,
                                "inputs": {
                                    "pixels": {
                                        "file": "input-1.raw",
                                        "dtype": "float32",
                                        "shape": [1, 2],
                                        "elements": 2,
                                    }
                                },
                                "outputs": {
                                    "logits": {
                                        "file": "output-1.raw",
                                        "dtype": "float32",
                                        "shape": [1, 2],
                                        "elements": 2,
                                    }
                                },
                            }
                        ],
                        "tolerances": {"fp32": {"atol": 1e-4, "rtol": 1e-4}},
                        "semantic_validation": {
                            "task": "depth",
                            "output_name": None,
                            "class_axis": None,
                            "minimum_pixel_agreement": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            work = root / "work"

            def write_output(arguments, cwd=None):
                self.assertIn("--dumpRawBindingsToFile", arguments)
                self.assertTrue(
                    any(value.startswith("--loadInputs=pixels:") for value in arguments)
                )
                values = [3.0, 4.0] if cwd.name == "sample-0" else [4.0, 3.0]
                np.array(values, dtype=np.float32).tofile(
                    cwd / "logits.output.1.2.FP32.raw"
                )
                return "ok"

            run.side_effect = write_output
            report, _ = validate_engine(
                root / "model.plan",
                "pixels:1x2",
                parity,
                work,
                "fp32",
            )

            self.assertTrue(report["passed"])
            self.assertEqual(
                report["samples"][0]["outputs"]["logits"]["candidate_shape"],
                [1, 2],
            )
            self.assertEqual(len(report["samples"]), 2)
            self.assertEqual(run.call_count, 2)

    @patch("model_bench.tensorrt.run_trtexec")
    def test_evaluates_tensorrt_miou_on_bound_dataset(self, run):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            samples = dataset / "samples"
            samples.mkdir(parents=True)
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            label = np.array([[0, 1], [0, 1]], dtype=np.uint8)
            np.savez(samples / "sample.npz", image=image, label=label)
            provenance = {
                "uri": "s3://datasets/cityscapes",
                "include": ["samples"],
                "format": "cityscapes_segmentation_npz_v1",
                "object_count": 1,
                "total_bytes": 100,
                "listing_sha256": "a" * 64,
            }
            (dataset / "_dataset_manifest.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            bundle = {
                "input_names": ["pixels"],
                "output_names": ["logits"],
                "input_shapes": [[1, 3, 2, 2]],
                "training_metrics": {"task": "segmentation"},
                "conversion_quality": {
                    "source_metrics": {"mIoU": 1.0},
                    "tolerance": 1e-4,
                    "dataset": provenance,
                },
            }

            def write_logits(_arguments, cwd=None):
                logits = np.array(
                    [[[[2, 0], [2, 0]], [[0, 2], [0, 2]]]], dtype=np.float32
                )
                logits.tofile(cwd / "logits.output.1.2.2.2.FP32.raw")
                return "ok"

            run.side_effect = write_logits
            with patch.multiple(
                "model_bench.tensorrt",
                CITYSCAPES_CLASSES=2,
                CITYSCAPES_IMAGE_SHAPE=(2, 2, 3),
                CITYSCAPES_TARGET_SHAPE=(2, 2),
            ):
                report, _ = evaluate_engine_quality(
                    root / "model.plan",
                    "pixels:1x3x2x2",
                    dataset,
                    root / "work",
                    bundle,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["post_conversion_metrics"], {"mIoU": 1.0})
            self.assertEqual(report["sample_count"], 1)

    def test_rejects_quality_dataset_with_different_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            dataset.mkdir()
            actual = {
                "uri": "s3://datasets/wrong",
                "include": ["samples"],
                "format": "cityscapes_segmentation_npz_v1",
                "object_count": 1,
                "total_bytes": 100,
                "listing_sha256": "a" * 64,
            }
            (dataset / "_dataset_manifest.json").write_text(
                json.dumps(actual), encoding="utf-8"
            )
            bundle = {
                "conversion_quality": {
                    "dataset": {**actual, "uri": "s3://datasets/cityscapes"}
                }
            }

            with self.assertRaisesRegex(ValueError, "provenance"):
                evaluate_engine_quality(
                    root / "model.plan",
                    "pixels:1x3x2x2",
                    dataset,
                    root / "work",
                    bundle,
                )


if __name__ == "__main__":
    unittest.main()
