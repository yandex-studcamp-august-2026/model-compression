import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

HAS_EXPORT_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "onnx", "onnxruntime", "torch")
)


@unittest.skipUnless(HAS_EXPORT_STACK, "export dependencies are not installed")
class PipelineIntegrationTest(unittest.TestCase):
    def test_exports_validates_and_benchmarks_tiny_model(self):
        import torch

        from model_bench.cpu import benchmark_cpu_bundle
        from model_bench.discovery import candidate_from_directory
        from model_bench.export import export_candidate

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiment = root / "experiments" / "tiny_model"
            experiment.mkdir(parents=True)
            (experiment / "CPU").write_bytes(b"")
            shutil.copy2(
                Path("tests/fixtures/tiny_forward.py"),
                experiment / "forward.py",
            )
            weights = experiment / "weights.pt"
            torch.save(
                {
                    "network.weight": torch.randn(19, 3, 1, 1),
                    "network.bias": torch.randn(19),
                },
                weights,
            )
            (experiment / "weights.sha256").write_text(
                hashlib.sha256(weights.read_bytes()).hexdigest() + "\n",
                encoding="ascii",
            )
            (experiment / "weights.url").write_text("weights.pt\n", encoding="utf-8")
            (experiment / "dataset.json").write_text(
                json.dumps(
                    {
                        "uri": "s3://datasets/validation",
                        "include": ["samples"],
                        "format": "cityscapes_segmentation_npz_v1",
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "tiny_model",
                        "author": "tester",
                        "hypothesis": "integration test",
                        "task": "segmentation",
                        "recipe": "integration",
                        "metrics": {"mIoU": 0.6},
                        "baseline_metrics": {"mIoU": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "conclusion.md").write_text("# Result\n", encoding="utf-8")
            candidate = candidate_from_directory(experiment)
            self.assertIsNotNone(candidate)

            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "sample.ready").write_text("ok\n", encoding="utf-8")
            (dataset / "_dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "uri": "s3://datasets/validation",
                        "include": ["samples"],
                        "format": "cityscapes_segmentation_npz_v1",
                        "object_count": 1,
                        "total_bytes": 3,
                        "listing_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )

            export_candidate(candidate, root / "bundles", weights, dataset)
            report_path = benchmark_cpu_bundle(
                root / "bundles" / "tiny_model",
                root / "results",
                warmup_iterations=2,
                iterations=10,
                threads=1,
            )

            conversion = json.loads(
                (root / "bundles/tiny_model/conversion_report.json").read_text()
            )
            tensorrt_parity = json.loads(
                (root / "bundles/tiny_model/tensorrt_parity/manifest.json").read_text()
            )
            report = json.loads(report_path.read_text())
            self.assertTrue(conversion["passed"])
            self.assertEqual(tensorrt_parity["reference_runtime"], "onnxruntime-cpu")
            self.assertEqual(
                tensorrt_parity["outputs"]["logits"]["shape"], [1, 19, 32, 32]
            )
            self.assertEqual(report["measured_iterations"], 10)
            self.assertEqual(report["performance"]["latency_kind"], "host_wall_clock")
            self.assertEqual(
                report["performance"]["throughput_kind"],
                "best_concurrency_sweep",
            )
            self.assertEqual(len(tensorrt_parity["additional_samples"]), 2)
            self.assertEqual(report["quality"]["status"], "measured")
            self.assertTrue(report["quality"]["passed"])


if __name__ == "__main__":
    unittest.main()
