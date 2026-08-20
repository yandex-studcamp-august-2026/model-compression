import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_bench.cli import main


class CliTest(unittest.TestCase):
    def test_fetch_uses_candidate_weights_url(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiment = root / "experiments/model"
            experiment.mkdir(parents=True)
            (experiment / "CPU").write_bytes(b"")
            (experiment / "forward.py").write_text("# contract\n", encoding="utf-8")
            (experiment / "checkpoint.pt").write_bytes(b"weights")
            (experiment / "weights.url").write_text("checkpoint.pt\n", encoding="utf-8")
            (experiment / "weights.sha256").write_text(
                "a" * 64 + "\n", encoding="ascii"
            )
            (experiment / "dataset.json").write_text(
                json.dumps({"uri": "s3://datasets/validation", "include": ["images"]}),
                encoding="utf-8",
            )
            (experiment / "metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "model",
                        "author": "tester",
                        "hypothesis": "test",
                        "task": "segmentation",
                        "recipe": "test",
                        "metrics": {"mIoU": 0.6},
                        "baseline_metrics": {"mIoU": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "conclusion.md").write_text("# Result\n", encoding="utf-8")
            output = root / "download/checkpoint.pt"

            with patch(
                "sys.argv",
                [
                    "model-bench",
                    "fetch",
                    str(experiment),
                    "--output",
                    str(output),
                ],
            ):
                main()

            self.assertEqual(output.read_bytes(), b"weights")

    @patch("model_bench.cli.cpu.benchmark_cpu_bundle")
    def test_benchmark_prints_report_path(self, benchmark):
        benchmark.return_value = Path("results/model/report.json")

        with (
            patch(
                "sys.argv",
                [
                    "model-bench",
                    "benchmark-cpu",
                    "--bundle",
                    "bundles/model",
                    "--results",
                    "results",
                ],
            ),
            patch("builtins.print") as output,
        ):
            main()

        output.assert_called_once_with(Path("results/model/report.json"))


if __name__ == "__main__":
    unittest.main()
