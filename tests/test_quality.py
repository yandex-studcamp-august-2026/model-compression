import json
import tempfile
import unittest
from pathlib import Path

from model_bench.quality import load_metrics


class QualityMetadataTest(unittest.TestCase):
    def test_loads_team_metrics_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "experiment_name": "student_model",
                        "author": "tester",
                        "hypothesis": "improve quality",
                        "task": "segmentation",
                        "recipe": "soft_label_kl",
                        "metrics": {"mIoU": 0.612},
                        "baseline_metrics": {"mIoU": 0.598},
                    }
                ),
                encoding="utf-8",
            )

            result = load_metrics(path, "student_model")

            self.assertEqual(result["metrics"]["mIoU"], 0.612)

    def test_rejects_mismatched_metric_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "experiment_name": "student_model",
                        "author": "tester",
                        "hypothesis": "improve quality",
                        "task": "depth",
                        "recipe": "distillation",
                        "metrics": {"abs_rel": 0.2, "rmse": 1.2, "delta1": 0.8},
                        "baseline_metrics": {
                            "abs_rel": 0.3,
                            "rmse": 1.4,
                            "delta1": 0.7,
                            "extra": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "same keys"):
                load_metrics(path, "student_model")

    def test_rejects_non_finite_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.json"
            path.write_text(
                '{"experiment_name":"student_model","author":"tester",'
                '"hypothesis":"test","task":"segmentation","recipe":"test",'
                '"metrics":{"mIoU":NaN},"baseline_metrics":{"mIoU":0.5}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_metrics(path, "student_model")
