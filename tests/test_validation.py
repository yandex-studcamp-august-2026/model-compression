import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from model_bench.adapter import input_shapes, make_inputs
from model_bench.validation import compare_outputs, compare_segmentation_raw_tensors


class ValidationTest(unittest.TestCase):
    def test_requires_explicit_benchmark_input_shape(self):
        with self.assertRaisesRegex(ValueError, "must define INPUT_SHAPE"):
            input_shapes(ModuleType("forward"))

    def test_rejects_input_that_disagrees_with_declared_shape(self):
        import torch

        module = ModuleType("forward")
        module.make_inputs = lambda _seed: torch.zeros(1, 3, 16, 16)

        with self.assertRaisesRegex(ValueError, "expected"):
            make_inputs(module, ("pixels",), ((1, 3, 32, 32),), seed=0)

    def test_rejects_batch_size_other_than_one(self):
        module = ModuleType("forward")
        module.INPUT_SHAPE = (2, 3, 32, 32)

        with self.assertRaisesRegex(ValueError, "batch dimension 1"):
            input_shapes(module)

    def test_accepts_outputs_within_tolerance(self):
        result = compare_outputs(
            ([1.0, 2.0],),
            ([1.0, 2.00001],),
            ("logits",),
            atol=1e-4,
            rtol=1e-4,
        )

        self.assertTrue(result["passed"])

    def test_rejects_shape_mismatch(self):
        result = compare_outputs(
            ([1.0, 2.0],),
            ([1.0],),
            ("logits",),
            atol=1e-4,
            rtol=1e-4,
        )

        self.assertFalse(result["passed"])

    def test_reports_segmentation_prediction_agreement(self):
        reference = [[[[3.0]], [[1.0]]]]
        candidate = [[[[2.0]], [[1.5]]]]

        result = compare_outputs(
            (reference,),
            (candidate,),
            ("logits",),
            atol=2.0,
            rtol=0.0,
            task="segmentation",
            semantic_output_name="logits",
        )

        agreement = result["outputs"]["logits"]["segmentation_agreement"]
        self.assertEqual(agreement["pixel_agreement"], 1.0)
        self.assertEqual(agreement["mean_iou_between_predictions"], 1.0)

    def test_segmentation_label_change_fails_even_within_numeric_tolerance(self):
        reference = [[[[1.0]], [[0.99]]]]
        candidate = [[[[0.99]], [[1.0]]]]

        result = compare_outputs(
            (reference,),
            (candidate,),
            ("logits",),
            atol=0.02,
            rtol=0.0,
            task="segmentation",
            semantic_output_name="logits",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            result["outputs"]["logits"]["segmentation_agreement"]["pixel_agreement"],
            0.0,
        )

    def test_raw_segmentation_agreement_detects_tensorrt_label_change(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.raw"
            candidate = root / "candidate.raw"
            np.array([1.0, 0.99], dtype=np.float32).tofile(reference)
            np.array([0.99, 1.0], dtype=np.float32).tofile(candidate)

            agreement = compare_segmentation_raw_tensors(
                reference,
                "float32",
                candidate,
                "float32",
                [1, 2, 1, 1],
                1,
            )

            self.assertEqual(agreement["pixel_agreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
