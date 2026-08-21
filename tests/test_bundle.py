import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from model_bench.bundle import load_bundle, load_parity_manifest


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def conversion_quality() -> dict:
    return {
        "passed": True,
        "task": "segmentation",
        "source_metrics": {"mIoU": 0.6},
        "post_conversion_metrics": {"mIoU": 0.6},
        "absolute_deltas": {"mIoU": 0.0},
        "tolerance": 1e-4,
        "dataset": {
            "uri": "s3://datasets/validation",
            "include": ["samples"],
            "format": "cityscapes_segmentation_npz_v1",
            "object_count": 1,
            "total_bytes": 1,
            "listing_sha256": "a" * 64,
        },
    }


class BundleValidationTest(unittest.TestCase):
    def test_rejects_post_conversion_quality_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.onnx").write_bytes(b"model")
            (root / "benchmark_inputs.npz").write_bytes(b"inputs")
            quality = conversion_quality()
            quality["post_conversion_metrics"]["mIoU"] = 0.5
            quality["absolute_deltas"]["mIoU"] = 0.1
            metadata = {
                "schema_version": 6,
                "experiment": "quality_regression",
                "backends": ["cpu"],
                "input_names": ["pixels"],
                "input_shapes": [[1, 3, 32, 32]],
                "output_names": ["logits"],
                "training_metrics": {"task": "segmentation"},
                "conversion_quality": quality,
                "onnx_sha256": digest(b"model"),
                "onnx_files": {"model.onnx": digest(b"model")},
                "benchmark_inputs_sha256": digest(b"inputs"),
            }
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
                load_bundle(root, expected_backend="cpu")

    def test_rejects_result_path_injection_and_onnx_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.onnx").write_bytes(b"model")
            (root / "benchmark_inputs.npz").write_bytes(b"inputs")
            metadata = {
                "schema_version": 6,
                "experiment": "../escape",
                "backends": ["cpu"],
                "input_names": ["pixels"],
                "input_shapes": [[1, 3, 32, 32]],
                "output_names": ["logits"],
                "training_metrics": {"task": "segmentation"},
                "conversion_quality": conversion_quality(),
                "onnx_sha256": digest(b"model"),
                "onnx_files": {"model.onnx": digest(b"model")},
                "benchmark_inputs_sha256": digest(b"inputs"),
            }
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "experiment name"):
                load_bundle(root, expected_backend="cpu")

            metadata["experiment"] = "safe_model"
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "model.onnx").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_bundle(root, expected_backend="cpu")

    def test_rejects_tampered_onnx_external_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.onnx").write_bytes(b"model")
            (root / "model.onnx.data").write_bytes(b"weights")
            (root / "benchmark_inputs.npz").write_bytes(b"inputs")
            metadata = {
                "schema_version": 6,
                "experiment": "safe_model",
                "backends": ["gpu"],
                "input_names": ["pixels"],
                "input_shapes": [[1, 3, 32, 32]],
                "output_names": ["logits"],
                "training_metrics": {"task": "segmentation"},
                "conversion_quality": conversion_quality(),
                "onnx_sha256": digest(b"model"),
                "onnx_files": {
                    "model.onnx": digest(b"model"),
                    "model.onnx.data": digest(b"weights"),
                },
                "benchmark_inputs_sha256": digest(b"inputs"),
            }
            (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")

            load_bundle(root, expected_backend="gpu")
            (root / "model.onnx.data").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "model.onnx.data"):
                load_bundle(root, expected_backend="gpu")

    def test_rejects_relaxed_parity_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tensor = root / "input.raw"
            tensor.write_bytes(b"\0\0\0\0")
            manifest = {
                "reference_runtime": "onnxruntime-cpu",
                "inputs": {
                    "pixels": {
                        "file": tensor.name,
                        "dtype": "float32",
                        "shape": [1],
                        "elements": 1,
                        "sha256": digest(tensor.read_bytes()),
                    }
                },
                "outputs": {
                    "logits": {
                        "file": tensor.name,
                        "dtype": "float32",
                        "shape": [1],
                        "elements": 1,
                        "sha256": digest(tensor.read_bytes()),
                    }
                },
                "tolerances": {
                    "fp32": {"atol": 1.0, "rtol": 1e-4},
                    "fp16": {"atol": 1e-3, "rtol": 1e-2},
                },
                "semantic_validation": {
                    "task": "segmentation",
                    "output_name": "logits",
                    "class_axis": 1,
                    "minimum_pixel_agreement": 0.9999,
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            bundle = {
                "input_names": ["pixels"],
                "output_names": ["logits"],
                "training_metrics": {"task": "segmentation"},
            }

            with self.assertRaisesRegex(ValueError, "exceeds policy"):
                load_parity_manifest(root, bundle)

            manifest["tolerances"]["fp32"]["atol"] = 1e-4
            manifest["inputs"]["pixels"]["dtype"] = "float64"
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported.*dtype"):
                load_parity_manifest(root, bundle)

    def test_rejects_semantic_task_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tensor = root / "tensor.raw"
            tensor.write_bytes(b"\0\0\0\0")
            metadata = {
                "file": tensor.name,
                "dtype": "float32",
                "shape": [1],
                "elements": 1,
                "sha256": digest(tensor.read_bytes()),
            }
            manifest = {
                "reference_runtime": "onnxruntime-cpu",
                "inputs": {"pixels": metadata},
                "outputs": {"depth": metadata},
                "tolerances": {
                    "fp32": {"atol": 1e-4, "rtol": 1e-4},
                    "fp16": {"atol": 1e-3, "rtol": 1e-2},
                },
                "semantic_validation": {
                    "task": "depth",
                    "output_name": None,
                    "class_axis": None,
                    "minimum_pixel_agreement": None,
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            bundle = {
                "input_names": ["pixels"],
                "output_names": ["depth"],
                "training_metrics": {"task": "segmentation"},
            }

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_parity_manifest(root, bundle)
