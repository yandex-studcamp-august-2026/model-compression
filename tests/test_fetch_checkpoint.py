import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "fetch_checkpoint", Path("scripts/fetch_checkpoint.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load fetch_checkpoint.py")
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class TrustedFetchTest(unittest.TestCase):
    def test_accepts_first_level_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "experiments/model"
            directory.mkdir(parents=True)
            (directory / "CPU").write_bytes(b"")
            url = directory / "weights.url"
            url.write_text("s3://models/checkpoint.pt\n", encoding="utf-8")
            (directory / "weights.sha256").write_text("a" * 64 + "\n", encoding="ascii")

            self.assertEqual(
                FETCH.validated_url_file(root, Path("experiments/model")), url.resolve()
            )

    def test_validates_expected_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            checksum = Path(temp) / "weights.sha256"
            checksum.write_text("invalid\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "weights.sha256"):
                FETCH.read_expected_sha256(checksum)

    def test_rejects_checkpoint_above_size_limit(self):
        with self.assertRaisesRegex(ValueError, "outside the allowed range"):
            FETCH.validate_content_length(101, 100)

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "experiments"):
                FETCH.validated_url_file(
                    Path(temp), Path("experiments/model/../../secrets")
                )

    def test_enforces_configured_bucket(self):
        with tempfile.TemporaryDirectory() as temp:
            url = Path(temp) / "weights.url"
            url.write_text("s3://unexpected/checkpoint.pt\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"CHECKPOINT_BUCKET": "models", "CHECKPOINT_PREFIX": "checkpoints"},
            ):
                with self.assertRaisesRegex(ValueError, "not allowed"):
                    FETCH.read_s3_uri(url)

    def test_enforces_experiment_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            url = Path(temp) / "weights.url"
            url.write_text("s3://models/checkpoints/other/model.pt\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"CHECKPOINT_BUCKET": "models", "CHECKPOINT_PREFIX": "checkpoints"},
            ):
                with self.assertRaisesRegex(ValueError, "must start"):
                    FETCH.read_s3_uri(url, "expected")


if __name__ == "__main__":
    unittest.main()
