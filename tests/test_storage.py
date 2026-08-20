import tempfile
import unittest
from pathlib import Path

from model_bench.storage import download_weights, read_weights_uri


class StorageTest(unittest.TestCase):
    def test_requires_exactly_one_uri(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.url"
            path.write_text("first.pt\nsecond.pt\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly one"):
                read_weights_uri(path)

    def test_copies_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.pt"
            source.write_bytes(b"checkpoint")
            url_file = root / "weights.url"
            url_file.write_text("model.pt\n", encoding="utf-8")
            target = root / "download" / "checkpoint.pt"

            result = download_weights(url_file, target)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"checkpoint")


if __name__ == "__main__":
    unittest.main()
