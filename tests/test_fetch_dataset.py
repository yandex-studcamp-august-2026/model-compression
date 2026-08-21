import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "fetch_dataset", Path("scripts/fetch_dataset.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load fetch_dataset.py")
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class FakePaginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, *, Bucket, Prefix):
        del Bucket
        return [
            {
                "Contents": [
                    item for item in self.objects if item["Key"].startswith(Prefix)
                ]
            }
        ]


class FakeClient:
    def __init__(self, objects, payloads):
        self.objects = objects
        self.payloads = payloads

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise AssertionError(name)
        return FakePaginator(self.objects)

    def download_file(self, bucket, key, destination):
        del bucket
        Path(destination).write_bytes(self.payloads[key])


class DatasetFetchTest(unittest.TestCase):
    def test_validates_bucket_prefix_and_includes(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "dataset.json"
            config.write_text(
                json.dumps(
                    {
                        "uri": "s3://datasets/cityscapes/Cityspaces",
                        "include": ["images/val", "gtFine/val"],
                        "format": "cityscapes_segmentation_npz_v1",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"DATASET_BUCKET": "datasets", "DATASET_PREFIX": "cityscapes"},
            ):
                self.assertEqual(
                    FETCH.load_config(config),
                    (
                        "datasets",
                        "cityscapes/Cityspaces",
                        ["images/val", "gtFine/val"],
                        "cityscapes_segmentation_npz_v1",
                    ),
                )

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "dataset.json"
            config.write_text(
                json.dumps(
                    {
                        "uri": "s3://datasets/cityscapes",
                        "include": ["../private"],
                        "format": "cityscapes_segmentation_npz_v1",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DATASET_BUCKET": "datasets"}):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    FETCH.load_config(config)

    def test_downloads_only_selected_prefixes_and_records_provenance(self):
        payloads = {
            "city/images/val/a.png": b"image",
            "city/gtFine/val/a.png": b"mask",
        }
        objects = [
            {"Key": key, "Size": len(value), "ETag": f'"etag-{index}"'}
            for index, (key, value) in enumerate(payloads.items())
        ]
        client = FakeClient(objects, payloads)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dataset"
            manifest = FETCH.download_dataset(
                client,
                "datasets",
                "city",
                ["images/val", "gtFine/val"],
                "cityscapes_segmentation_npz_v1",
                output,
            )

            self.assertEqual((output / "images/val/a.png").read_bytes(), b"image")
            self.assertEqual((output / "gtFine/val/a.png").read_bytes(), b"mask")
            self.assertEqual(manifest["object_count"], 2)
            self.assertEqual(manifest["total_bytes"], 9)
            self.assertEqual(manifest["format"], "cityscapes_segmentation_npz_v1")
            self.assertRegex(manifest["listing_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
