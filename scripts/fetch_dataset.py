#!/usr/bin/env python3
"""Download an allowlisted validation subset without importing PR code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

EXPERIMENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_MAX_DATASET_BYTES = 10 * 1024**3
DEFAULT_MAX_DATASET_OBJECTS = 20_000
MAX_CONFIG_BYTES = 64 * 1024
SUPPORTED_DATASET_FORMATS = {"cityscapes_segmentation_npz_v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validated_config_file(source_root: Path, experiment_dir: Path) -> Path:
    root = source_root.resolve()
    if (
        experiment_dir.is_absolute()
        or len(experiment_dir.parts) != 2
        or experiment_dir.parts[0] != "experiments"
        or not EXPERIMENT_NAME.fullmatch(experiment_dir.parts[1])
    ):
        raise ValueError("experiment must be experiments/<snake_case_name>")
    directory = root / experiment_dir
    config = directory / "dataset.json"
    for path in (directory, config):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed: {path}")
    if not config.is_file() or config.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("missing or oversized dataset.json")
    return config


def load_config(path: Path) -> tuple[str, str, list[str], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"uri", "include", "format"}:
        raise ValueError("dataset.json must contain exactly uri, include, and format")
    uri = value["uri"]
    includes = value["include"]
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError("dataset uri must be an s3:// URI")
    bucket, separator, prefix = uri[5:].partition("/")
    if not bucket or not separator or not prefix or prefix.startswith("/"):
        raise ValueError("dataset uri must include a bucket and prefix")
    allowed_bucket = os.environ.get("DATASET_BUCKET")
    if not allowed_bucket:
        raise RuntimeError("DATASET_BUCKET is required")
    if bucket != allowed_bucket:
        raise ValueError(f"dataset bucket {bucket!r} is not allowed")
    normalized_prefix = prefix.strip("/")
    if _unsafe_relative_path(normalized_prefix):
        raise ValueError("dataset URI contains an unsafe prefix")
    allowed_prefix = os.environ.get("DATASET_PREFIX", "").strip("/")
    if allowed_prefix and not (
        normalized_prefix == allowed_prefix
        or normalized_prefix.startswith(f"{allowed_prefix}/")
    ):
        raise ValueError(f"dataset prefix must start with {allowed_prefix!r}")
    if (
        not isinstance(includes, list)
        or not 1 <= len(includes) <= 16
        or any(not isinstance(item, str) for item in includes)
    ):
        raise ValueError("dataset include must contain 1..16 prefixes")
    normalized_includes = []
    for item in includes:
        normalized = item.strip("/")
        if not normalized or _unsafe_relative_path(normalized):
            raise ValueError(f"unsafe dataset include prefix: {item!r}")
        normalized_includes.append(normalized)
    if len(set(normalized_includes)) != len(normalized_includes):
        raise ValueError("dataset include prefixes must be unique")
    dataset_format = value["format"]
    if dataset_format not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(f"unsupported dataset format: {dataset_format!r}")
    return bucket, normalized_prefix, normalized_includes, dataset_format


def _unsafe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)


def list_objects(
    client: Any, bucket: str, root_prefix: str, includes: list[str]
) -> list[dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for include in includes:
        object_prefix = f"{root_prefix}/{include}/"
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=object_prefix):
            for item in page.get("Contents", []):
                key = item.get("Key")
                size = item.get("Size")
                if (
                    not isinstance(key, str)
                    or not key.startswith(object_prefix)
                    or type(size) is not int
                    or size < 0
                ):
                    raise ValueError("S3 returned invalid dataset object metadata")
                relative = key.removeprefix(f"{root_prefix}/")
                if key.endswith("/") and size == 0:
                    continue
                if _unsafe_relative_path(relative):
                    raise ValueError(f"unsafe dataset object key: {key!r}")
                objects[key] = {
                    "key": key,
                    "relative": relative,
                    "size": size,
                    "etag": str(item.get("ETag", "")).strip('"'),
                }
                found = True
        if not found:
            raise ValueError(f"dataset prefix has no objects: {object_prefix}")
    selected = [objects[key] for key in sorted(objects)]
    maximum_objects = int(
        os.environ.get("MAX_DATASET_OBJECTS", DEFAULT_MAX_DATASET_OBJECTS)
    )
    maximum_bytes = int(os.environ.get("MAX_DATASET_BYTES", DEFAULT_MAX_DATASET_BYTES))
    total_bytes = sum(item["size"] for item in selected)
    if not selected or len(selected) > maximum_objects:
        raise ValueError("dataset object count is outside the allowed range")
    if total_bytes <= 0 or total_bytes > maximum_bytes:
        raise ValueError("dataset byte size is outside the allowed range")
    return selected


def download_dataset(
    client: Any,
    bucket: str,
    root_prefix: str,
    includes: list[str],
    dataset_format: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"dataset output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    objects = list_objects(client, bucket, root_prefix, includes)
    listing_hash = hashlib.sha256()
    for item in objects:
        destination = output.joinpath(*PurePosixPath(item["relative"]).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            client.download_file(bucket, item["key"], str(temporary))
            if temporary.stat().st_size != item["size"]:
                raise ValueError(f"downloaded size mismatch for {item['key']}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        listing_hash.update(f"{item['key']}\0{item['size']}\0{item['etag']}\n".encode())
    return {
        "uri": f"s3://{bucket}/{root_prefix}",
        "include": includes,
        "format": dataset_format,
        "object_count": len(objects),
        "total_bytes": sum(item["size"] for item in objects),
        "listing_sha256": listing_hash.hexdigest(),
    }


def main() -> None:
    args = parse_args()
    config = validated_config_file(args.source_root, args.experiment_dir)
    bucket, prefix, includes, dataset_format = load_config(config)
    import boto3

    endpoint = os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    client = boto3.client("s3", endpoint_url=endpoint)
    manifest = download_dataset(
        client, bucket, prefix, includes, dataset_format, args.output
    )
    (args.output / "_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"downloaded {manifest['object_count']} dataset objects "
        f"({manifest['total_bytes']} bytes)"
    )


if __name__ == "__main__":
    main()
