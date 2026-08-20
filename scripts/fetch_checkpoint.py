#!/usr/bin/env python3
"""Download a checkpoint without importing pull-request code."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

EXPERIMENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_MAX_CHECKPOINT_BYTES = 5 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validated_url_file(source_root: Path, experiment_dir: Path) -> Path:
    root = source_root.resolve()
    if (
        experiment_dir.is_absolute()
        or len(experiment_dir.parts) != 2
        or experiment_dir.parts[0] != "experiments"
        or not EXPERIMENT_NAME.fullmatch(experiment_dir.parts[1])
    ):
        raise ValueError("experiment must be experiments/<snake_case_name>")
    directory = root / experiment_dir
    for path in (directory, directory / "weights.url", directory / "weights.sha256"):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed: {path}")
    markers = [
        path for path in (directory / "CPU", directory / "GPU") if path.is_file()
    ]
    if len(markers) != 1 or markers[0].read_bytes():
        raise ValueError("candidate must contain exactly one empty CPU or GPU marker")
    url_file = directory / "weights.url"
    if not url_file.is_file():
        raise ValueError(f"missing {url_file}")
    read_expected_sha256(directory / "weights.sha256")
    return url_file


def read_expected_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or invalid {path}")
    digest = path.read_text(encoding="ascii").strip()
    if SHA256.fullmatch(digest) is None:
        raise ValueError("weights.sha256 must contain one lowercase SHA-256 digest")
    return digest


def validate_content_length(value: object, maximum: int) -> int:
    if type(value) is not int or value <= 0 or value > maximum:
        raise ValueError(
            f"checkpoint size {value!r} is outside the allowed range 1..{maximum}"
        )
    return value


def read_s3_uri(path: Path, experiment_name: str | None = None) -> tuple[str, str]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("s3://"):
        raise ValueError("weights.url must contain exactly one s3:// URI")
    bucket, separator, key = lines[0][5:].partition("/")
    if not separator or not bucket or not key or key.startswith("/"):
        raise ValueError("invalid S3 URI")
    allowed_bucket = os.environ.get("CHECKPOINT_BUCKET")
    if not allowed_bucket:
        raise RuntimeError("CHECKPOINT_BUCKET is required")
    if bucket != allowed_bucket:
        raise ValueError(f"checkpoint bucket {bucket!r} is not allowed")
    allowed_prefix = os.environ.get("CHECKPOINT_PREFIX")
    if not allowed_prefix:
        raise RuntimeError("CHECKPOINT_PREFIX is required")
    normalized_prefix = allowed_prefix.strip("/")
    if not normalized_prefix:
        raise ValueError("CHECKPOINT_PREFIX must not be empty")
    required_prefix = f"{normalized_prefix}/"
    if experiment_name is not None:
        required_prefix = f"{required_prefix}{experiment_name}/"
    if not key.startswith(required_prefix):
        raise ValueError(f"checkpoint key must start with {required_prefix!r}")
    return bucket, key


def main() -> None:
    args = parse_args()
    url_file = validated_url_file(args.source_root, args.experiment_dir)
    bucket, key = read_s3_uri(url_file, args.experiment_dir.parts[1])
    expected_digest = read_expected_sha256(url_file.with_name("weights.sha256"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        import boto3

        endpoint = os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
        client = boto3.client("s3", endpoint_url=endpoint)
        metadata = client.head_object(Bucket=bucket, Key=key)
        content_length = metadata.get("ContentLength")
        maximum = int(
            os.environ.get("MAX_CHECKPOINT_BYTES", DEFAULT_MAX_CHECKPOINT_BYTES)
        )
        validate_content_length(content_length, maximum)
        client.download_file(bucket, key, str(temporary))
        digest = sha256_path(temporary)
        if digest != expected_digest:
            raise ValueError(
                "downloaded checkpoint SHA-256 mismatch: "
                f"expected {expected_digest}, got {digest}"
            )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"downloaded {args.output.name} sha256={expected_digest}")


def sha256_path(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


if __name__ == "__main__":
    main()
