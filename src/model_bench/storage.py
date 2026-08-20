from __future__ import annotations

import os
import shutil
from pathlib import Path


def read_weights_uri(url_file: Path) -> str:
    lines = [
        line.strip()
        for line in url_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"{url_file} must contain exactly one non-empty URI")
    return lines[0]


def download_weights(url_file: Path, target: Path) -> Path:
    uri = read_weights_uri(url_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    temporary.unlink(missing_ok=True)

    try:
        if uri.startswith("s3://"):
            _download_s3(uri, temporary)
        elif "://" not in uri:
            _copy_local(uri, url_file, temporary)
        else:
            raise ValueError("weights.url supports only s3:// or local paths")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _download_s3(uri: str, target: Path) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required to download s3:// checkpoints") from exc
    bucket, separator, key = uri[5:].partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    endpoint = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    boto3.client("s3", endpoint_url=endpoint).download_file(bucket, key, str(target))


def _copy_local(uri: str, url_file: Path, target: Path) -> None:
    source = Path(uri)
    if not source.is_absolute():
        source = (url_file.parent / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    shutil.copy2(source, target)


def resolve_weights(url_file: Path, destination: Path) -> Path:
    uri = read_weights_uri(url_file)
    filename = Path(uri).name
    if not filename:
        raise ValueError(f"Checkpoint URI has no filename: {uri}")
    return download_weights(url_file, destination / filename)
