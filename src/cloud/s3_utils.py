"""Helpers for local paths and S3 object movement."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from src.logging_utils import get_logger

logger = get_logger(__name__)


def is_s3_uri(uri: str) -> bool:
    """Return True when a URI starts with the S3 scheme."""
    return urlparse(uri).scheme == "s3"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 URI into bucket and key.

    Raises:
        ValueError: if the URI is not a valid object URI like
            ``s3://bucket/path/to/file.csv``.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_from_s3(s3_uri: str, local_path: str | Path) -> Path:
    """Download an S3 object to a local path using boto3's credential chain."""
    bucket, key = parse_s3_uri(s3_uri)
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    import boto3  # noqa: WPS433

    logger.info("Downloading %s to %s", s3_uri, destination)
    boto3.client("s3").download_file(bucket, key, str(destination))
    return destination


def upload_to_s3(local_path: str | Path, s3_uri: str) -> str:
    """Upload a local file to S3 using boto3's credential chain."""
    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(f"Local file not found: {source}")

    bucket, key = parse_s3_uri(s3_uri)

    import boto3  # noqa: WPS433

    logger.info("Uploading %s to %s", source, s3_uri)
    boto3.client("s3").upload_file(str(source), bucket, key)
    return s3_uri


def ensure_local_path(uri: str, local_cache_dir: str | Path = "data/raw") -> Path:
    """Return a local path for a local URI, or download S3 data into a cache."""
    if not is_s3_uri(uri):
        return Path(uri)

    _bucket, key = parse_s3_uri(uri)
    filename = Path(key).name
    if not filename:
        raise ValueError(f"S3 URI does not include a file name: {uri}")

    destination = Path(local_cache_dir) / filename
    return download_from_s3(uri, destination)

