"""Tests for S3/local path helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.cloud.s3_utils import (
    ensure_local_path,
    is_s3_uri,
    parse_s3_uri,
    upload_to_s3,
)


def test_is_s3_uri_detects_s3_and_local_paths() -> None:
    assert is_s3_uri("s3://bucket/path/file.csv")
    assert not is_s3_uri("data/raw/file.csv")
    assert not is_s3_uri("/tmp/file.csv")


def test_parse_s3_uri_returns_bucket_and_key() -> None:
    bucket, key = parse_s3_uri("s3://my-bucket/folder/file.csv")

    assert bucket == "my-bucket"
    assert key == "folder/file.csv"


def test_parse_s3_uri_rejects_invalid_uri() -> None:
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        parse_s3_uri("s3://my-bucket")


def test_ensure_local_path_returns_local_path_without_aws_call() -> None:
    assert ensure_local_path("data/raw/file.csv") == Path("data/raw/file.csv")


def test_upload_to_s3_uses_boto3_client(tmp_path: Path) -> None:
    local_file = tmp_path / "model.joblib"
    local_file.write_text("placeholder", encoding="utf-8")
    fake_client = Mock()

    with patch("boto3.client", return_value=fake_client) as boto_client:
        result = upload_to_s3(local_file, "s3://bucket/models/model.joblib")

    assert result == "s3://bucket/models/model.joblib"
    boto_client.assert_called_once_with("s3")
    fake_client.upload_file.assert_called_once_with(
        str(local_file),
        "bucket",
        "models/model.joblib",
    )

