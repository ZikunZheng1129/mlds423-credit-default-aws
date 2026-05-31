"""Tests for configuration loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.config import load_config


def _write_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            project: {name: test, version: 0.1.0}
            aws: {region: us-east-1, s3_bucket: null}
            data:
              raw_uri: data/raw/test.csv
              processed_uri: data/processed/test.csv
            features: {}
            models: {}
            evaluation: {}
            artifacts:
              model_artifact_uri: artifacts/models/best_model.joblib
              best_model_path: artifacts/models/best_model.joblib
            api:
              model_path: artifacts/models/best_model.joblib
            logging: {level: INFO}
            """
        ).strip(),
        encoding="utf-8",
    )


def test_load_config_reads_required_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    config = load_config(config_path)

    assert config["project"]["name"] == "test"
    assert config["aws"]["region"] == "us-east-1"


def test_load_config_applies_environment_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("S3_BUCKET", "example-bucket")
    monkeypatch.setenv("RAW_DATA_URI", "s3://example-bucket/raw.csv")
    monkeypatch.setenv("PROCESSED_DATA_URI", "s3://example-bucket/processed.csv")
    monkeypatch.setenv("MODEL_ARTIFACT_URI", "s3://example-bucket/model.joblib")

    config = load_config(config_path)

    assert config["aws"]["region"] == "us-west-2"
    assert config["aws"]["s3_bucket"] == "example-bucket"
    assert config["data"]["raw_uri"] == "s3://example-bucket/raw.csv"
    assert config["data"]["processed_uri"] == "s3://example-bucket/processed.csv"
    assert config["artifacts"]["model_artifact_uri"] == "s3://example-bucket/model.joblib"
    assert config["api"]["model_path"] == "s3://example-bucket/model.joblib"

