"""Configuration loading for the credit default pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

REQUIRED_SECTIONS = {
    "project",
    "aws",
    "data",
    "features",
    "models",
    "evaluation",
    "artifacts",
    "api",
    "logging",
}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply supported environment variable overrides to a config mapping."""
    cfg = {**config}
    cfg["aws"] = {**cfg.get("aws", {})}
    cfg["data"] = {**cfg.get("data", {})}
    cfg["artifacts"] = {**cfg.get("artifacts", {})}
    cfg["api"] = {**cfg.get("api", {})}
    cfg["logging"] = {**cfg.get("logging", {})}

    region = os.getenv("AWS_REGION")
    if region:
        cfg["aws"]["region"] = region

    bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
    if bucket:
        cfg["aws"]["s3_bucket"] = bucket

    raw_uri = os.getenv("RAW_DATA_URI")
    if raw_uri:
        cfg["data"]["raw_uri"] = raw_uri

    processed_uri = os.getenv("PROCESSED_DATA_URI")
    if processed_uri:
        cfg["data"]["processed_uri"] = processed_uri

    model_uri = os.getenv("MODEL_ARTIFACT_URI")
    if model_uri:
        cfg["artifacts"]["model_artifact_uri"] = model_uri
        cfg["artifacts"]["best_model_path"] = model_uri
        cfg["api"]["model_path"] = model_uri

    log_level = os.getenv("CREDIT_DEFAULT_LOG_LEVEL")
    if log_level:
        cfg["logging"]["level"] = log_level

    return cfg


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    """Validate that the config has the small set of required sections."""
    missing = sorted(REQUIRED_SECTIONS.difference(config))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Config {config_path} is missing required sections: {joined}")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML config and apply supported environment variable overrides.

    AWS credentials are intentionally not loaded from config files. boto3 should
    use its default credential chain: environment variables, AWS CLI profile,
    ECS/EC2 IAM role, or another supported provider.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    if not isinstance(config, dict):
        raise ValueError(f"Config {path} did not parse to a mapping.")

    _validate_config(config, path)
    return _apply_env_overrides(config)

