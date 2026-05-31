"""Data ingestion helpers for the credit default dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cloud.s3_utils import ensure_local_path, is_s3_uri
from src.logging_utils import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "default.payment.next.month"
RAW_REQUIRED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    TARGET_COLUMN,
]


def validate_raw_credit_data(df: pd.DataFrame) -> None:
    """Validate that the raw credit default dataset has expected columns."""
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Raw credit data is missing required columns: {joined}")


def load_raw_data(uri: str | Path) -> pd.DataFrame:
    """Load raw credit default CSV data from a local path or S3 URI."""
    uri_text = str(uri)
    local_path = ensure_local_path(uri_text) if is_s3_uri(uri_text) else Path(uri_text)

    if not local_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {local_path}. Place the CSV in data/raw/ "
            "or set RAW_DATA_URI to a valid local path or S3 URI."
        )
    if not local_path.is_file():
        raise ValueError(f"Raw data path is not a file: {local_path}")

    logger.info("Loading raw credit data from %s", local_path)
    df = pd.read_csv(local_path)
    validate_raw_credit_data(df)
    logger.info("Loaded raw credit data with shape rows=%s columns=%s", *df.shape)
    return df

