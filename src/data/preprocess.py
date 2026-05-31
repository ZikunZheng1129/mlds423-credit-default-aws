"""Cleaning and preprocessing helpers for credit default data."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.cloud.s3_utils import is_s3_uri, upload_to_s3
from src.data.ingest import TARGET_COLUMN
from src.logging_utils import get_logger

logger = get_logger(__name__)

CLEAN_TARGET_COLUMN = "default"


def _map_categories(
    series: pd.Series,
    mapping: dict[int, str],
    default_value: str,
) -> pd.Series:
    """Map numeric categories while keeping unexpected values explicit."""
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.map(mapping).fillna(default_value).astype("category")


def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw UCI credit default data using the notebook as reference."""
    if TARGET_COLUMN not in df.columns and CLEAN_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' or '{CLEAN_TARGET_COLUMN}'."
        )

    cleaned = df.copy()
    cleaned = cleaned.drop(columns=["ID"], errors="ignore")

    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.rename(columns={TARGET_COLUMN: CLEAN_TARGET_COLUMN})

    target = pd.to_numeric(cleaned[CLEAN_TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not set(target.dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only binary 0/1 values.")
    cleaned[CLEAN_TARGET_COLUMN] = target.astype(int)

    if "EDUCATION" in cleaned.columns:
        education = pd.to_numeric(cleaned["EDUCATION"], errors="coerce")
        education = education.replace({0: 4, 5: 4, 6: 4})
        cleaned["EDUCATION"] = _map_categories(
            education,
            {
                1: "graduate_school",
                2: "university",
                3: "high_school",
                4: "other_unknown",
            },
            "other_unknown",
        )

    if "MARRIAGE" in cleaned.columns:
        marriage = pd.to_numeric(cleaned["MARRIAGE"], errors="coerce")
        marriage = marriage.replace({0: 3})
        cleaned["MARRIAGE"] = _map_categories(
            marriage,
            {1: "married", 2: "single", 3: "other"},
            "other",
        )

    if "SEX" in cleaned.columns:
        cleaned["SEX"] = _map_categories(
            cleaned["SEX"],
            {1: "male", 2: "female"},
            "unknown",
        )

    logger.info("Cleaned credit data with shape rows=%s columns=%s", *cleaned.shape)
    return cleaned


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features and binary target."""
    if CLEAN_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Prepared data is missing target column '{CLEAN_TARGET_COLUMN}'.")

    y = pd.to_numeric(df[CLEAN_TARGET_COLUMN], errors="coerce")
    if y.isna().any():
        raise ValueError("Target column contains missing or non-numeric values.")

    X = df.drop(columns=[CLEAN_TARGET_COLUMN]).copy()
    return X, y.astype(int)


def save_processed_data(df: pd.DataFrame, output_uri: str | Path) -> str:
    """Save processed data to a local CSV path or upload it to S3."""
    output_text = str(output_uri)

    if is_s3_uri(output_text):
        with TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "processed.csv"
            df.to_csv(local_path, index=False)
            upload_to_s3(local_path, output_text)
        logger.info("Saved processed data to %s", output_text)
        return output_text

    output_path = Path(output_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)
    return str(output_path)

