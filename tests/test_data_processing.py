"""Tests for ingestion, preprocessing, and feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.feature_engineering import add_credit_features
from src.data.ingest import RAW_REQUIRED_COLUMNS, load_raw_data, validate_raw_credit_data
from src.data.preprocess import clean_credit_data, prepare_dataset, save_processed_data


def _raw_credit_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "LIMIT_BAL": [20000, 120000, 90000],
            "SEX": [1, 2, 9],
            "EDUCATION": [1, 5, 99],
            "MARRIAGE": [1, 0, 9],
            "AGE": [24, 35, 42],
            "PAY_0": [2, -1, 0],
            "PAY_2": [2, 0, 0],
            "PAY_3": [0, 0, -1],
            "PAY_4": [0, 0, 0],
            "PAY_5": [0, -1, 0],
            "PAY_6": [0, -2, 0],
            "BILL_AMT1": [3913, 2682, 10000],
            "BILL_AMT2": [3102, 1725, 9000],
            "BILL_AMT3": [689, 2682, 8000],
            "BILL_AMT4": [0, 3272, 7000],
            "BILL_AMT5": [0, 3455, 6000],
            "BILL_AMT6": [0, 3261, 5000],
            "PAY_AMT1": [0, 1000, 1200],
            "PAY_AMT2": [689, 1000, 1100],
            "PAY_AMT3": [0, 1000, 1000],
            "PAY_AMT4": [0, 1000, 900],
            "PAY_AMT5": [0, 1000, 800],
            "PAY_AMT6": [0, 2000, 700],
            "default.payment.next.month": [1, 0, 0],
        }
    )


def test_clean_credit_data_drops_id_and_renames_target() -> None:
    cleaned = clean_credit_data(_raw_credit_df())

    assert "ID" not in cleaned.columns
    assert "default.payment.next.month" not in cleaned.columns
    assert cleaned["default"].tolist() == [1, 0, 0]


def test_clean_credit_data_recodes_categories_safely() -> None:
    cleaned = clean_credit_data(_raw_credit_df())

    assert cleaned["SEX"].astype(str).tolist() == ["male", "female", "unknown"]
    assert cleaned["EDUCATION"].astype(str).tolist() == [
        "graduate_school",
        "other_unknown",
        "other_unknown",
    ]
    assert cleaned["MARRIAGE"].astype(str).tolist() == ["married", "other", "other"]


def test_add_credit_features_creates_expected_columns() -> None:
    featured = add_credit_features(clean_credit_data(_raw_credit_df()))

    expected_columns = {
        "DELINQ_persistence",
        "DELINQ_max",
        "DELINQ_trend_recent",
        "UTIL_recent",
        "PAY_RATIO_recent",
        "PAY_INTENSITY_total",
        "BILL_VOLATILITY",
        "PAY_VOLATILITY",
        "DELINQ_x_UTIL",
    }
    assert expected_columns.issubset(featured.columns)
    assert featured["DELINQ_persistence"].tolist() == [2, 0, 0]


def test_prepare_dataset_splits_target_and_features() -> None:
    cleaned = clean_credit_data(_raw_credit_df())

    X, y = prepare_dataset(cleaned)

    assert "default" not in X.columns
    assert y.tolist() == [1, 0, 0]


def test_validate_raw_credit_data_missing_required_columns_raises() -> None:
    raw = _raw_credit_df().drop(columns=["PAY_0"])

    with pytest.raises(ValueError, match="missing required columns: PAY_0"):
        validate_raw_credit_data(raw)


def test_load_raw_data_reads_local_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "credit.csv"
    _raw_credit_df().to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path)

    assert list(loaded.columns) == RAW_REQUIRED_COLUMNS
    assert loaded.shape == (3, len(RAW_REQUIRED_COLUMNS))


def test_save_processed_data_local_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "processed" / "credit.csv"
    cleaned = clean_credit_data(_raw_credit_df())

    result = save_processed_data(cleaned, output_path)

    assert result == str(output_path)
    saved = pd.read_csv(output_path)
    assert saved.shape == cleaned.shape

