"""Interpretable credit-risk feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMOUNT_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
EPSILON = 1e-6


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Feature engineering requires missing columns: {joined}")


def _replace_non_finite(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    updated = df.copy()
    updated[columns] = updated[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    return updated


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clean behavioral features inspired by the prototype notebook."""
    required = ["LIMIT_BAL", *PAY_STATUS_COLUMNS, *BILL_AMOUNT_COLUMNS, *PAY_AMOUNT_COLUMNS]
    _require_columns(df, required)

    featured = df.copy()
    delinquency_positive = featured[PAY_STATUS_COLUMNS].clip(lower=0)

    featured["DELINQ_persistence"] = (delinquency_positive > 0).sum(axis=1)
    featured["DELINQ_max"] = delinquency_positive.max(axis=1)
    featured["DELINQ_mean"] = delinquency_positive.mean(axis=1)
    featured["DELINQ_trend_recent"] = (
        featured["PAY_0"] - featured[["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    )

    status_pairs = [
        ("PAY_0", "PAY_2"),
        ("PAY_2", "PAY_3"),
        ("PAY_3", "PAY_4"),
        ("PAY_4", "PAY_5"),
        ("PAY_5", "PAY_6"),
    ]
    status_diffs = pd.DataFrame(index=featured.index)
    for current, previous in status_pairs:
        status_diffs[f"{current}_minus_{previous}"] = featured[current] - featured[previous]
    featured["DELINQ_worsening_count"] = (status_diffs > 0).sum(axis=1)
    featured["DELINQ_improving_count"] = (status_diffs < 0).sum(axis=1)

    limit = featured["LIMIT_BAL"] + EPSILON
    featured["UTIL_recent"] = featured["BILL_AMT1"] / limit
    featured["UTIL_mean"] = featured[BILL_AMOUNT_COLUMNS].mean(axis=1) / limit
    featured["UTIL_max"] = featured[BILL_AMOUNT_COLUMNS].max(axis=1) / limit

    recent_bill = featured["BILL_AMT1"].abs() + EPSILON
    mean_bill = featured[BILL_AMOUNT_COLUMNS].mean(axis=1).abs() + EPSILON
    total_bill = featured[BILL_AMOUNT_COLUMNS].sum(axis=1).abs() + EPSILON
    featured["PAY_RATIO_recent"] = (featured["PAY_AMT1"] / recent_bill).clip(0, 5)
    featured["PAY_RATIO_mean"] = (
        featured[PAY_AMOUNT_COLUMNS].mean(axis=1) / mean_bill
    ).clip(0, 5)
    featured["PAY_RATIO_total"] = (
        featured[PAY_AMOUNT_COLUMNS].sum(axis=1) / total_bill
    ).clip(0, 5)

    featured["PAY_INTENSITY_recent"] = featured["PAY_AMT1"] / limit
    featured["PAY_INTENSITY_mean"] = featured[PAY_AMOUNT_COLUMNS].mean(axis=1) / limit
    featured["PAY_INTENSITY_total"] = featured[PAY_AMOUNT_COLUMNS].sum(axis=1) / limit

    featured["BILL_VOLATILITY"] = featured[BILL_AMOUNT_COLUMNS].std(axis=1)
    featured["PAY_VOLATILITY"] = featured[PAY_AMOUNT_COLUMNS].std(axis=1)

    featured["DELINQ_x_LIMIT_BAL"] = featured["DELINQ_max"] * featured["LIMIT_BAL"]
    featured["DELINQ_x_UTIL"] = featured["DELINQ_max"] * featured["UTIL_recent"]
    featured["DELINQ_x_PAY_RATIO"] = featured["DELINQ_max"] * featured["PAY_RATIO_recent"]

    engineered_columns = [
        "DELINQ_persistence",
        "DELINQ_max",
        "DELINQ_mean",
        "DELINQ_trend_recent",
        "DELINQ_worsening_count",
        "DELINQ_improving_count",
        "UTIL_recent",
        "UTIL_mean",
        "UTIL_max",
        "PAY_RATIO_recent",
        "PAY_RATIO_mean",
        "PAY_RATIO_total",
        "PAY_INTENSITY_recent",
        "PAY_INTENSITY_mean",
        "PAY_INTENSITY_total",
        "BILL_VOLATILITY",
        "PAY_VOLATILITY",
        "DELINQ_x_LIMIT_BAL",
        "DELINQ_x_UTIL",
        "DELINQ_x_PAY_RATIO",
    ]
    featured = _replace_non_finite(featured, engineered_columns)

    logger.info("Added %s engineered credit features", len(engineered_columns))
    return featured

