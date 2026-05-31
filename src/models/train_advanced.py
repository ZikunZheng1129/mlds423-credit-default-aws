"""Advanced model training utilities."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.train_baseline import build_preprocessor


class XGBoostUnavailableError(RuntimeError):
    """Raised when XGBoost cannot be imported or initialized locally."""


def build_random_forest_pipeline(
    X: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create a Random Forest pipeline."""
    model_params = {
        "n_estimators": 300,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 7,
        "n_jobs": -1,
    }
    model_params.update(params or {})
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, scale_numeric=False)),
            ("model", RandomForestClassifier(**model_params)),
        ]
    )


def build_xgboost_pipeline(
    X: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create an XGBoost pipeline, raising a clear error if unavailable."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise XGBoostUnavailableError(
            "XGBoost could not be imported or initialized. On macOS this is "
            "often caused by missing OpenMP / libomp.dylib. Suggested local "
            "fix: `brew install libomp`. Original error: "
            f"{exc}"
        ) from exc

    model_params = {
        "objective": "binary:logistic",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "auc",
        "random_state": 7,
        "n_jobs": -1,
    }
    model_params.update(params or {})
    try:
        xgb_model = XGBClassifier(**model_params)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise XGBoostUnavailableError(
            "XGBoost could not be initialized. On macOS this may be caused by "
            "missing OpenMP / libomp.dylib. Suggested local fix: "
            "`brew install libomp`. Original error: "
            f"{exc}"
        ) from exc

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, scale_numeric=False)),
            ("model", xgb_model),
        ]
    )


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> tuple[Pipeline, float]:
    """Fit Random Forest and return model plus train seconds."""
    model = build_random_forest_pipeline(X_train, params)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> tuple[Pipeline, float]:
    """Fit XGBoost and return model plus train seconds."""
    model = build_xgboost_pipeline(X_train, params)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start
