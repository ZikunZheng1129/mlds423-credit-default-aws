"""Baseline model training utilities."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with recent and older sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns for sklearn preprocessing."""
    categorical_columns = [
        column
        for column in X.columns
        if pd.api.types.is_object_dtype(X[column])
        or isinstance(X[column].dtype, pd.CategoricalDtype)
    ]
    numeric_columns = [column for column in X.columns if column not in categorical_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Build a ColumnTransformer fitted later inside a sklearn Pipeline."""
    numeric_columns, categorical_columns = infer_feature_types(X)
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logistic_regression_pipeline(
    X: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create the baseline Logistic Regression pipeline."""
    model_params = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": 7,
    }
    model_params.update(params or {})
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, scale_numeric=True)),
            ("model", LogisticRegression(**model_params)),
        ]
    )


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> tuple[Pipeline, float]:
    """Fit the Logistic Regression baseline and return model plus train seconds."""
    model = build_logistic_regression_pipeline(X_train, params)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start
