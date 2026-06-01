"""Model evaluation and threshold tuning utilities."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def tune_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select the threshold that maximizes minority-class F1."""
    threshold_grid = thresholds if thresholds is not None else np.arange(0.05, 0.951, 0.01)
    scores = [
        f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        for threshold in threshold_grid
    ]
    best_index = int(np.argmax(scores))
    return float(threshold_grid[best_index]), float(scores[best_index])


def tune_threshold_with_cv(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> tuple[float, float]:
    """Tune threshold from out-of-fold predicted probabilities."""
    min_class_count = int(y_train.value_counts().min())
    folds = min(cv_folds, min_class_count)
    if folds < 2:
        return 0.5, 0.0

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]
    return tune_threshold(y_train, probabilities)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict[str, float | int], dict[str, int]]:
    """Evaluate a fitted model at a selected decision threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    metrics: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "inference_rows": int(len(X_test)),
    }
    confusion = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics, confusion

