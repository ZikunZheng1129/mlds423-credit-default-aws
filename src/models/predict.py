"""Prediction helpers for trained model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model_artifact(model_path: str | Path) -> Any:
    """Load a joblib model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def load_model_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load model metadata JSON when it exists."""
    import json

    path = Path(metadata_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    return metadata if isinstance(metadata, dict) else {}


def predict_proba(model: Any, records: pd.DataFrame) -> pd.Series:
    """Return positive-class probabilities for one or more records."""
    probabilities = model.predict_proba(records)[:, 1]
    return pd.Series(probabilities, index=records.index, name="probability")


def predict_classes(
    model: Any,
    records: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return probabilities and thresholded binary predictions."""
    probabilities = predict_proba(model, records)
    return pd.DataFrame(
        {
            "predicted_probability": probabilities,
            "predicted_class": (probabilities >= threshold).astype(int),
        },
        index=records.index,
    )
