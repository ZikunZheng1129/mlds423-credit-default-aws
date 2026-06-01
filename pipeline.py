"""End-to-end local/AWS-ready training pipeline for credit default prediction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.feature_engineering import add_credit_features
from src.data.ingest import load_raw_data
from src.data.preprocess import clean_credit_data, prepare_dataset, save_processed_data
from src.logging_utils import get_logger, setup_logging
from src.cloud.s3_utils import is_s3_uri, upload_to_s3
from src.models.evaluate import evaluate_model, tune_threshold_with_cv
from src.models.train_advanced import (
    XGBoostUnavailableError,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
)
from src.models.train_baseline import build_logistic_regression_pipeline

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train credit default models.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    artifact_cfg = cfg["artifacts"]
    for key in ["base_dir", "models_dir", "metadata_dir", "metrics_dir"]:
        Path(artifact_cfg[key]).mkdir(parents=True, exist_ok=True)


def _model_specs(cfg: dict[str, Any], X_train: pd.DataFrame) -> list[tuple[str, Any]]:
    random_state = int(cfg["training"].get("random_state", cfg["project"].get("random_state", 7)))
    model_cfg = cfg["models"]

    specs: list[tuple[str, Any]] = [
        (
            model_cfg["baseline"]["name"],
            build_logistic_regression_pipeline(
                X_train,
                {**model_cfg["baseline"].get("params", {}), "random_state": random_state},
            ),
        ),
        (
            model_cfg["random_forest"]["name"],
            build_random_forest_pipeline(
                X_train,
                {**model_cfg["random_forest"].get("params", {}), "random_state": random_state},
            ),
        ),
    ]

    try:
        specs.append(
            (
                model_cfg["xgboost"]["name"],
                build_xgboost_pipeline(
                    X_train,
                    {**model_cfg["xgboost"].get("params", {}), "random_state": random_state},
                ),
            )
        )
    except XGBoostUnavailableError as exc:
        logger.warning(
            "XGBoost skipped: %s. The pipeline will continue with available "
            "models. If you are on macOS and see libomp.dylib/OpenMP errors, "
            "run `brew install libomp`.",
            exc,
        )

    return specs


def _save_model_outputs(
    model_name: str,
    model: Any,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[Path, Path]:
    models_dir = Path(cfg["artifacts"]["models_dir"])
    metadata_dir = Path(cfg["artifacts"]["metadata_dir"])
    model_path = models_dir / f"{model_name}.joblib"
    metadata_path = metadata_dir / f"{model_name}_metadata.json"

    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    return model_path, metadata_path


def _select_best_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda item: (
            item["metrics"]["roc_auc"],
            item["metrics"]["f1"],
        ),
    )


def _write_reports(results: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    rows = []
    confusion_matrices = {}
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "model_name": result["model_name"],
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "selected_threshold": result["threshold"],
            }
        )
        confusion_matrices[result["model_name"]] = result["confusion_matrix"]

    metrics_path = Path(cfg["artifacts"]["metrics_summary_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["roc_auc", "f1"],
        ascending=False,
    ).to_csv(metrics_path, index=False)

    confusion_path = Path(cfg["artifacts"]["confusion_matrices_path"])
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_path.write_text(
        json.dumps(confusion_matrices, indent=2),
        encoding="utf-8",
    )


def run_pipeline(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Run data preparation, model training, evaluation, and artifact saving."""
    cfg = load_config(config_path)
    setup_logging(cfg["logging"].get("level", "INFO"))
    _ensure_output_dirs(cfg)

    training_cfg = cfg["training"]
    random_state = int(training_cfg.get("random_state", cfg["project"].get("random_state", 7)))
    cv_folds = int(cfg["evaluation"].get("cv_folds", 5))

    raw_df = load_raw_data(cfg["data"]["raw_uri"])
    clean_df = clean_credit_data(raw_df)
    featured_df = (
        add_credit_features(clean_df)
        if cfg["features"].get("use_engineered_features", True)
        else clean_df
    )
    save_processed_data(featured_df, cfg["data"]["processed_uri"])

    X, y = prepare_dataset(featured_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(training_cfg.get("test_size", 0.2)),
        stratify=y if training_cfg.get("stratify", True) else None,
        random_state=random_state,
    )

    results = []
    for model_name, estimator in _model_specs(cfg, X_train):
        logger.info("Training model %s", model_name)
        threshold, threshold_f1 = (
            tune_threshold_with_cv(estimator, X_train, y_train, cv_folds, random_state)
            if training_cfg.get("threshold_tuning_enabled", True)
            else (0.5, 0.0)
        )

        estimator.fit(X_train, y_train)
        metrics, confusion = evaluate_model(estimator, X_test, y_test, threshold)
        metrics["threshold_cv_f1"] = float(threshold_f1)

        metadata = {
            "model_name": model_name,
            "feature_columns": list(X.columns),
            "threshold": threshold,
            "metrics": metrics,
            "confusion_matrix": confusion,
        }
        model_path, metadata_path = _save_model_outputs(model_name, estimator, metadata, cfg)
        results.append(
            {
                "model_name": model_name,
                "model": estimator,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "threshold": threshold,
                "metrics": metrics,
                "confusion_matrix": confusion,
                "metadata": metadata,
            }
        )

    best = _select_best_model(results)
    best_model_path = Path(cfg["artifacts"]["best_model_path"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best["model_path"], best_model_path)

    model_artifact_uri = cfg["artifacts"].get("model_artifact_uri")
    if model_artifact_uri and is_s3_uri(str(model_artifact_uri)):
        upload_to_s3(best_model_path, str(model_artifact_uri))

    best_metadata_path = Path(cfg["artifacts"]["base_dir"]) / "best_model_metadata.json"
    best_metadata = {**best["metadata"], "source_model_path": best["model_path"]}
    best_metadata_path.write_text(
        json.dumps(_json_safe(best_metadata), indent=2),
        encoding="utf-8",
    )

    _write_reports(results, cfg)
    logger.info("Best model selected: %s", best["model_name"])
    return {
        "best_model_name": best["model_name"],
        "best_model_path": str(best_model_path),
        "metrics_summary_path": cfg["artifacts"]["metrics_summary_path"],
        "results": [
            {key: value for key, value in result.items() if key != "model"}
            for result in results
        ],
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
