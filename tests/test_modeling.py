"""Tests for modeling, evaluation, prediction, and pipeline orchestration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

import pipeline
from pipeline import run_pipeline
from src.data.preprocess import clean_credit_data
from src.models.evaluate import evaluate_model, tune_threshold
from src.models.predict import predict_classes, predict_proba
from src.models.train_advanced import XGBoostUnavailableError


def _raw_credit_df(rows: int = 24) -> pd.DataFrame:
    data = []
    for index in range(rows):
        default = int(index % 4 == 0)
        delinquency = 2 if default else 0
        data.append(
            {
                "ID": index + 1,
                "LIMIT_BAL": 20000 + index * 1000,
                "SEX": 1 if index % 2 else 2,
                "EDUCATION": (index % 4) + 1,
                "MARRIAGE": (index % 3) + 1,
                "AGE": 24 + index,
                "PAY_0": delinquency,
                "PAY_2": delinquency,
                "PAY_3": 1 if default else 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 3000 + index * 10,
                "BILL_AMT2": 2800 + index * 10,
                "BILL_AMT3": 2600 + index * 10,
                "BILL_AMT4": 2400 + index * 10,
                "BILL_AMT5": 2200 + index * 10,
                "BILL_AMT6": 2000 + index * 10,
                "PAY_AMT1": 100 if default else 900,
                "PAY_AMT2": 100 if default else 850,
                "PAY_AMT3": 100 if default else 800,
                "PAY_AMT4": 100 if default else 750,
                "PAY_AMT5": 100 if default else 700,
                "PAY_AMT6": 100 if default else 650,
                "default.payment.next.month": default,
            }
        )
    return pd.DataFrame(data)


def test_tune_threshold_selects_best_f1_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])

    threshold, score = tune_threshold(y_true, y_prob, np.array([0.3, 0.5]))

    assert threshold == 0.3
    assert score == 0.8


def test_evaluate_model_computes_expected_metrics() -> None:
    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(X, y)

    metrics, confusion = evaluate_model(model, X, y, threshold=0.5)

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert confusion == {
        "true_negative": 0,
        "false_positive": 2,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_prediction_helpers_return_expected_shape() -> None:
    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    model = DummyClassifier(strategy="prior")
    model.fit(X, y)

    probabilities = predict_proba(model, X)
    predictions = predict_classes(model, X, threshold=0.5)

    assert probabilities.shape == (4,)
    assert list(predictions.columns) == ["predicted_probability", "predicted_class"]
    assert predictions.shape == (4, 2)


def test_run_pipeline_with_small_synthetic_dataset(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    _raw_credit_df().to_csv(raw_path, index=False)

    config_path.write_text(
        textwrap.dedent(
            f"""
            project:
              name: test-credit-default
              version: 0.1.0
              random_state: 7
            aws:
              region: us-east-2
              s3_bucket: null
              upload_enabled: false
              s3_prefix: test
            data:
              raw_uri: {raw_path}
              processed_uri: {processed_path}
              target_column: default.payment.next.month
              id_column: ID
            training:
              test_size: 0.25
              random_state: 7
              stratify: true
              threshold_tuning_enabled: true
            features:
              use_engineered_features: true
              categorical_columns: [SEX, EDUCATION, MARRIAGE]
              scale_numeric_for_linear_models: true
            models:
              baseline:
                name: logistic_regression
                params:
                  max_iter: 200
                  class_weight: balanced
              random_forest:
                name: random_forest
                params:
                  n_estimators: 5
                  min_samples_leaf: 1
                  max_features: sqrt
                  class_weight: balanced
                  n_jobs: 1
              xgboost:
                name: xgboost
                params:
                  n_estimators: 5
                  learning_rate: 0.1
                  max_depth: 2
                  subsample: 1.0
                  colsample_bytree: 1.0
                  eval_metric: auc
                  n_jobs: 1
            evaluation:
              cv_folds: 2
              threshold_metric: f1
              positive_class: 1
              best_model_primary_metric: roc_auc
              best_model_tie_breaker: f1
            artifacts:
              base_dir: {artifacts_dir}
              models_dir: {artifacts_dir / "models"}
              metadata_dir: {artifacts_dir / "metadata"}
              metrics_dir: {reports_dir}
              processed_data_dir: {tmp_path}
              model_artifact_uri: {artifacts_dir / "best_model.joblib"}
              best_model_path: {artifacts_dir / "best_model.joblib"}
              metrics_summary_path: {reports_dir / "metrics_summary.csv"}
              confusion_matrices_path: {reports_dir / "confusion_matrices.json"}
            api:
              host: 0.0.0.0
              port: 8000
              model_path: {artifacts_dir / "best_model.joblib"}
              default_threshold: 0.5
            logging:
              level: WARNING
            """
        ).strip(),
        encoding="utf-8",
    )

    result = run_pipeline(config_path)

    assert Path(result["best_model_path"]).exists()
    assert processed_path.exists()
    assert (reports_dir / "metrics_summary.csv").exists()
    assert len(result["results"]) >= 2


def test_model_specs_skip_xgboost_when_construction_fails(
    monkeypatch,
    caplog,
) -> None:
    X = clean_credit_data(_raw_credit_df()).drop(columns=["default"])
    cfg = {
        "project": {"random_state": 7},
        "training": {"random_state": 7},
        "models": {
            "baseline": {
                "name": "logistic_regression",
                "params": {"max_iter": 100},
            },
            "random_forest": {
                "name": "random_forest",
                "params": {"n_estimators": 5, "n_jobs": 1},
            },
            "xgboost": {
                "name": "xgboost",
                "params": {"n_estimators": 5},
            },
        },
    }

    def fail_xgboost(*_args, **_kwargs):
        raise XGBoostUnavailableError("Library not loaded: @rpath/libomp.dylib")

    monkeypatch.setattr(pipeline, "build_xgboost_pipeline", fail_xgboost)

    specs = pipeline._model_specs(cfg, X)  # pylint: disable=protected-access

    model_names = [name for name, _estimator in specs]
    assert model_names == ["logistic_regression", "random_forest"]
    assert "XGBoost skipped" in caplog.text
    assert "brew install libomp" in caplog.text


def test_model_specs_injects_random_state_into_baseline() -> None:
    X = clean_credit_data(_raw_credit_df()).drop(columns=["default"])
    cfg = {
        "project": {"random_state": 7},
        "training": {"random_state": 42},
        "models": {
            "baseline": {
                "name": "logistic_regression",
                "params": {"max_iter": 100},
            },
            "random_forest": {
                "name": "random_forest",
                "params": {"n_estimators": 5, "n_jobs": 1},
            },
            "xgboost": {
                "name": "xgboost",
                "params": {"n_estimators": 5},
            },
        },
    }

    specs = pipeline._model_specs(cfg, X)
    baseline_model = specs[0][1].named_steps["model"]

    assert baseline_model.get_params()["random_state"] == 42
