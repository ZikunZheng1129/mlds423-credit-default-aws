"""API tests for the FastAPI inference service."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.data.feature_engineering import add_credit_features
from src.data.preprocess import clean_credit_data, prepare_dataset
from src.models.train_baseline import train_baseline_model


def _sample_record() -> dict[str, int | float]:
    return {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }


def _raw_training_df() -> pd.DataFrame:
    records = []
    for index in range(16):
        default = int(index % 4 == 0)
        row = _sample_record()
        row.update(
            {
                "ID": index + 1,
                "LIMIT_BAL": 20000 + index * 1000,
                "PAY_0": 2 if default else 0,
                "PAY_2": 2 if default else 0,
                "PAY_3": 1 if default else 0,
                "PAY_AMT1": 0 if default else 1000,
                "PAY_AMT2": 100 if default else 900,
                "default.payment.next.month": default,
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def _write_api_config(
    tmp_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project:
              name: api-test
              version: 0.1.0
              random_state: 7
            aws:
              region: us-east-2
              s3_bucket: null
              upload_enabled: false
              s3_prefix: test
            data:
              raw_uri: data/raw/default_of_credit_card_clients.csv
              processed_uri: data/processed/credit_default_processed.csv
              target_column: default.payment.next.month
              id_column: ID
            training:
              test_size: 0.2
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
                params: {{max_iter: 200, class_weight: balanced}}
              random_forest:
                name: random_forest
                params: {{}}
              xgboost:
                name: xgboost
                params: {{}}
            evaluation:
              cv_folds: 2
              threshold_metric: f1
              positive_class: 1
              best_model_primary_metric: roc_auc
              best_model_tie_breaker: f1
            artifacts:
              base_dir: {tmp_path / "artifacts"}
              models_dir: {tmp_path / "artifacts" / "models"}
              metadata_dir: {tmp_path / "artifacts" / "metadata"}
              metrics_dir: {tmp_path / "reports"}
              processed_data_dir: {tmp_path}
              model_artifact_uri: {model_path}
              best_model_path: {model_path}
              metrics_summary_path: {tmp_path / "reports" / "metrics_summary.csv"}
              confusion_matrices_path: {tmp_path / "reports" / "confusion_matrices.json"}
            api:
              host: 0.0.0.0
              port: 8000
              model_path: {model_path}
              metadata_path: {metadata_path}
              default_threshold: 0.5
            logging:
              level: WARNING
            """
        ).strip(),
        encoding="utf-8",
    )
    return config_path


def test_health_works_when_model_artifact_missing(tmp_path: Path) -> None:
    config_path = _write_api_config(
        tmp_path,
        tmp_path / "missing_model.joblib",
        tmp_path / "missing_metadata.json",
    )

    with TestClient(create_app(config_path)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert "missing_model.joblib" in body["model_artifact_path"]


def test_predict_returns_clear_error_when_model_missing(tmp_path: Path) -> None:
    config_path = _write_api_config(
        tmp_path,
        tmp_path / "missing_model.joblib",
        tmp_path / "missing_metadata.json",
    )

    with TestClient(create_app(config_path)) as client:
        response = client.post("/predict", json=_sample_record())

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "Model artifact is not loaded. Run training first."


def test_predict_validates_missing_required_field(tmp_path: Path) -> None:
    config_path = _write_api_config(
        tmp_path,
        tmp_path / "missing_model.joblib",
        tmp_path / "missing_metadata.json",
    )
    payload = _sample_record()
    payload.pop("PAY_0")

    with TestClient(create_app(config_path)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "PAY_0" in str(response.json()["detail"])


def test_predict_works_with_trained_model_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "best_model.joblib"
    metadata_path = tmp_path / "best_model_metadata.json"
    config_path = _write_api_config(tmp_path, model_path, metadata_path)

    training_df = add_credit_features(clean_credit_data(_raw_training_df()))
    X, y = prepare_dataset(training_df)
    model, _seconds = train_baseline_model(X, y, {"max_iter": 200, "class_weight": "balanced"})
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "logistic_regression",
                "threshold": 0.4,
                "training_timestamp": "2026-05-29T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with TestClient(create_app(config_path)) as client:
        response = client.post("/predict", json={"records": [_sample_record(), _sample_record()]})

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "logistic_regression"
    assert body["threshold"] == 0.4
    assert body["record_count"] == 2
    assert len(body["predictions"]) == 2
    assert set(body["predictions"][0]) == {"predicted_class", "predicted_probability"}

