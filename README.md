# Credit Card Default Prediction AWS ML System

This repository contains an end-to-end, AWS-ready machine learning system for predicting credit card default risk. The project uses the existing notebook prototype, `MLDS420_Final_Project_Group_4 (2).ipynb`, as the modeling reference and aligns the implementation with MLDS423 Cloud Engineering requirements.

The focus is deployment readiness rather than model novelty: modular code, S3-ready data/artifact paths, configuration management, logging, Docker support, and a FastAPI inference service.

## Business Problem

Credit card issuers need an operational way to identify customers who are at elevated risk of defaulting next month. This project turns the notebook prototype into a reproducible cloud-ready ML system that can train credit-risk models, store artifacts, and serve predictions through an API.

## Dataset

The project uses the public UCI Default of Credit Card Clients dataset from KaggleHub:

```text
uciml/default-of-credit-card-clients-dataset
```

The dataset contains 30,000 customer records with credit limits, demographic variables, repayment status, bill amounts, payment amounts, and the target `default.payment.next.month`.

## AWS Architecture Summary

- Store raw data, processed data, and model artifacts in S3.
- Run training locally, on EC2, or as an ECS task.
- Train Logistic Regression, Random Forest, and XGBoost models.
- Save model artifacts and metrics for reproducibility.
- Serve the selected model through a Dockerized FastAPI app.
- Use environment variables or IAM roles for AWS credentials.
- Emit logs compatible with CloudWatch collection.

## Current Status

The repository includes reusable configuration, logging, S3 path utilities, data preparation modules, model training/evaluation, artifact saving, a FastAPI inference service, Docker packaging, and AWS deployment-ready documentation.

## Local Setup Preview

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

The default project settings live in `configs/config.yaml`. Later pipeline steps will load this file through `src.config.load_config()` and apply environment variable overrides for cloud-specific paths and deployment settings.

Supported overrides:

- `AWS_REGION`
- `S3_BUCKET`
- `RAW_DATA_URI`
- `PROCESSED_DATA_URI`
- `MODEL_ARTIFACT_URI`
- `CREDIT_DEFAULT_LOG_LEVEL`

For local development, copy `.env.example` to `.env` and fill in only non-secret project settings. AWS credentials should not be stored in YAML files or committed to the repository.

## Data Preparation

The project expects the UCI credit card default CSV from the notebook prototype. For local runs, place it here:

```text
data/raw/default_of_credit_card_clients.csv
```

The easiest reproducible path is KaggleHub:

```bash
python scripts/download_data.py
python pipeline.py --config configs/config.yaml
```

The script downloads the public KaggleHub dataset `uciml/default-of-credit-card-clients-dataset`, locates `UCI_Credit_Card.csv` from the local KaggleHub download directory, and copies it to `data/raw/default_of_credit_card_clients.csv`, which is the pipeline default. KaggleHub is pinned to `0.3.13` in `requirements.txt` because newer versions may currently trigger a `kagglesdk` import issue in some local environments.

The code does not use Kaggle notebook paths such as `/kaggle/input/...`. If KaggleHub or Kaggle API credentials are needed in your local environment, keep them outside this repository. Do not commit Kaggle credentials, AWS credentials, or a real `.env`.

To test only ingestion, cleaning, feature engineering, and local processed-data saving once the CSV is available:

```bash
python - <<'PY'
from src.config import load_config
from src.data.ingest import load_raw_data
from src.data.preprocess import clean_credit_data, save_processed_data
from src.data.feature_engineering import add_credit_features

cfg = load_config()
raw = load_raw_data(cfg["data"]["raw_uri"])
clean = clean_credit_data(raw)
features = add_credit_features(clean)
save_processed_data(features, cfg["data"]["processed_uri"])
PY
```

`RAW_DATA_URI` and `PROCESSED_DATA_URI` may be local paths or S3 URIs such as `s3://your-bucket/raw/default_of_credit_card_clients.csv`. S3 downloads/uploads use boto3's default credential chain, so use environment variables, AWS CLI config, or IAM roles. Do not commit Kaggle or AWS credentials.

## Training Pipeline

Install dependencies and place the raw CSV before running training:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Expected local data location:

```text
data/raw/default_of_credit_card_clients.csv
```

Run the local training pipeline:

```bash
python pipeline.py --config configs/config.yaml
```

The pipeline performs ingestion, cleaning, feature engineering, stratified train/test split, threshold tuning, model training, evaluation, and artifact saving. It trains:

- Logistic Regression baseline
- Random Forest challenger
- XGBoost challenger, when `xgboost` is installed

Generated outputs:

- `data/processed/credit_default_processed.csv`
- `artifacts/models/logistic_regression.joblib`
- `artifacts/models/random_forest.joblib`
- `artifacts/models/xgboost.joblib`, if available
- `artifacts/best_model.joblib`
- `artifacts/best_model_metadata.json`
- `artifacts/metadata/*_metadata.json`
- `reports/metrics_summary.csv`
- `reports/confusion_matrices.json`

The best model is selected by highest ROC AUC, with default-class F1 as the tie-breaker. The selected threshold is tuned on training folds to improve minority-class F1.

## Model Results

Local verification selected Random Forest as the best available model.

| Metric | Value |
| --- | ---: |
| Best model | `random_forest` |
| ROC AUC | `0.7998815043401374` |
| Accuracy | `0.8068333333333333` |
| Precision | `0.5625` |
| Recall | `0.5697061039939714` |
| F1 | `0.5660801198053164` |
| Selected threshold | `0.55` |

XGBoost is supported by the training pipeline. During the verified macOS run, XGBoost was skipped because OpenMP / `libomp.dylib` was missing locally. The fallback worked as designed: the pipeline logged the issue, continued with Logistic Regression and Random Forest, saved reports/artifacts, and selected the best available model.

## FastAPI Inference Service

Generate the model artifact first:

```bash
python pipeline.py --config configs/config.yaml
```

Start the API locally:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Check service health:

```bash
curl http://localhost:8000/health
```

Send a single prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "PAY_AMT6": 0
  }'
```

Batch requests are also supported with `{"records": [{...}, {...}]}`. The API loads `artifacts/best_model.joblib` and uses `artifacts/best_model_metadata.json` for the model name and tuned threshold when available. If the model artifact is missing, `/health` still works and `/predict` returns a clear 503 error.

## Docker

Build the API image:

```bash
docker build -t mlds423-credit-default-api .
```

Run the container directly:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  mlds423-credit-default-api
```

Or run with Docker Compose:

```bash
docker compose up --build
```

Test health from another terminal:

```bash
curl http://localhost:8000/health
# or
./scripts/smoke_test_api.sh
```

Run the training pipeline before using `/predict`:

```bash
python pipeline.py --config configs/config.yaml
```

If `artifacts/best_model.joblib` does not exist yet, the container still starts and `/health` still works, but `/predict` returns a clear missing-model error. AWS credentials are not baked into the image; boto3 uses environment variables, AWS CLI config, or IAM roles at runtime.

## Troubleshooting

**XGBoost fails on macOS with `libomp.dylib` missing**

If training logs show an XGBoost library loading error such as:

```text
Library not loaded: @rpath/libomp.dylib
```

install OpenMP locally:

```bash
brew install libomp
```

The pipeline is defensive: if XGBoost cannot be imported or initialized, it logs a warning, skips XGBoost, and continues training Logistic Regression and Random Forest. Metrics, artifacts, reports, and best-model selection are still produced from the available models.

## Local Verification Results

Local verification completed successfully with the following evidence:

- Dataset downloaded successfully with KaggleHub.
- Raw data saved to `data/raw/default_of_credit_card_clients.csv`.
- Raw data shape: `(30000, 25)`.
- Pipeline completed successfully.
- XGBoost was skipped on macOS because `libomp.dylib` is missing, and the fallback worked correctly.
- Trained models: Logistic Regression and Random Forest.
- Best model selected: `random_forest`.
- Best model artifacts created:
  - `artifacts/best_model.joblib`
  - `artifacts/best_model_metadata.json`
- Reports created:
  - `reports/metrics_summary.csv`
  - `reports/confusion_matrices.json`
- FastAPI `/health` returned `status: ok`, `model_loaded: true`, and `model_name: random_forest`.
- FastAPI `/predict` worked for one sample record:
  - `predicted_class: 1`
  - `predicted_probability: 0.8644564746839503`

## Cloud Engineering Documentation

Submission-ready cloud notes are in `reports/`:

- [Architecture Notes](reports/architecture_notes.md): intended AWS architecture, S3/training/inference flow, and text diagram.
- [Cost Estimate Notes](reports/cost_estimate_notes.md): high-level AWS Cost Calculator assumptions for a small `us-east-2` demo.
- [Deployment Guide](reports/deployment_guide.md): manual EC2 and ECS/Fargate deployment paths with placeholder commands.
- [AWS Demo Runbook](reports/aws_demo_runbook.md): minimal live S3 + EC2 evidence path for the final presentation.
- [Security, Logging, And Monitoring](reports/security_logging_monitoring.md): credential handling, IAM, CloudWatch, health checks, and model versioning.

No live AWS resources are created by this repository by default.

## Security Notes

- Do not commit `.env` files.
- Do not hard-code AWS credentials.
- boto3 integration uses the default AWS credential chain: environment variables, AWS CLI profiles, ECS task roles, EC2 instance profiles, or other IAM-based providers.

## Submission Checklist

- [x] Source code scaffold and modular pipeline implemented.
- [x] Original notebook preserved: `MLDS420_Final_Project_Group_4 (2).ipynb`.
- [x] Project requirement PDF preserved: `Project.pdf`.
- [x] Local training pipeline runnable with `python pipeline.py --config configs/config.yaml`.
- [x] FastAPI service runnable with `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`.
- [x] Docker build/run instructions available.
- [x] AWS architecture documented.
- [x] Cost estimate documented.
- [x] Security, logging, and monitoring documented.
- [x] No AWS credentials or account-specific values committed.
