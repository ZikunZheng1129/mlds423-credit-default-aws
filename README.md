# MLDS423 Credit Default Prediction AWS Project

## Brief Overview

This project is an AWS-ready end-to-end machine learning pipeline for credit card default prediction. It includes an S3-ready data pipeline, model training, FastAPI inference, Docker support, and cloud deployment documentation for an MLDS423 Cloud Engineering final project.

## Repository Structure

- `src/` - reusable Python modules for config, data processing, modeling, cloud utilities, and API
- `configs/` - local and AWS demo YAML configuration files
- `scripts/` - data download, API smoke test, and optional S3 upload helpers
- `reports/` - metrics, architecture notes, cost notes, deployment guide, and AWS demo runbook
- `tests/` - unit and API tests
- `artifacts/` - generated model metadata and artifact README; large model files are ignored
- `data/` - raw/processed data folders; CSV files are ignored

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Download

```bash
python scripts/download_data.py
```

This downloads the public KaggleHub dataset and saves:

```text
data/raw/default_of_credit_card_clients.csv
```

Raw data is intentionally not committed to GitHub.

## Run Training Pipeline

```bash
python pipeline.py --config configs/config.yaml
```

## Local Verification Results

- Tests: `25 passed`
- Raw data shape: `(30000, 25)`
- Best model: `random_forest`
- ROC AUC: `0.7998815043401374`
- Accuracy: `0.8068333333333333`
- Precision: `0.5625`
- Recall: `0.5697061039939714`
- F1: `0.5660801198053164`
- Threshold: `0.55`
- FastAPI `/health`: successful, `model_loaded: true`
- FastAPI `/predict`: successful with one sample record

## Run API Locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t mlds423-credit-default-api .
docker run --rm -p 8000:8000 mlds423-credit-default-api
```

For local model/data access during a real API demo, mount local folders as shown in `reports/aws_demo_runbook.md` or use `docker-compose.yml`.

## AWS Documentation

- [Architecture Notes](reports/architecture_notes.md)
- [Cost Estimate Notes](reports/cost_estimate_notes.md)
- [Deployment Guide](reports/deployment_guide.md)
- [Security, Logging, And Monitoring Notes](reports/security_logging_monitoring.md)
- [AWS Demo Runbook](reports/aws_demo_runbook.md)

## Completed Requirements

- [x] Data ingestion and preprocessing
- [x] Feature engineering
- [x] Baseline Logistic Regression
- [x] Advanced Random Forest
- [x] Optional XGBoost support with graceful fallback
- [x] Metrics and model artifact generation
- [x] FastAPI inference service
- [x] Docker support
- [x] S3-ready config and utilities
- [x] AWS architecture documentation
- [x] Cost estimate notes
- [x] Security/logging/monitoring notes
- [x] Unit tests

## Still Needed According To MLDS423 Requirements

- [ ] Final live AWS deployment evidence still needs to be completed by the team.
- [ ] Capture screenshots/logs for AWS demo:
  - S3 raw dataset
  - cloud training run
  - artifacts/reports uploaded to S3
  - `/health` and `/predict` if API is deployed
- [ ] Final architecture diagram should be exported from diagrams.net or equivalent.
- [ ] AWS Cost Calculator screenshot/export should be added if required by instructor.
- [ ] Final presentation PPT/PDF still needs to be prepared.
- [ ] Final source-code zip/package should be prepared for submission.

## Security Note

`.env`, AWS credentials, raw data, processed data, and large model artifacts are not committed. boto3 uses IAM roles, environment variables, AWS CLI profiles, or the default credential chain.

