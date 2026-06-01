# MLDS423 Credit Default Prediction

This project builds an end-to-end credit card default prediction pipeline for the MLDS423 final submission. It includes data ingestion, preprocessing, model training, evaluation, a FastAPI inference service, tests, and AWS deployment evidence.

## Deployment Approach

Final deployment approach: `EC2 Training + EC2 API Demo`.

Raw data was stored in S3, EC2 was used for model training, and the FastAPI service was tested on EC2. Deployment evidence is included in the report materials under `reports/`, including S3 screenshots, EC2 terminal output, training logs, model artifact evidence, and API response screenshots.

## Local Run Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python pipeline.py --config configs/config.yaml
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Key Result

The best model is Random Forest with ROC AUC about `0.80`.

## Repository Structure

- `src/` - pipeline, modeling, API, and utility modules
- `configs/` - local and AWS demo configuration files
- `scripts/` - dataset download, S3 upload helper, and API smoke test
- `tests/` - unit and API tests
- `reports/` - final metrics and AWS deployment evidence
- `artifacts/` - model metadata; large generated artifacts are ignored

## Completed Requirements

- [x] Data ingestion and preprocessing
- [x] Feature engineering
- [x] Baseline Logistic Regression model
- [x] Random Forest model selection
- [x] Model evaluation and metrics export
- [x] FastAPI inference service
- [x] Docker support
- [x] S3 raw data storage evidence
- [x] EC2 model training evidence
- [x] EC2 FastAPI demo evidence
- [x] Automated tests and compile verification

## Remaining Items

- [ ] Final presentation PPT/PDF
- [ ] Final source zip if required
- [ ] Confirm no secrets or large artifacts are included

## Security Note

AWS credentials, `.env` files, raw data, processed data, and large model artifacts are not committed.
