# Security, Logging, And Monitoring Notes

## Configuration Management

Project defaults live in `configs/config.yaml`. Cloud-specific overrides can be supplied through environment variables:

- `AWS_REGION`
- `S3_BUCKET`
- `RAW_DATA_URI`
- `PROCESSED_DATA_URI`
- `MODEL_ARTIFACT_URI`
- `CREDIT_DEFAULT_LOG_LEVEL`

`.env.example` is safe to commit because it contains placeholders only. A real `.env` file is local-only and ignored by Git and Docker.

## Credential Handling

AWS credentials are not hard-coded in source files, YAML, Dockerfile, or README commands.

boto3 uses the standard AWS credential chain:

- environment variables
- AWS CLI profile
- EC2 instance profile
- ECS task role
- other IAM-backed providers

For AWS demos, prefer IAM roles over static access keys.

## Least-Privilege IAM

Training and inference should use separate IAM roles when practical.

Training role example scope:

- read raw data from `s3://<bucket-name>/raw/*`
- write processed data to `s3://<bucket-name>/processed/*`
- write models to `s3://<bucket-name>/models/*`
- write reports and metadata to `s3://<bucket-name>/reports/*` and `metadata/*`

Inference role example scope:

- read `s3://<bucket-name>/models/best_model.joblib`
- read model metadata if used
- write logs through the ECS execution role/CloudWatch integration

Avoid broad permissions such as `s3:*` on all buckets.

## Docker Image Security

The Docker build excludes:

- `.env`
- `.aws/`
- generated data
- generated artifacts
- notebook checkpoints
- cache directories
- local keys/certificates

Secrets must be injected at runtime through environment variables, AWS Secrets Manager, SSM Parameter Store, or IAM roles. They should never be baked into the image.

## Logging

`src/logging_utils.py` emits JSON logs that are easy for CloudWatch to ingest and search.

Training logs should capture:

- pipeline start/end
- data loading source
- processed data output
- model training start/end
- selected threshold
- metrics summary
- best model selected
- artifact save/upload failures

API logs should capture:

- startup
- model loading success/failure
- prediction request record count
- prediction latency
- validation errors
- unexpected errors

## Monitoring

Recommended CloudWatch metrics and alarms for a future ECS/EC2 deployment:

- container/task restarts
- API 5xx errors
- prediction latency
- high missing-model errors
- CPU and memory utilization
- log volume anomalies

The API exposes:

```text
GET /health
```

This endpoint reports whether the service is running and whether the model artifact is loaded.

## Model Artifact Versioning

Each trained model has its own artifact and metadata file:

```text
artifacts/models/<model_name>.joblib
artifacts/metadata/<model_name>_metadata.json
```

The selected model is copied to:

```text
artifacts/best_model.joblib
artifacts/best_model_metadata.json
```

Metadata includes model name, feature columns, threshold, metrics, confusion matrix, and training timestamp. In S3, future versions can be separated by run ID or timestamp prefix.

## Submission Safety Checklist

- No real `.env` committed.
- No AWS keys committed.
- No credentials copied into Docker image.
- No account-specific IDs in source code.
- Protected original notebook and project PDF preserved.
- Generated data and artifacts ignored unless intentionally included as evidence.

