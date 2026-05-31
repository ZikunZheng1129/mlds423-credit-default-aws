#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <bucket-name>" >&2
  exit 1
fi

BUCKET_NAME="$1"
REGION="${AWS_REGION:-us-east-2}"

aws s3 cp artifacts/best_model_metadata.json "s3://${BUCKET_NAME}/metadata/best_model_metadata.json" --region "${REGION}"
aws s3 cp reports/metrics_summary.csv "s3://${BUCKET_NAME}/reports/metrics_summary.csv" --region "${REGION}"
aws s3 cp reports/confusion_matrices.json "s3://${BUCKET_NAME}/reports/confusion_matrices.json" --region "${REGION}"

if [[ -d artifacts/models ]]; then
  aws s3 cp artifacts/models "s3://${BUCKET_NAME}/models/" --recursive --exclude "*" --include "*.joblib" --region "${REGION}"
fi

if [[ -d artifacts/metadata ]]; then
  aws s3 cp artifacts/metadata "s3://${BUCKET_NAME}/metadata/" --recursive --exclude "*" --include "*.json" --region "${REGION}"
fi

echo "Uploaded demo outputs to s3://${BUCKET_NAME}/"

