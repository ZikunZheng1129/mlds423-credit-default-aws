# Minimal AWS Demo Runbook: S3 + EC2

This runbook describes a simple live AWS evidence path for the MLDS423 final project. It intentionally avoids Terraform, CDK, and automated resource creation. Create resources manually in the AWS Console and use placeholder values in commands.

Default region: `us-east-2` / US East (Ohio).

Use placeholders:

- `<bucket-name>`
- `<date>`
- `<ec2-public-ip>`
- `<your-repo-url>`
- `<security-group-id>`

Suggested S3 bucket name pattern:

```text
mlds423-credit-default-zikun-<date>
```

## 1. Create S3 Bucket

In the AWS Console:

1. Open S3.
2. Choose `Create bucket`.
3. Region: `us-east-2`.
4. Bucket name example: `mlds423-credit-default-zikun-<date>`.
5. Keep public access blocked.
6. Enable default encryption if available.
7. Create the bucket.

Recommended prefix structure:

```text
raw/
processed/
models/
metadata/
reports/
```

## 2. Upload Raw Dataset To S3

From your local machine after running `python scripts/download_data.py`:

```bash
aws s3 cp data/raw/default_of_credit_card_clients.csv \
  s3://<bucket-name>/raw/default_of_credit_card_clients.csv \
  --region us-east-2
```

Confirm:

```bash
aws s3 ls s3://<bucket-name>/raw/ --region us-east-2
```

Evidence to capture:

- S3 bucket screenshot showing `raw/default_of_credit_card_clients.csv`.

## 3. Create EC2 IAM Role

Create or use an IAM role for EC2 with least-privilege S3 access to your demo bucket.

Minimum policy shape, scoped to your bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::<bucket-name>"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::<bucket-name>/*"
    }
  ]
}
```

Do not put AWS access keys on EC2. Use the IAM instance role.

## 4. Launch EC2

Recommended demo instance:

- Instance type: `t3.micro` or `t3.small`
- AMI: Amazon Linux 2023 or Ubuntu LTS
- Region: `us-east-2`
- IAM role: the S3 access role from the previous step
- Security group:
  - SSH `22` from your IP only
  - Optional API demo port `8000` from your IP only

Evidence to capture:

- EC2 instance screenshot showing instance state and type.

## 5. Install System Dependencies On EC2

### Amazon Linux 2023

```bash
sudo dnf update -y
sudo dnf install -y git python3 python3-pip python3-devel gcc
```

Optional Docker:

```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
```

Log out and back in after adding the Docker group.

### Ubuntu LTS

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip build-essential
```

Optional Docker:

```bash
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
```

Log out and back in after adding the Docker group.

## 6. Copy Or Clone Project

Option A, clone from Git:

```bash
git clone <your-repo-url>
cd project
```

Option B, upload a zip file:

```bash
unzip project.zip
cd project
```

## 7. Configure AWS Demo Config

Edit `configs/config_aws_demo.yaml` and replace every `<bucket-name>` with your bucket name.

Quick replacement example:

```bash
BUCKET_NAME=mlds423-credit-default-zikun-<date>
sed -i.bak "s/<bucket-name>/${BUCKET_NAME}/g" configs/config_aws_demo.yaml
```

When using the example above, replace `<date>` with your actual date suffix first.

The AWS demo config uses:

```text
raw_uri: s3://<bucket-name>/raw/default_of_credit_card_clients.csv
processed_uri: s3://<bucket-name>/processed/credit_default_processed.csv
model_artifact_uri: s3://<bucket-name>/models/best_model.joblib
```

The API model path remains local:

```text
artifacts/best_model.joblib
```

This lets the EC2 API demo load the model from disk after training.

## 8. Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If XGBoost fails due to system libraries, the pipeline logs a warning and continues with Logistic Regression and Random Forest.

## 9. Run Training Pipeline On EC2

```bash
python pipeline.py --config configs/config_aws_demo.yaml
```

Expected behavior:

- raw CSV is read from S3
- processed data is written to S3
- model training runs on EC2
- best local model is saved to `artifacts/best_model.joblib`
- best model is uploaded to `s3://<bucket-name>/models/best_model.joblib`
- metrics and confusion matrix are generated locally under `reports/`

Current automatic S3 uploads:

- processed dataset through `data.processed_uri`
- best model through `artifacts.model_artifact_uri`

Current local-only outputs:

- `artifacts/best_model_metadata.json`
- `artifacts/models/*.joblib`
- `artifacts/metadata/*.json`
- `reports/metrics_summary.csv`
- `reports/confusion_matrices.json`

Evidence to capture:

- terminal screenshot showing pipeline completion
- terminal screenshot showing best model selected

## 10. Upload Local Reports And Metadata To S3

Use the helper script:

```bash
chmod +x scripts/upload_demo_outputs_to_s3.sh
./scripts/upload_demo_outputs_to_s3.sh <bucket-name>
```

Or run the AWS CLI commands manually:

```bash
aws s3 cp artifacts/best_model_metadata.json \
  s3://<bucket-name>/metadata/best_model_metadata.json \
  --region us-east-2

aws s3 cp reports/metrics_summary.csv \
  s3://<bucket-name>/reports/metrics_summary.csv \
  --region us-east-2

aws s3 cp reports/confusion_matrices.json \
  s3://<bucket-name>/reports/confusion_matrices.json \
  --region us-east-2

aws s3 cp artifacts/models \
  s3://<bucket-name>/models/ \
  --recursive \
  --exclude "*" \
  --include "*.joblib" \
  --region us-east-2

aws s3 cp artifacts/metadata \
  s3://<bucket-name>/metadata/ \
  --recursive \
  --exclude "*" \
  --include "*.json" \
  --region us-east-2
```

Confirm:

```bash
aws s3 ls s3://<bucket-name>/ --recursive --region us-east-2
```

Evidence to capture:

- S3 screenshot showing `processed/`, `models/`, `metadata/`, and `reports/`.

## 11. Run FastAPI On EC2

```bash
source .venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

In a second SSH session:

```bash
curl http://localhost:8000/health
```

If your security group allows port `8000` from your IP only:

```bash
curl http://<ec2-public-ip>:8000/health
```

Evidence to capture:

- terminal screenshot showing `/health` with `model_loaded: true`.

## 12. Test Predict Endpoint

From EC2 or your local terminal if port `8000` is open:

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

Evidence to capture:

- terminal screenshot showing predicted class and predicted probability.

## 13. Optional Docker API On EC2

```bash
docker build -t mlds423-credit-default-api .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  mlds423-credit-default-api
```

Then test:

```bash
curl http://localhost:8000/health
```

## 14. Stop EC2 After Demo

To avoid charges:

1. Stop the EC2 instance after the demo.
2. Delete unused EBS volumes if any are left behind.
3. Remove demo S3 objects or apply lifecycle rules if you no longer need them.
4. Keep the S3 bucket only if you need it for final evidence.

## AWS Demo Evidence Checklist

- [ ] S3 bucket screenshot showing `raw/default_of_credit_card_clients.csv`.
- [ ] EC2 instance screenshot showing instance running in `us-east-2`.
- [ ] Terminal screenshot of `python pipeline.py --config configs/config_aws_demo.yaml` completing.
- [ ] Terminal screenshot or log line showing best model selected.
- [ ] S3 screenshot showing processed data, model artifacts, metadata, and reports.
- [ ] Terminal screenshot of `curl http://localhost:8000/health`.
- [ ] Terminal screenshot of `curl /predict` response.
- [ ] Optional screenshot of public EC2 endpoint if port `8000` is opened to your IP.

## Security Notes

- Do not use root AWS credentials.
- Do not copy AWS access keys to EC2.
- Use an EC2 IAM role for S3 access.
- Keep S3 public access blocked.
- Restrict SSH and API security group inbound rules to your IP.
- Stop EC2 after demo to avoid charges.
