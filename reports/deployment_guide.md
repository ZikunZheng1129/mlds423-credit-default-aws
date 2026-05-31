# Manual AWS Deployment Guide

This guide describes manual AWS deployment paths for an MLDS423 demo. It does not create live resources automatically and does not require credentials to be stored in the repository.

Default region in examples: `us-east-2`.

Use placeholders:

- `<bucket-name>`
- `<account-id>`
- `<region>`
- `<ecr-repo-name>`
- `<security-group-id>`
- `<subnet-id>`

## Option A: EC2 Training + EC2 API Demo

### 1. Create an S3 Bucket

Create a bucket in `us-east-2`, for example:

```text
<bucket-name>
```

Recommended prefixes:

```text
raw/
processed/
models/
metadata/
reports/
logs/
```

### 2. Upload Raw Dataset

```bash
aws s3 cp data/raw/default_of_credit_card_clients.csv \
  s3://<bucket-name>/raw/default_of_credit_card_clients.csv \
  --region us-east-2
```

### 3. Launch EC2 Instance

Suggested demo instance:

- Amazon Linux 2023 or Ubuntu
- `t3.micro` or `t3.small`
- IAM instance role with least-privilege S3 access to `<bucket-name>`

Do not place AWS keys on the instance. Use the instance profile/IAM role.

### 4. Copy or Clone Project

```bash
git clone <your-repo-url>
cd project
```

Or upload the final submission ZIP and unzip it.

### 5. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure Cloud Paths

Example environment settings:

```bash
export AWS_REGION=us-east-2
export S3_BUCKET=<bucket-name>
export RAW_DATA_URI=s3://<bucket-name>/raw/default_of_credit_card_clients.csv
export PROCESSED_DATA_URI=s3://<bucket-name>/processed/credit_default_processed.csv
export MODEL_ARTIFACT_URI=s3://<bucket-name>/models/best_model.joblib
```

### 7. Run Training

```bash
python pipeline.py --config configs/config.yaml
```

The current code saves local artifacts and uploads the best model if `MODEL_ARTIFACT_URI` is an S3 URI. Processed data S3 output is also supported through `save_processed_data`.

### 8. Run API Directly On EC2

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Security group demo rule:

```text
Inbound TCP 8000 from your IP only
```

Test:

```bash
curl http://<ec2-public-ip>:8000/health
```

### 9. Run API With Docker On EC2

```bash
docker build -t mlds423-credit-default-api .
docker run --rm -p 8000:8000 \
  -e AWS_REGION=us-east-2 \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  mlds423-credit-default-api
```

## Option B: ECS/Fargate API Demo

### 1. Build Docker Image

```bash
docker build -t mlds423-credit-default-api .
```

### 2. Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name <ecr-repo-name> \
  --region us-east-2
```

### 3. Authenticate Docker To ECR

```bash
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-2.amazonaws.com
```

### 4. Tag And Push Image

```bash
docker tag mlds423-credit-default-api:latest \
  <account-id>.dkr.ecr.us-east-2.amazonaws.com/<ecr-repo-name>:latest

docker push \
  <account-id>.dkr.ecr.us-east-2.amazonaws.com/<ecr-repo-name>:latest
```

### 5. Create ECS Cluster, Task Definition, And Service

Use the AWS console or CLI to create:

- ECS cluster
- Fargate task definition
- container port `8000`
- task execution role for ECR/CloudWatch
- task role for S3 model/artifact access
- CloudWatch log group

Environment variables for the task:

```text
AWS_REGION=us-east-2
S3_BUCKET=<bucket-name>
CREDIT_DEFAULT_CONFIG_PATH=configs/config.yaml
CREDIT_DEFAULT_LOG_LEVEL=INFO
MODEL_ARTIFACT_URI=s3://<bucket-name>/models/best_model.joblib
```

For a fully cloud-hosted API, add code or startup logic later to download the model artifact from S3 into the container before serving. For the current local-container workflow, the model artifact is mounted at `artifacts/best_model.joblib`.

### 6. Networking

Demo choices:

- Public IP on Fargate task for a short demo.
- Preferred production shape: private ECS tasks behind an Application Load Balancer.

Optional ALB health check:

```text
GET /health
```

### 7. Logs

Configure ECS container logs to CloudWatch Logs. Review:

```bash
aws logs tail /ecs/<service-name> --follow --region us-east-2
```

## No Credentials In Repository

Never commit:

- `.env`
- AWS access keys
- AWS secret keys
- session tokens
- private key files
- account-specific deployment output

