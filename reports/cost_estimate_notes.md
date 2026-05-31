# AWS Cost Estimate Notes

Region for estimates: `us-east-2` / US East (Ohio).

These are high-level planning estimates for a small academic demo. Actual AWS costs vary by uptime, exact instance type, storage volume, log volume, network traffic, and current AWS pricing. Use AWS Pricing Calculator for final numbers before running live resources.

## Assumptions

- Dataset: UCI credit default CSV, roughly tens of thousands of rows.
- Processed data and reports: small CSV/JSON files.
- Model artifacts: joblib files, likely small for Logistic Regression, Random Forest, and XGBoost.
- Training: short-running job on a small EC2 instance or ECS task.
- Inference: low-traffic demo API.
- Logs: low-volume CloudWatch logs.
- Data transfer: minimal.

## Estimated Monthly Cost Categories

| Component | Small-project assumption | Expected cost level |
| --- | --- | --- |
| S3 storage | Raw CSV, processed CSV, reports, model artifacts, metadata | Likely under $1/month at this project scale |
| EC2 training | Short run on `t3.micro` or `t3.small`; stop after training | Very low if used only for demos |
| ECS/Fargate API | Small CPU/memory task for demo; cost depends on uptime | Low for short demos; higher if left running all month |
| EC2 API alternative | Small EC2 instance running Docker/FastAPI | Depends heavily on uptime |
| CloudWatch logs | API startup, health, prediction, training logs | Likely low for small log volume |
| Data transfer | Minimal test requests and small artifacts | Likely low for demo usage |
| ECR storage | One or a few Docker images | Usually low for a small repository |

## Demo Cost-Control Recommendations

- Stop EC2 instances after training/demo.
- Scale ECS service desired count to `0` after the presentation if not needed.
- Delete old ECR images if rebuilding frequently.
- Keep CloudWatch log retention short for class demos, for example 7-14 days.
- Use S3 lifecycle rules if many experiment artifacts are created.
- Avoid leaving public demo endpoints running indefinitely.

## AWS Pricing Calculator Inputs To Use

Suggested placeholders:

- Region: `us-east-2`
- S3 Standard storage: less than 1 GB
- EC2 training: `t3.micro` or `t3.small`, a few hours per month
- ECS/Fargate inference: small task size, demo uptime only
- CloudWatch Logs: low ingestion volume, short retention
- Data transfer out: minimal

## Summary

For a short-running class demo, this project should be inexpensive because the dataset and artifacts are small and the compute workload is brief. The main cost risk is leaving EC2 or ECS inference resources running after the demo.

