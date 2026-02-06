# Run the pipeline on SageMaker

Use this after you have: an S3 bucket, a SageMaker IAM role, AWS CLI configured, and **approved service quotas** for `ml.m5.xlarge` (training and processing) in your region.

---

## 1. Set environment variables

**Windows (PowerShell or CMD):**

```powershell
$env:SAGEMAKER_BUCKET = "your-bucket-name"
$env:SAGEMAKER_ROLE = "arn:aws:iam::YOUR_ACCOUNT_ID:role/YourSageMakerRole"
$env:AWS_DEFAULT_REGION = "ca-central-1"
```

**Linux / macOS:**

```bash
export SAGEMAKER_BUCKET=your-bucket-name
export SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT_ID:role/YourSageMakerRole
export AWS_DEFAULT_REGION=ca-central-1
```

Use your real bucket name, role ARN, and region (e.g. `ca-central-1` for Canada Central).

---

## 2. Put raw data in S3

The processing job expects raw CSV at `s3://<bucket>/data/raw/sensor_data.csv`.

**Option A – Generate locally and upload**

From the repo root:

```powershell
python scripts/generate_synthetic_data.py --n-samples 5000 --output data/raw/sensor_data.csv
aws s3 cp data/raw/sensor_data.csv s3://$env:SAGEMAKER_BUCKET/data/raw/sensor_data.csv
```

(Linux/macOS: use `s3://$SAGEMAKER_BUCKET/data/raw/sensor_data.csv`.)

**Option B – Use your own CSV**

Upload any CSV with columns `timestamp`, `machine_id`, `sensor_1`, `sensor_2`, `sensor_3` to:

`s3://<your-bucket>/data/raw/sensor_data.csv`

---

## 3. Run the three SageMaker jobs in order

From the repo root, with `SAGEMAKER_BUCKET` and `SAGEMAKER_ROLE` set:

### Step 1: Feature processing

```powershell
python sagemaker/run_processing.py
```

Waits for the job to finish. Outputs features to `s3://<bucket>/data/features/`.

### Step 2: Training

```powershell
python sagemaker/run_training.py
```

Waits for the training job. Note the **job name** printed at the end (e.g. `anomaly-autoencoder-2026-02-03-12-00-00`). Outputs model to `s3://<bucket>/models/autoencoder/output/`.

### Step 3: Batch inference

Set the training job name from Step 2, then run:

**Windows:**

```powershell
$env:TRAINING_JOB_NAME = "anomaly-autoencoder-YYYY-MM-DD-HH-MM-SS"
python sagemaker/run_batch_inference.py
```

**Linux/macOS:**

```bash
export TRAINING_JOB_NAME=anomaly-autoencoder-YYYY-MM-DD-HH-MM-SS
python sagemaker/run_batch_inference.py
```

Replace `anomaly-autoencoder-YYYY-MM-DD-HH-MM-SS` with the actual job name from Step 2.

Scores are written to `s3://<bucket>/data/scores/`.

---

## 4. (Optional) Download scores and run evaluation locally

```powershell
mkdir -p data/scores data/evaluation
aws s3 cp s3://$env:SAGEMAKER_BUCKET/data/scores/ data/scores/ --recursive
python scripts/evaluate.py --input data/scores/anomaly_scores.csv --output-dir data/evaluation
```

Open `data/evaluation/evaluation_report.html` for the report.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `ResourceLimitExceeded` | Service quotas for `ml.m5.xlarge` (training and processing) in your region. See [SAGEMAKER_SETUP.md](SAGEMAKER_SETUP.md#6-service-quotas-required-in-your-region). |
| `Set SAGEMAKER_BUCKET` / `Set SAGEMAKER_ROLE` | Env vars not set in this shell. Set them and run the script again. |
| Raw data not found | File must be at `s3://<bucket>/data/raw/sensor_data.csv` (or the prefix set by `RAW_DATA_PREFIX`). |
| Inference fails: training job not found | `TRAINING_JOB_NAME` must match the exact name from `run_training.py` (e.g. from the console or the script output). |
