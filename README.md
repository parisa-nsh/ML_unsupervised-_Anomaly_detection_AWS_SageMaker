# Cloud-Native ML System on AWS SageMaker

A production-style, cloud-native ML system for end-to-end lifecycle on AWS SageMaker: **unsupervised anomaly detection** on time-series sensor data using a PyTorch autoencoder. All stages run on managed AWS infrastructure; artifacts live in S3.

---

## Problem Statement

Modern industrial ML systems often fail not because of model choice, but because of **poor ML lifecycle design**—tight coupling to proprietary data, non-reproducible experiments, and ad-hoc infrastructure.

The system cleanly separates:

- **Data generation and ingestion**
- **Feature computation**
- **Model training**
- **Inference and evaluation**

The reference use case is **unsupervised anomaly detection on time-series sensor data**, chosen because it:

- Reflects realistic industrial ML problems
- Does not require labeled data
- Emphasizes representation learning and system design over model novelty

**Synthetic data** is used intentionally to validate architecture and ML lifecycle ownership without coupling the system to proprietary datasets.

---

## System Architecture

At a high level, the system:

1. **Generates or ingests** time-series sensor data (synthetic in this repo).
2. **Computes features** in a SageMaker Processing Job (rolling statistics, trend, change rate).
3. **Trains** an unsupervised autoencoder using a SageMaker PyTorch Training Job.
4. **Runs batch inference** to produce anomaly scores; all artifacts are stored in S3.

This enables reproducibility, versioning, and inspection of every stage.

```
┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ Synthetic Data   │────▶│ SageMaker Processing │────▶│ SageMaker Training  │────▶│ Batch Inference  │
│ (or real CSV)    │     │ (features)           │     │ (autoencoder)       │     │ (anomaly scores) │
└────────┬─────────┘     └──────────┬───────────┘     └──────────┬──────────┘     └────────┬─────────┘
         │                          │                           │                         │
         ▼                          ▼                           ▼                         ▼
    S3 (raw)                   S3 (features)                S3 (model)                S3 (scores)
```

---

## SageMaker ML Lifecycle

| Stage | Component | Input | Output |
|-------|-----------|--------|--------|
| Data | `scripts/generate_synthetic_data.py` | — | CSV (timestamp, machine_id, sensor_1–3) |
| Processing | SageMaker Processing Job (`processing.py`) | Raw CSV from S3 | Feature matrix (S3) |
| Training | SageMaker Training Job (`train.py`) | Features from S3 | Model weights + metadata (S3) |
| Inference | Processing Job (`inference.py`) | Features + model from S3 | Anomaly scores CSV (S3) |

All paths and the IAM role are **configurable via environment variables**; no hard-coded credentials.

---

## Why Synthetic Data

- **Data-agnostic design**: The pipeline accepts any CSV conforming to the expected schema. Swapping in real industrial data is a matter of ingestion and path configuration.
- **Reproducibility**: Deterministic generation (fixed seed) makes runs reproducible and avoids dependency on proprietary or shifting datasets.
- **Systems focus**: Keeps the scope on ML lifecycle, SageMaker usage, and MLOps; no need for sensitive or domain-specific data to validate the workflow.

---

## Using Real Industrial Data

1. **Ingestion**: Replace or supplement `generate_synthetic_data.py` with a custom loader (e.g. data lake, warehouse, or streaming export). Keep the same schema (timestamp, machine_id, sensor_*) or extend the feature script to match.
2. **Upload**: Put raw CSV (or Parquet) at the S3 path used by the Processing Job (`SAGEMAKER_BUCKET` + `RAW_DATA_PREFIX`).
3. **Features**: Adjust `scripts/processing.py` if the domain needs different or extra features (e.g. FFT, lags). Contract stays: CSV in, feature CSV out.
4. **Training / Inference**: Unchanged if the feature schema is consistent; the autoencoder and inference script read the feature matrix and model artifact from S3.

---

## Project Scope

**In scope**

- Cloud-native ML execution on AWS SageMaker  
- SageMaker Processing Jobs for feature computation  
- SageMaker Training Jobs (PyTorch)  
- Batch inference for anomaly scoring  
- Artifact management via S3  
- Deterministic, reproducible pipelines  
- Script-based execution (no notebook-only workflows)  

**Out of scope (by design)**

- UI or dashboards  
- Business logic or ERP integration  
- Multimodal data  
- Real-time inference endpoints  
- Kubernetes or custom orchestration  
- Dataset-specific tuning or benchmark chasing  

The scope is kept narrow so the workflow stays clear and can be extended later (e.g. packaging as a service or productising for industry use).

---

## Repository Layout

```
config.yaml                    # Paths and hyperparameters (env vars override)
config_loader.py               # Load config.yaml; used by pipeline and sagemaker launchers
run_pipeline.py                # Local end-to-end pipeline (generate → process → train → infer)
pytest.ini                     # Pytest config

scripts/
  generate_synthetic_data.py   # Deterministic synthetic time-series
  processing.py                # Feature computation (Processing Job)
  model.py                     # Shared autoencoder definition
  train.py                     # Training entrypoint (Training Job)
  inference.py                 # Batch inference (anomaly scores)
  evaluate.py                  # Summary stats and report from anomaly scores

sagemaker/
  run_processing.py            # Launch feature Processing Job
  run_training.py              # Launch Training Job
  run_batch_inference.py       # Launch batch inference (Processing)

tests/
  conftest.py                  # Pytest: add scripts to path
  test_generate_synthetic_data.py
  test_processing.py
  test_model.py
  test_train.py
  test_inference.py
  test_evaluate.py
  test_integration.py          # Full pipeline (pytest -m integration)
```

---

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies (numpy, pandas, PyTorch, boto3, sagemaker, pytest for tests).

---

## Usage

### Prerequisites for running on SageMaker

To run Processing, Training, and Batch Inference on AWS (instead of locally), you need: an S3 bucket, an IAM role for SageMaker, and AWS CLI (or env vars) configured on your machine. Step-by-step: **[docs/SAGEMAKER_SETUP.md](docs/SAGEMAKER_SETUP.md)**. To run the pipeline end-to-end on SageMaker: **[docs/RUN_ON_SAGEMAKER.md](docs/RUN_ON_SAGEMAKER.md)**.

### Local end-to-end pipeline (no AWS)

Run all five stages in order (generate → process → train → infer → evaluate):

```bash
python run_pipeline.py --n-samples 1000 --epochs 2
```

Outputs: `data/raw/`, `data/features/`, `data/model/`, `data/scores/anomaly_scores.csv`, `data/evaluation/` (evaluation_report.json and evaluation_report.html). Use `--data-dir` to change the base data directory.

### 1. Data generation (local or pipeline)

```bash
export OUTPUT_PATH=data/raw/sensor_data.csv
# Optional: UPLOAD_S3_URI=s3://your-bucket/data/raw/sensor_data.csv
python scripts/generate_synthetic_data.py --n-samples 10000
```

### 2. SageMaker Processing (features)

Set environment variables, then:

```bash
export SAGEMAKER_ROLE=arn:aws:iam::ACCOUNT:role/YourSageMakerRole
export SAGEMAKER_BUCKET=your-bucket
# Optional: RAW_DATA_PREFIX, FEATURES_PREFIX, ROLLING_WINDOW
python sagemaker/run_processing.py
```

Ensure raw data exists at `s3://$SAGEMAKER_BUCKET/$RAW_DATA_PREFIX/sensor_data.csv` (e.g. upload the CSV from step 1).

### 3. SageMaker Training

```bash
export SAGEMAKER_ROLE=...
export SAGEMAKER_BUCKET=...
# Optional: FEATURES_PREFIX, MODEL_PREFIX, EPOCHS, BATCH_SIZE
python sagemaker/run_training.py
```

### 4. Batch inference

```bash
export SAGEMAKER_ROLE=...
export SAGEMAKER_BUCKET=...
export TRAINING_JOB_NAME=anomaly-autoencoder-YYYY-MM-DD-HH-MM-SS  # from training run
# Or: MODEL_ARTIFACT_URI=s3://bucket/models/autoencoder/output/.../output/model.tar.gz
python sagemaker/run_batch_inference.py
```

Scores are written to `s3://$SAGEMAKER_BUCKET/$SCORES_PREFIX/`.

### 5. Evaluation (local)

After inference, run evaluation to get summary stats and a report:

```bash
python scripts/evaluate.py --input data/scores/anomaly_scores.csv --output-dir data/evaluation
```

Optional: `--top-n 20`, `--percentiles 50,90,95,99`. Writes `evaluation_report.json` and `evaluation_report.html` (single artifact: stats, score distribution and time-series plots, top-N table).

### Tests

From the repo root:

```bash
pytest
```

Runs unit tests for data generation (schema, determinism), processing (build_features), model (forward, save/load), train (get_feature_columns), inference (scores output, missing model/metadata), and evaluate (stats, NaN handling). An optional integration test runs the full local pipeline with minimal data (`pytest -m integration`). Skip it for faster runs: `pytest -m "not integration"`.

---

## Configuration

Defaults are read from **`config.yaml`** at the repo root (paths, pipeline options, training and SageMaker settings). **Environment variables override** these (e.g. `SAGEMAKER_BUCKET`, `SAGEMAKER_ROLE`). If `config.yaml` is missing, built-in defaults are used.

| Variable | Description | Default (config or built-in) |
|----------|-------------|------------------------------|
| `SAGEMAKER_ROLE` | IAM role for SageMaker jobs | (required, or from SageMaker context) |
| `SAGEMAKER_BUCKET` | S3 bucket for data and artifacts | (required; set in env) |
| `RAW_DATA_PREFIX` | S3 prefix for raw CSV | `data/raw` |
| `FEATURES_PREFIX` | S3 prefix for feature output | `data/features` |
| `MODEL_PREFIX` | S3 prefix for model output | `models/autoencoder` |
| `SCORES_PREFIX` | S3 prefix for anomaly scores | `data/scores` |
| `ROLLING_WINDOW` | Rolling window size in processing | `12` |
| `EPOCHS`, `BATCH_SIZE`, `ENCODING_DIM` | Training hyperparameters | from `config.yaml` or `20`, `64`, `8` |

---

## License

See repository license file.
