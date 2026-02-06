"""
Launch SageMaker PyTorch Training Job for autoencoder training.

Uses script train.py; reads feature CSV from S3, writes model and metadata to S3.
IAM role and paths from environment variables.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_config

import sagemaker
from sagemaker.pytorch import PyTorch


def get_role():
    """IAM role for SageMaker; must be set externally."""
    role = os.environ.get("SAGEMAKER_ROLE")
    if role:
        return role
    try:
        return sagemaker.get_execution_role()
    except Exception as e:
        print("Set SAGEMAKER_ROLE or run in a SageMaker context.", file=sys.stderr)
        raise e


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = get_config(repo_root)
    sm = cfg.get("sagemaker", {})
    pl = cfg.get("pipeline", {})
    tr = cfg.get("training", {})

    bucket = os.environ.get("SAGEMAKER_BUCKET") or sm.get("bucket")
    features_prefix = os.environ.get("FEATURES_PREFIX") or sm.get("features_prefix", "data/features")
    model_prefix = os.environ.get("MODEL_PREFIX") or sm.get("model_prefix", "models/autoencoder")
    job_name_prefix = os.environ.get("TRAINING_JOB_PREFIX") or sm.get("training_job_prefix", "anomaly-autoencoder")
    instance_type = os.environ.get("TRAINING_INSTANCE_TYPE") or sm.get("training_instance_type", "ml.m5.xlarge")
    epochs = os.environ.get("EPOCHS") or str(pl.get("epochs", 20))
    batch_size = os.environ.get("BATCH_SIZE") or str(pl.get("batch_size", 64))
    encoding_dim = os.environ.get("ENCODING_DIM") or str(tr.get("encoding_dim", 8))

    if not bucket:
        print("Set SAGEMAKER_BUCKET.", file=sys.stderr)
        return 1

    role = get_role()
    train_uri = f"s3://{bucket}/{features_prefix}"
    output_path = f"s3://{bucket}/{model_prefix}/output"

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(scripts_dir),
        role=role,
        framework_version="2.0",
        py_version="py310",
        instance_type=instance_type,
        instance_count=1,
        output_path=output_path,
        base_job_name=job_name_prefix,
        hyperparameters={
            "epochs": int(epochs),
            "batch-size": int(batch_size),
            "encoding-dim": int(encoding_dim),
        },
    )

    estimator.fit({"training": train_uri})

    job_name = estimator.latest_training_job.name
    print(f"Training completed. Model at {estimator.model_data}", file=sys.stderr)
    print(f"Set TRAINING_JOB_NAME={job_name} before running run_batch_inference.py", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
