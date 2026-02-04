"""
Launch SageMaker Processing Job for feature computation.

Uses script processing.py; reads raw CSV from S3, writes feature matrix to S3.
No hard-coded credentials: IAM role and paths from environment variables.
"""

import os
import sys
from pathlib import Path

# Repo root for config_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_config

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor


def get_role():
    """IAM role for SageMaker; must be set externally (e.g. SAGEMAKER_ROLE or notebook role)."""
    role = os.environ.get("SAGEMAKER_ROLE")
    if role:
        return role
    try:
        return sagemaker.get_execution_role()
    except Exception as e:
        print("Set SAGEMAKER_ROLE or run in a SageMaker context that provides get_execution_role().", file=sys.stderr)
        raise e


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = get_config(repo_root)
    sm = cfg.get("sagemaker", {})

    bucket = os.environ.get("SAGEMAKER_BUCKET") or sm.get("bucket")
    raw_prefix = os.environ.get("RAW_DATA_PREFIX") or sm.get("raw_data_prefix", "data/raw")
    features_prefix = os.environ.get("FEATURES_PREFIX") or sm.get("features_prefix", "data/features")
    job_name_prefix = os.environ.get("PROCESSING_JOB_PREFIX") or sm.get("processing_job_prefix", "sensor-features")
    instance_type = os.environ.get("PROCESSING_INSTANCE_TYPE") or sm.get("processing_instance_type", "ml.m5.xlarge")
    rolling_window = os.environ.get("ROLLING_WINDOW") or str(cfg.get("pipeline", {}).get("rolling_window", "12"))

    if not bucket:
        print("Set SAGEMAKER_BUCKET.", file=sys.stderr)
        return 1

    role = get_role()
    raw_uri = f"s3://{bucket}/{raw_prefix}"
    output_uri = f"s3://{bucket}/{features_prefix}"

    scripts_dir = repo_root / "scripts"

    processor = PyTorchProcessor(
        framework_version="2.0",
        py_version="py310",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name=job_name_prefix,
    )

    processor.run(
        code="processing.py",
        source_dir=str(scripts_dir),
        inputs=[
            ProcessingInput(
                source=raw_uri,
                destination="/opt/ml/processing/input",
                input_name="input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=output_uri,
                output_name="features",
            ),
        ],
        arguments=[
            "--rolling-window", rolling_window,
            "--input-file", "sensor_data.csv",
        ],
    )

    print(f"Processing job completed. Features at {output_uri}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
