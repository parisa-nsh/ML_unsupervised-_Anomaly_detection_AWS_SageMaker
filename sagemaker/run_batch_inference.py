"""
Launch SageMaker Batch Transform or Processing Job for batch inference.

Computes anomaly scores from feature matrix using trained model; writes scores to S3.
IAM role and paths from environment variables.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_config

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor


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

    bucket = os.environ.get("SAGEMAKER_BUCKET") or sm.get("bucket")
    features_prefix = os.environ.get("FEATURES_PREFIX") or sm.get("features_prefix", "data/features")
    model_prefix = os.environ.get("MODEL_PREFIX") or sm.get("model_prefix", "models/autoencoder")
    scores_prefix = os.environ.get("SCORES_PREFIX") or sm.get("scores_prefix", "data/scores")
    job_name_prefix = os.environ.get("INFERENCE_JOB_PREFIX") or sm.get("inference_job_prefix", "anomaly-scores")
    instance_type = os.environ.get("INFERENCE_INSTANCE_TYPE") or sm.get("inference_instance_type", "ml.m5.xlarge")
    training_job_name = os.environ.get("TRAINING_JOB_NAME")
    model_artifact_uri = os.environ.get("MODEL_ARTIFACT_URI")

    if not bucket:
        print("Set SAGEMAKER_BUCKET.", file=sys.stderr)
        return 1

    if not model_artifact_uri and training_job_name:
        # Resolve model tarball from training job
        sm_client = sagemaker.Session().sagemaker_client
        desc = sm_client.describe_training_job(TrainingJobName=training_job_name)
        model_artifact_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]

    if not model_artifact_uri:
        print("Set MODEL_ARTIFACT_URI or TRAINING_JOB_NAME to point to trained model.", file=sys.stderr)
        return 1

    role = get_role()
    features_uri = f"s3://{bucket}/{features_prefix}"
    scores_uri = f"s3://{bucket}/{scores_prefix}"

    scripts_dir = repo_root / "scripts"

    # Use a Processing Job to run inference: download model artifact, run inference.py, upload scores.
    # Alternative: use SageMaker Model + Batch Transform; Processing keeps one script flow.
    processor = PyTorchProcessor(
        framework_version="2.0",
        py_version="py310",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name=job_name_prefix,
    )

    processor.run(
        code="inference.py",
        source_dir=str(scripts_dir),
        inputs=[
            ProcessingInput(
                source=features_uri,
                destination="/opt/ml/processing/input",
                input_name="input",
            ),
            ProcessingInput(
                source=model_artifact_uri,
                destination="/opt/ml/processing/model",
                input_name="model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=scores_uri,
                output_name="scores",
            ),
        ],
        arguments=[
            "--input-data", "/opt/ml/processing/input",
            "--model-dir", "/opt/ml/processing/model",
            "--output-data", "/opt/ml/processing/output",
            "--input-file", "features.csv",
        ],
    )

    print(f"Batch inference completed. Scores at {scores_uri}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
