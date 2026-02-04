"""Load config.yaml from repo root. Environment variables override when used by callers."""

import yaml
from pathlib import Path


def get_config(root: Path | None = None) -> dict:
    """Load config.yaml from root; return merged dict. Returns {} if file missing."""
    if root is None:
        root = Path(__file__).resolve().parent
    path = root / "config.yaml"
    if not path.exists():
        return _default_config()
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else _default_config()


def _default_config() -> dict:
    """Defaults when config.yaml is absent."""
    return {
        "pipeline": {
            "data_dir": "data",
            "n_samples": 1000,
            "epochs": 2,
            "batch_size": 64,
            "rolling_window": 12,
            "top_n": 10,
        },
        "data": {
            "raw_subdir": "raw",
            "features_subdir": "features",
            "model_subdir": "model",
            "scores_subdir": "scores",
            "evaluation_subdir": "evaluation",
        },
        "training": {"encoding_dim": 8, "lr": 0.001},
        "sagemaker": {
            "raw_data_prefix": "data/raw",
            "features_prefix": "data/features",
            "model_prefix": "models/autoencoder",
            "scores_prefix": "data/scores",
            "processing_job_prefix": "sensor-features",
            "training_job_prefix": "anomaly-autoencoder",
            "inference_job_prefix": "anomaly-scores",
            "processing_instance_type": "ml.m5.xlarge",
            "training_instance_type": "ml.m5.xlarge",
            "inference_instance_type": "ml.m5.xlarge",
        },
    }
