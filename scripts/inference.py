"""
Batch inference: compute anomaly scores from feature matrix using trained autoencoder.

Designed to run in SageMaker Batch Transform or as a Processing Job.
Loads model and metadata from model dir, scores each sample (reconstruction error), writes CSV to S3.
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

from model import Autoencoder


def _resolve_model_dir(model_dir: Path) -> Path:
    """Return path to dir containing model.pt and metadata.json; extract model.tar.gz if needed."""
    if (model_dir / "model.pt").exists() and (model_dir / "metadata.json").exists():
        return model_dir
    # SageMaker downloads training artifact as model.tar.gz; input dir may be read-only
    tarball = model_dir / "model.tar.gz"
    if not tarball.exists():
        for f in model_dir.iterdir():
            if f.name.endswith(".tar.gz"):
                tarball = f
                break
    if tarball.exists():
        tmp = Path(tempfile.mkdtemp())
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(tmp)
        if (tmp / "model.pt").exists():
            return tmp
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch inference for anomaly scoring.")
    parser.add_argument(
        "--input-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_INPUT", "/opt/ml/processing/input"),
        help="Path to input features (directory or CSV file)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", os.environ.get("SAGEMAKER_MODEL_DIR", "/opt/ml/model")),
        help="Directory containing model.pt and metadata.json (or model.tar.gz)",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_OUTPUT", "/opt/ml/processing/output"),
        help="Path to write anomaly scores CSV",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=os.environ.get("INPUT_FILE", "features.csv"),
        help="Input CSV filename when input-data is a directory",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir = _resolve_model_dir(model_dir)
    if not (model_dir / "model.pt").exists():
        print(f"Model not found: {model_dir / 'model.pt'} (and no model.tar.gz to extract)", file=sys.stderr)
        return 1
    if not (model_dir / "metadata.json").exists():
        print(f"Metadata not found: {model_dir / 'metadata.json'}", file=sys.stderr)
        return 1

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    input_dim = metadata["input_dim"]
    encoding_dim = metadata["encoding_dim"]
    mean = np.array(metadata["mean"], dtype=np.float32)
    std = np.array(metadata["std"], dtype=np.float32)

    model = Autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        hidden_dims=[32, 16],
        dropout=0.0,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    input_path = Path(args.input_data)
    if input_path.is_dir():
        candidate = input_path / args.input_file
        if candidate.exists():
            input_path = candidate
        else:
            # SageMaker downloads S3 prefix into subdirs (e.g. job-name/output/features.csv)
            found = list(input_path.rglob(args.input_file))
            if found:
                input_path = found[0]
            else:
                input_path = candidate
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    import pandas as pd
    df = pd.read_csv(input_path)
    for c in feature_cols:
        if c not in df.columns:
            print(f"Missing feature column: {c}", file=sys.stderr)
            return 1
    X = df[feature_cols].astype(np.float32).values
    X_norm = (X - mean) / std
    np.nan_to_num(X_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        t = torch.from_numpy(X_norm)
        recon = model(t)
        # Anomaly score = MSE per sample (reconstruction error)
        mse = ((t - recon) ** 2).mean(dim=1).numpy()

    id_cols = [c for c in ["timestamp", "machine_id"] if c in df.columns]
    if id_cols:
        out_df = df[id_cols].copy()
    else:
        out_df = pd.DataFrame({"row_index": range(len(df))})
    out_df["anomaly_score"] = mse

    output_path = Path(args.output_data)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "anomaly_scores.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Wrote {len(out_df)} anomaly scores to {out_file}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
