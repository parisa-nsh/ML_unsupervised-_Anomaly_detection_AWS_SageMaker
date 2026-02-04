"""
Unsupervised autoencoder training for anomaly detection.

Designed to run as a SageMaker PyTorch Training Job.
- Reads feature CSV from channel (training data).
- Normalizes features, trains autoencoder, saves model and metadata to S3.
No notebook-only execution paths.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import Autoencoder


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def get_feature_columns(df) -> list[str]:
    """Numeric feature columns only (exclude timestamp, machine_id if present)."""
    exclude = {"timestamp", "machine_id"}
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train autoencoder for anomaly detection.")
    parser.add_argument(
        "--train-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        help="Path to training data (directory containing features.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to save model and metadata",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_env_int("EPOCHS", 20),
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_int("BATCH_SIZE", 64),
        help="Batch size",
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=_env_int("ENCODING_DIM", 8),
        help="Autoencoder bottleneck dimension",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_env_float("LR", 1e-3),
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("RANDOM_SEED", 42),
        help="Random seed",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_path = Path(args.train_data)
    feature_file = train_path / "features.csv"
    if not feature_file.exists():
        # Allow single file path
        if train_path.suffix == ".csv":
            feature_file = train_path
        else:
            print(f"Training data not found: {feature_file}", file=sys.stderr)
            return 1

    import pandas as pd
    df = pd.read_csv(feature_file)
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        print("No numeric feature columns found.", file=sys.stderr)
        return 1

    X = df[feature_cols].astype(np.float32).values
    # Normalize: fit on training data only
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    # Avoid nan/inf from upstream (e.g. rolling on short series)
    np.nan_to_num(X_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    dataset = TensorDataset(torch.from_numpy(X_norm))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = len(feature_cols)
    model = Autoencoder(
        input_dim=input_dim,
        encoding_dim=args.encoding_dim,
        hidden_dims=[32, 16],
        dropout=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/len(loader):.6f}", file=sys.stderr)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model state and normalization params for inference
    torch.save(model.state_dict(), model_dir / "model.pt")
    metadata = {
        "feature_columns": feature_cols,
        "input_dim": input_dim,
        "encoding_dim": args.encoding_dim,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model and metadata saved to {model_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
