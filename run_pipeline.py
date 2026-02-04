"""
Local end-to-end pipeline: generate → process → train → infer → evaluate.

Runs all five stages in order. Defaults from config.yaml; CLI overrides.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config_loader import get_config


def run(cmd: list[str], step: str) -> bool:
    """Run a command; return True on success, False on failure."""
    print(f"\n--- {step} ---")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Failed: {step}", file=sys.stderr)
        return False
    return True


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg = get_config(root)
    pl = cfg.get("pipeline", {})
    data_cfg = cfg.get("data", {})

    parser = argparse.ArgumentParser(description="Run full ML pipeline locally.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=pl.get("n_samples", 1000),
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=pl.get("epochs", 2),
        help="Training epochs (use more for real runs)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=pl.get("data_dir", "data"),
        help="Base directory for data/raw, data/features, data/model, data/scores",
    )
    args = parser.parse_args()

    scripts = root / "scripts"
    data = root / args.data_dir
    raw = data / data_cfg.get("raw_subdir", "raw")
    features = data / data_cfg.get("features_subdir", "features")
    model_dir = data / data_cfg.get("model_subdir", "model")
    scores = data / data_cfg.get("scores_subdir", "scores")

    raw.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    scores.mkdir(parents=True, exist_ok=True)

    raw_csv = raw / "sensor_data.csv"
    features_csv = features / "features.csv"

    # 1. Generate
    if not run(
        [
            sys.executable,
            str(scripts / "generate_synthetic_data.py"),
            "--n-samples", str(args.n_samples),
            "--output", str(raw_csv),
        ],
        "1. Generate synthetic data",
    ):
        return 1

    # 2. Process
    rolling_window = str(pl.get("rolling_window", 12))
    if not run(
        [
            sys.executable,
            str(scripts / "processing.py"),
            "--input-data", str(raw),
            "--output-data", str(features),
            "--input-file", "sensor_data.csv",
            "--rolling-window", rolling_window,
        ],
        "2. Feature computation",
    ):
        return 1

    # 3. Train
    batch_size = str(pl.get("batch_size", 64))
    tr = cfg.get("training", {})
    lr = str(tr.get("lr", 0.0005))
    if not run(
        [
            sys.executable,
            str(scripts / "train.py"),
            "--train-data", str(features),
            "--model-dir", str(model_dir),
            "--epochs", str(args.epochs),
            "--batch-size", batch_size,
            "--lr", lr,
        ],
        "3. Train autoencoder",
    ):
        return 1

    # 4. Infer
    if not run(
        [
            sys.executable,
            str(scripts / "inference.py"),
            "--input-data", str(features),
            "--model-dir", str(model_dir),
            "--output-data", str(scores),
            "--input-file", "features.csv",
        ],
        "4. Batch inference",
    ):
        return 1

    # 5. Evaluate
    eval_subdir = data_cfg.get("evaluation_subdir", "evaluation")
    eval_out = data / eval_subdir
    eval_out.mkdir(parents=True, exist_ok=True)
    top_n = str(pl.get("top_n", 10))
    if not run(
        [
            sys.executable,
            str(scripts / "evaluate.py"),
            "--input", str(scores / "anomaly_scores.csv"),
            "--output-dir", str(eval_out),
            "--top-n", top_n,
        ],
        "5. Evaluate (summary + report)",
    ):
        return 1

    print(f"\nDone. Scores: {scores / 'anomaly_scores.csv'}")
    print(f"Report: {eval_out / 'evaluation_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
