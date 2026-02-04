"""Tests for batch inference script (anomaly scoring)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

# conftest adds scripts to path
from model import Autoencoder


@pytest.fixture
def model_dir(tmp_path):
    """Minimal model dir: model.pt + metadata.json matching feature columns f1, f2."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Must match inference.py: hidden_dims=[32, 16], dropout=0.0
    model = Autoencoder(input_dim=2, encoding_dim=2, hidden_dims=[32, 16], dropout=0.0)
    torch.save(model.state_dict(), model_dir / "model.pt")
    metadata = {
        "feature_columns": ["f1", "f2"],
        "input_dim": 2,
        "encoding_dim": 2,
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return model_dir


@pytest.fixture
def features_csv(tmp_path):
    """Small features CSV with timestamp, machine_id, f1, f2."""
    path = tmp_path / "features.csv"
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:01:00"],
        "machine_id": [0, 1],
        "f1": [0.1, -0.2],
        "f2": [0.5, 0.3],
    })
    df.to_csv(path, index=False)
    return path


def _run_inference(features_path, model_dir, output_dir, repo_root, scripts_dir):
    """Run scripts/inference.py; return (returncode, path_to_anomaly_scores_csv)."""
    out_csv = Path(output_dir) / "anomaly_scores.csv"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "inference.py"),
        "--input-data", str(features_path.parent),
        "--input-file", features_path.name,
        "--model-dir", str(model_dir),
        "--output-data", str(output_dir),
    ]
    env = {**os.environ, "PYTHONPATH": str(scripts_dir)}
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), env=env)
    return result.returncode, out_csv


def test_inference_produces_scores(model_dir, features_csv, tmp_path):
    """Inference writes anomaly_scores.csv with expected columns and numeric scores."""
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    code, out_csv = _run_inference(features_csv, model_dir, out_dir, repo_root, scripts_dir)

    assert code == 0
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "anomaly_score" in df.columns
    assert len(df) == 2
    assert df["anomaly_score"].dtype in (float, "float64")
    assert (df["anomaly_score"] >= 0).all()
    if "timestamp" in df.columns:
        assert "machine_id" in df.columns


def test_inference_input_as_file(model_dir, tmp_path):
    """Inference accepts --input-data pointing to a CSV file directly."""
    features_file = tmp_path / "features.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00"],
        "machine_id": [0],
        "f1": [0.0],
        "f2": [0.0],
    }).to_csv(features_file, index=False)

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "inference.py"),
        "--input-data", str(features_file),
        "--model-dir", str(model_dir),
        "--output-data", str(out_dir),
    ]
    env = {**os.environ, "PYTHONPATH": str(scripts_dir)}
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), env=env)

    assert result.returncode == 0
    out_csv = out_dir / "anomaly_scores.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1 and "anomaly_score" in df.columns


def test_inference_fails_when_model_missing(features_csv, tmp_path):
    """Inference exits with code 1 when model.pt is missing."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text('{"feature_columns":["f1","f2"],"input_dim":2,"encoding_dim":2,"mean":[0,0],"std":[1,1]}')

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    code, _ = _run_inference(features_csv, model_dir, out_dir, repo_root, scripts_dir)
    assert code == 1


def test_inference_fails_when_metadata_missing(model_dir, features_csv, tmp_path):
    """Inference exits with code 1 when metadata.json is missing."""
    (model_dir / "metadata.json").unlink()

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    code, _ = _run_inference(features_csv, model_dir, out_dir, repo_root, scripts_dir)
    assert code == 1
