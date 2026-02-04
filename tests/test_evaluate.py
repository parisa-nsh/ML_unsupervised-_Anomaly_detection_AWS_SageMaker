"""Tests for evaluation script (stats and report)."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# We need to run evaluate.py as main with args; its logic is in main().
# Option 1: refactor evaluate to expose a function(Path, output_dir, ...) and call that from tests.
# Option 2: run evaluate.py via subprocess with temp CSV and capture output.
# Option 3: build the CSV and patch argparse / run main with sys.argv.
# Simplest: create temp CSV files and run the script via subprocess, then check report JSON.
# Alternatively: extract the logic that computes stats from df into a function and test that.
# For minimal changes to evaluate.py, we can run it as subprocess with temp dirs.

import subprocess
import sys


@pytest.fixture
def scores_csv_valid(tmp_path):
    """Valid anomaly_scores.csv with numeric scores."""
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:02:00"],
        "machine_id": [0, 1, 0],
        "anomaly_score": [0.1, 1.5, 0.3],
    })
    path = tmp_path / "anomaly_scores.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def scores_csv_all_nan(tmp_path):
    """anomaly_scores.csv with all NaN scores."""
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00"],
        "machine_id": [0],
        "anomaly_score": [float("nan")],
    })
    path = tmp_path / "anomaly_scores.csv"
    df.to_csv(path, index=False)
    return path


def _run_evaluate(input_path, output_dir):
    """Run scripts/evaluate.py and return (returncode, report_path)."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "evaluate.py"
    report_path = Path(output_dir) / "evaluation_report.json"
    cmd = [
        sys.executable,
        str(script),
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--top-n", "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    return result.returncode, report_path


def test_evaluate_produces_report_valid(scores_csv_valid, tmp_path):
    """With valid scores, evaluate writes report with numeric stats."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    code, report_path = _run_evaluate(scores_csv_valid, out_dir)
    assert code == 0
    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
    assert report["n_samples"] == 3
    assert report["n_valid"] == 3
    assert report["mean"] is not None
    assert report["std"] is not None
    assert report["percentiles"]["50"] is not None
    assert len(report["top_anomalies"]) <= 2


def test_evaluate_handles_all_nan(scores_csv_all_nan, tmp_path):
    """With all NaN scores, report has n_valid=0 and null stats; no crash."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    code, report_path = _run_evaluate(scores_csv_all_nan, out_dir)
    assert code == 0
    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
    assert report["n_samples"] == 1
    assert report["n_valid"] == 0
    assert report["mean"] is None
    assert report["std"] is None
