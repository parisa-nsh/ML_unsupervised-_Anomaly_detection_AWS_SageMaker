"""Integration test: full local pipeline (generate → process → train → infer → evaluate)."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_full_pipeline_minimal(tmp_path):
    """Run pipeline with minimal data/epochs; check that scores and report exist."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "data"
    cmd = [
        sys.executable,
        str(repo_root / "run_pipeline.py"),
        "--n-samples", "100",
        "--epochs", "1",
        "--data-dir", str(data_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), timeout=120)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

    scores = data_dir / "scores" / "anomaly_scores.csv"
    assert scores.exists(), "anomaly_scores.csv should exist"
    report_json = data_dir / "evaluation" / "evaluation_report.json"
    report_html = data_dir / "evaluation" / "evaluation_report.html"
    assert report_json.exists(), "evaluation_report.json should exist"
    assert report_html.exists(), "evaluation_report.html should exist"
