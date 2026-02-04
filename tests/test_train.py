"""Tests for training helpers (get_feature_columns)."""

import pandas as pd
import pytest

# Import train module; conftest adds scripts to path so "model" resolves
import train as train_module


def test_get_feature_columns_excludes_ids():
    """timestamp and machine_id are excluded."""
    df = pd.DataFrame({
        "timestamp": [1, 2],
        "machine_id": [0, 1],
        "sensor_1_rolling_mean": [1.0, 2.0],
        "sensor_1_rolling_std": [0.1, 0.2],
    })
    out = train_module.get_feature_columns(df)
    assert "timestamp" not in out
    assert "machine_id" not in out
    assert "sensor_1_rolling_mean" in out
    assert "sensor_1_rolling_std" in out


def test_get_feature_columns_only_numeric():
    """Non-numeric columns are excluded."""
    df = pd.DataFrame({
        "id": [1, 2],
        "value": [1.0, 2.0],
        "label": ["a", "b"],
    })
    out = train_module.get_feature_columns(df)
    assert "value" in out
    assert "id" in out  # int is numeric
    assert "label" not in out


def test_get_feature_columns_empty():
    """DataFrame with no numeric feature columns returns empty list."""
    df = pd.DataFrame({"timestamp": [1, 2], "name": ["x", "y"]})
    out = train_module.get_feature_columns(df)
    assert out == []
