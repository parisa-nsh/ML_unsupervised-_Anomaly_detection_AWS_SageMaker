"""Tests for synthetic data generation."""

import pandas as pd
import pytest

import generate_synthetic_data as gen


def test_schema():
    """Output has required columns: timestamp, machine_id, sensor_1, sensor_2, sensor_3."""
    df = gen.generate_synthetic_data(n_samples=50, seed=42)
    assert list(df.columns) == ["timestamp", "machine_id", "sensor_1", "sensor_2", "sensor_3"]


def test_shape():
    """Number of rows equals n_samples."""
    for n in (10, 100, 1000):
        df = gen.generate_synthetic_data(n_samples=n, seed=1)
        assert len(df) == n


def test_determinism():
    """Same seed produces identical output."""
    df1 = gen.generate_synthetic_data(n_samples=100, seed=42)
    df2 = gen.generate_synthetic_data(n_samples=100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    """Different seeds produce different output."""
    df1 = gen.generate_synthetic_data(n_samples=100, seed=1)
    df2 = gen.generate_synthetic_data(n_samples=100, seed=2)
    assert not df1["sensor_1"].equals(df2["sensor_1"])


def test_machine_id_range():
    """machine_id is in [0, n_machines - 1]."""
    df = gen.generate_synthetic_data(n_samples=20, n_machines=5, seed=0)
    assert df["machine_id"].min() >= 0
    assert df["machine_id"].max() < 5


def test_sensor_numeric():
    """Sensor columns are numeric."""
    df = gen.generate_synthetic_data(n_samples=30, seed=0)
    for col in ["sensor_1", "sensor_2", "sensor_3"]:
        assert pd.api.types.is_numeric_dtype(df[col])
