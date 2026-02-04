"""Tests for feature computation (processing)."""

import pandas as pd
import pytest

import processing as proc


@pytest.fixture
def sample_df():
    """Small DataFrame with schema expected by build_features."""
    n = 20
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "machine_id": list(range(2)) * (n // 2),
        "sensor_1": range(n),
        "sensor_2": [x * 0.5 for x in range(n)],
        "sensor_3": [1.0] * n,
    })


def test_build_features_shape(sample_df):
    """Feature matrix has same number of rows as input."""
    out = proc.build_features(sample_df, ["sensor_1", "sensor_2", "sensor_3"], window=5)
    assert len(out) == len(sample_df)


def test_build_features_columns(sample_df):
    """Output includes timestamp, machine_id, and rolling/trend/change for each sensor."""
    out = proc.build_features(sample_df, ["sensor_1", "sensor_2", "sensor_3"], window=5)
    expected = [
        "timestamp", "machine_id",
        "sensor_1_rolling_mean", "sensor_1_rolling_std", "sensor_1_trend", "sensor_1_abs_change",
        "sensor_2_rolling_mean", "sensor_2_rolling_std", "sensor_2_trend", "sensor_2_abs_change",
        "sensor_3_rolling_mean", "sensor_3_rolling_std", "sensor_3_trend", "sensor_3_abs_change",
    ]
    assert list(out.columns) == expected


def test_build_features_deterministic(sample_df):
    """Same input and window give same output."""
    out1 = proc.build_features(sample_df, ["sensor_1", "sensor_2", "sensor_3"], window=5)
    out2 = proc.build_features(sample_df, ["sensor_1", "sensor_2", "sensor_3"], window=5)
    pd.testing.assert_frame_equal(out1, out2)


def test_rolling_mean_std():
    """Rolling mean and std have expected behavior."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    rmean, rstd = proc.compute_rolling_features(s, window=2)
    assert rmean.iloc[1] == 1.5
    assert rstd.iloc[0] == 0.0
    assert rstd.iloc[1] >= 0
