"""
Feature computation for time-series sensor data.

Designed to run as a SageMaker Processing Job.
Input: raw CSV from S3 (timestamp, machine_id, sensor_1, sensor_2, sensor_3).
Output: feature matrix saved to S3.

Features: rolling mean, rolling std, trend (slope), absolute change rate.
Deterministic and configurable via environment variables / arguments.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Defaults overridable by env
def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def compute_rolling_features(
    series: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """Rolling mean and rolling standard deviation."""
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
    return rolling_mean, rolling_std


def compute_trend(series: pd.Series, window: int) -> pd.Series:
    """Slope of linear trend over rolling window (per-sample slope estimate)."""
    def slope(y: np.ndarray) -> float:
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y), dtype=float)
        return np.polyfit(x, y, 1)[0]

    return series.rolling(window=window, min_periods=2).apply(slope, raw=True)


def compute_abs_change_rate(series: pd.Series) -> pd.Series:
    """Absolute change from previous value (rate of change magnitude)."""
    return series.diff().abs().fillna(0.0)


def build_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 12,
) -> pd.DataFrame:
    """
    Build feature matrix from sensor columns.
    Deterministic given same inputs and window.
    """
    out_list = []

    # Keep identifiers for joining / tracing
    if "timestamp" in df.columns:
        out_list.append(df["timestamp"])
    if "machine_id" in df.columns:
        out_list.append(df["machine_id"])

    for col in sensor_cols:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        rmean, rstd = compute_rolling_features(s, window)
        trend = compute_trend(s, window)
        abs_change = compute_abs_change_rate(s)

        out_list.append(rmean.rename(f"{col}_rolling_mean"))
        out_list.append(rstd.rename(f"{col}_rolling_std"))
        out_list.append(trend.rename(f"{col}_trend"))
        out_list.append(abs_change.rename(f"{col}_abs_change"))

    return pd.concat(out_list, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature computation for sensor time-series.")
    parser.add_argument(
        "--input-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_INPUT", "/opt/ml/processing/input"),
        help="Path to input CSV (directory or file)",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_OUTPUT", "/opt/ml/processing/output"),
        help="Path to write output feature CSV",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=_env_int("ROLLING_WINDOW", 12),
        help="Rolling window size for mean/std/trend",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=os.environ.get("INPUT_FILE", "sensor_data.csv"),
        help="Input CSV filename when input-data is a directory",
    )
    args = parser.parse_args()

    input_path = Path(args.input_data)
    if input_path.is_dir():
        input_path = input_path / args.input_file
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        sensor_cols = [c for c in ["sensor_1", "sensor_2", "sensor_3"] if c in df.columns]
    if not sensor_cols:
        print("No sensor_* columns found.", file=sys.stderr)
        return 1

    features_df = build_features(df, sensor_cols, window=args.rolling_window)

    output_path = Path(args.output_data)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "features.csv"
    features_df.to_csv(out_file, index=False)
    print(f"Wrote features shape {features_df.shape} to {out_file}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
