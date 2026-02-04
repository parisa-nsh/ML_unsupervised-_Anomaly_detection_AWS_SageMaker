"""
Deterministic synthetic time-series sensor data generator.

Simulates realistic industrial sensor behavior:
- Stable operating regime
- Gradual degradation (trend + variance increase)
- Failure regime with distribution shift

Output: CSV compatible with downstream SageMaker jobs.
Schema: timestamp, machine_id, sensor_1, sensor_2, sensor_3
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_timestamps(
    n_samples: int,
    start_ts: str = "2024-01-01 00:00:00",
    freq_seconds: int = 60,
) -> pd.DatetimeIndex:
    """Generate deterministic timestamp index."""
    return pd.date_range(
        start=start_ts,
        periods=n_samples,
        freq=f"{freq_seconds}s",
    )


def generate_regime(
    n: int,
    base_mean: float,
    base_std: float,
    trend: float = 0.0,
    variance_growth: float = 0.0,
    seed_offset: int = 0,
    base_seed: int = 42,
) -> np.ndarray:
    """Generate one sensor channel for a regime (stable / degradation / failure)."""
    rng = np.random.default_rng(base_seed + seed_offset)
    t = np.arange(n, dtype=float)
    mean = base_mean + trend * t
    std = base_std * (1.0 + variance_growth * t / max(n, 1))
    return mean + std * rng.standard_normal(n)


def generate_synthetic_data(
    n_samples: int = 10_000,
    n_machines: int = 5,
    stable_ratio: float = 0.5,
    degradation_ratio: float = 0.35,
    failure_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate deterministic synthetic time-series with three regimes.

    - Stable: constant mean/variance
    - Degradation: linear trend + increasing variance
    - Failure: distribution shift (mean and variance jump)
    """
    set_seed(seed)

    n_stable = int(n_samples * stable_ratio)
    n_degradation = int(n_samples * degradation_ratio)
    n_failure = n_samples - n_stable - n_degradation

    timestamps = generate_timestamps(n_samples)

    # Sensor parameters per regime (deterministic given seed)
    # Stable: normal operation
    s1_stable = generate_regime(n_stable, 100.0, 2.0, trend=0, variance_growth=0, seed_offset=0, base_seed=seed)
    s2_stable = generate_regime(n_stable, 50.0, 1.0, trend=0, variance_growth=0, seed_offset=1, base_seed=seed)
    s3_stable = generate_regime(n_stable, 25.0, 0.5, trend=0, variance_growth=0, seed_offset=2, base_seed=seed)

    # Degradation: drift + increasing variance
    s1_deg = generate_regime(n_degradation, 100.0, 2.0, trend=0.02, variance_growth=0.01, seed_offset=3, base_seed=seed)
    s2_deg = generate_regime(n_degradation, 50.0, 1.0, trend=-0.01, variance_growth=0.02, seed_offset=4, base_seed=seed)
    s3_deg = generate_regime(n_degradation, 25.0, 0.5, trend=0.005, variance_growth=0.03, seed_offset=5, base_seed=seed)

    # Failure: distribution shift
    s1_fail = generate_regime(n_failure, 120.0, 8.0, trend=0.1, variance_growth=0.05, seed_offset=6, base_seed=seed)
    s2_fail = generate_regime(n_failure, 30.0, 5.0, trend=-0.05, variance_growth=0.08, seed_offset=7, base_seed=seed)
    s3_fail = generate_regime(n_failure, 40.0, 4.0, trend=0.02, variance_growth=0.06, seed_offset=8, base_seed=seed)

    sensor_1 = np.concatenate([s1_stable, s1_deg, s1_fail])
    sensor_2 = np.concatenate([s2_stable, s2_deg, s2_fail])
    sensor_3 = np.concatenate([s3_stable, s3_deg, s3_fail])

    # Assign machine_id in round-robin for variety
    machine_id = np.arange(n_samples) % n_machines

    df = pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machine_id,
        "sensor_1": sensor_1,
        "sensor_2": sensor_2,
        "sensor_3": sensor_3,
    })

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic time-series sensor data.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(os.environ.get("N_SAMPLES", 10_000)),
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--n-machines",
        type=int,
        default=int(os.environ.get("N_MACHINES", 5)),
        help="Number of distinct machine_id values",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("RANDOM_SEED", 42)),
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("OUTPUT_PATH", "data/raw/sensor_data.csv"),
        help="Output CSV path (local or S3)",
    )
    parser.add_argument(
        "--upload-s3",
        type=str,
        default=os.environ.get("UPLOAD_S3_URI", ""),
        help="If set, upload CSV to this S3 URI (e.g. s3://bucket/prefix/raw/data.csv)",
    )
    args = parser.parse_args()

    df = generate_synthetic_data(
        n_samples=args.n_samples,
        n_machines=args.n_machines,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}", file=sys.stderr)

    if args.upload_s3:
        try:
            import boto3
            from urllib.parse import urlparse
            parsed = urlparse(args.upload_s3)
            bucket, key = parsed.netloc, parsed.path.lstrip("/")
            s3 = boto3.client("s3")
            s3.upload_file(str(output_path), bucket, key)
            print(f"Uploaded to {args.upload_s3}", file=sys.stderr)
        except Exception as e:
            print(f"Upload failed: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
