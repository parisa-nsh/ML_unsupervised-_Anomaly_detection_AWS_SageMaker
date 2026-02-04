"""
Evaluation: summary statistics and report from anomaly scores.

Reads anomaly_scores.csv, computes percentiles and top anomalies, writes JSON and a single HTML report (stats + plots + table).
"""

import argparse
import base64
import io
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_html(df: pd.DataFrame, stats: dict, top_n: int) -> str:
    """Build single HTML report with stats, plots, and top-anomalies table."""
    scores = df["anomaly_score"].astype(float)
    valid = scores.dropna()
    has_valid = len(valid) > 0

    # Stats section
    stats_rows = [
        ("n_samples", stats["n_samples"]),
        ("n_valid", stats["n_valid"]),
        ("mean", f"{stats['mean']:.4f}" if stats.get("mean") is not None else "N/A"),
        ("std", f"{stats['std']:.4f}" if stats.get("std") is not None else "N/A"),
        ("min", f"{stats['min']:.4f}" if stats.get("min") is not None else "N/A"),
        ("max", f"{stats['max']:.4f}" if stats.get("max") is not None else "N/A"),
    ]
    for p, v in (stats.get("percentiles") or {}).items():
        stats_rows.append((f"p{p}", f"{v:.4f}" if v is not None else "N/A"))

    stats_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats_rows
    )

    # Plots (only if we have valid scores)
    hist_b64 = ""
    ts_b64 = ""
    if has_valid:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(valid, bins=min(50, max(10, len(valid) // 10)), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.set_title("Score distribution")
        hist_b64 = _fig_to_base64(fig)
        plt.close(fig)

        if "timestamp" in df.columns and len(valid) > 0:
            try:
                ts = pd.to_datetime(df.loc[valid.index, "timestamp"], errors="coerce")
                if ts.notna().any():
                    fig, ax = plt.subplots(figsize=(8, 2.5))
                    ax.scatter(ts, valid, alpha=0.6, s=8)
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Anomaly score")
                    ax.set_title("Scores over time")
                    plt.xticks(rotation=25)
                    ts_b64 = _fig_to_base64(fig)
                    plt.close(fig)
            except Exception:
                pass

    # Top anomalies table
    top_list = stats.get("top_anomalies", [])
    if top_list:
        cols = list(top_list[0].keys())
        th = "".join(f"<th>{c}</th>" for c in cols)
        rows = []
        for r in top_list:
            rows.append("".join(f"<td>{r.get(c, '')}</td>" for c in cols))
        table_body = "".join(f"<tr>{row}</tr>" for row in rows)
        table_html = f"<table><thead><tr>{th}</tr></thead><tbody>{table_body}</tbody></table>"
    else:
        table_html = "<p>No anomalies to list.</p>"

    plots_html = ""
    if hist_b64:
        plots_html += f'<h3>Score distribution</h3><img src="data:image/png;base64,{hist_b64}" alt="histogram"/>'
    if ts_b64:
        plots_html += f'<h3>Scores over time</h3><img src="data:image/png;base64,{ts_b64}" alt="time series"/>'
    if not has_valid:
        plots_html = "<p>No valid scores; plots skipped.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><title>Anomaly evaluation report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 900px; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.15rem; margin-top: 1.5rem; }} h3 {{ font-size: 1rem; margin-top: 1rem; }}
table {{ border-collapse: collapse; margin-top: 0.5rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: left; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; height: auto; margin-bottom: 1rem; }}
</style>
</head>
<body>
<h1>Anomaly evaluation report</h1>
<h2>Summary</h2>
<table>{stats_html}</table>
<h2>Plots</h2>
{plots_html}
<h2>Top {top_n} anomalies</h2>
{table_html}
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.environ.get("SCORES_CSV", "anomaly_scores.csv"),
        help="Path to anomaly_scores.csv (file or directory containing it)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("EVAL_OUTPUT_DIR", ""),
        help="If set, write report JSON and summary here",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top anomalies to list",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="50,90,95,99",
        help="Comma-separated percentiles to compute (e.g. 50,90,95,99)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        input_path = input_path / "anomaly_scores.csv"
    if not input_path.exists():
        print(f"Not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    if "anomaly_score" not in df.columns:
        print("Missing column: anomaly_score", file=sys.stderr)
        return 1

    scores = df["anomaly_score"].astype(float)
    valid = scores.dropna()
    if len(valid) == 0:
        print("Warning: all anomaly_score values are NaN (e.g. model not trained enough).", file=sys.stderr)

    def _f(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)

    percentiles = [int(p) for p in args.percentiles.split(",")]
    quantiles = [p / 100.0 for p in percentiles]
    values = scores.quantile(quantiles).tolist()
    stats = {
        "n_samples": int(len(scores)),
        "n_valid": int(len(valid)),
        "mean": _f(scores.mean()) if len(valid) else None,
        "std": _f(scores.std()) if len(valid) else None,
        "min": _f(scores.min()) if len(valid) else None,
        "max": _f(scores.max()) if len(valid) else None,
        "percentiles": {str(p): _f(v) for p, v in zip(percentiles, values)},
    }

    def _serialize(v):
        if hasattr(v, "item") and callable(getattr(v, "item")):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return float(v) if isinstance(v, (int, float)) else str(v)

    top_idx = scores.nlargest(args.top_n).index
    top_rows = df.loc[top_idx]
    top_list = [{k: _serialize(v) for k, v in row.items()} for _, row in top_rows.iterrows()]
    stats["top_anomalies"] = top_list

    print("Summary:", file=sys.stderr)
    print(f"  n_samples = {stats['n_samples']}, n_valid = {stats['n_valid']}", file=sys.stderr)
    mean_s = f"{stats['mean']:.6f}" if stats["mean"] is not None else "N/A"
    std_s = f"{stats['std']:.6f}" if stats["std"] is not None else "N/A"
    print(f"  mean = {mean_s}, std = {std_s}", file=sys.stderr)
    print(f"  percentiles: {stats['percentiles']}", file=sys.stderr)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Report written to {report_path}", file=sys.stderr)

        html = _build_html(df, stats, args.top_n)
        html_path = out_dir / "evaluation_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report written to {html_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
