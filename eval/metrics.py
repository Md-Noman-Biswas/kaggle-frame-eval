"""
Metrics for video QA, matching the MDP^3 / AKS reporting protocol:
- overall accuracy
- per-duration-bucket accuracy
- per-question-category accuracy
"""

from collections import defaultdict
from typing import Dict, List
import pandas as pd


# LongVideoBench duration buckets (seconds), per the MDP^3 paper.
DURATION_BUCKETS = [
    ("8s_15s", 8, 15),
    ("15s_1m", 15, 60),
    ("3m_10m", 180, 600),
    ("15m_60m", 900, 3600),
]


def duration_bucket(seconds: float) -> str:
    """Return the bucket name for a video duration in seconds."""
    for name, lo, hi in DURATION_BUCKETS:
        if lo < seconds <= hi:
            return name
    return "other"


def compute_metrics(results: List[dict]) -> Dict[str, dict]:
    """
    Compute overall, per-duration, and per-category accuracy.

    Args:
        results: list of dicts with keys {pred_idx, correct_idx, duration,
                 question_category}.

    Returns:
        Dict with keys 'overall', 'by_duration', 'by_category'.
    """
    df = pd.DataFrame(results)
    df["correct"] = (df["pred_idx"] == df["correct_idx"]).astype(int)
    df["bucket"] = df["duration"].apply(duration_bucket)

    overall = {
        "accuracy": df["correct"].mean() if len(df) else 0.0,
        "n": len(df),
    }

    by_duration = (
        df.groupby("bucket")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .to_dict(orient="index")
    )

    by_category = (
        df.groupby("question_category")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .to_dict(orient="index")
    )

    return {
        "overall": overall,
        "by_duration": by_duration,
        "by_category": by_category,
    }


def format_metrics(metrics: Dict[str, dict]) -> str:
    """Pretty-print metrics as a string for logs."""
    lines = []
    o = metrics["overall"]
    lines.append(f"Overall accuracy: {o['accuracy']:.4f}  (n={o['n']})")

    if metrics["by_duration"]:
        lines.append("By duration:")
        for bucket, m in metrics["by_duration"].items():
            lines.append(f"  {bucket:>10s}: {m['accuracy']:.4f}  (n={m['n']})")

    if metrics["by_category"]:
        lines.append("By category:")
        for cat, m in metrics["by_category"].items():
            lines.append(f"  {cat:>20s}: {m['accuracy']:.4f}  (n={m['n']})")
    return "\n".join(lines)
