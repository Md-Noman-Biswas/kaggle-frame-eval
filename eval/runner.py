"""
The main evaluation loop.

For each (video, question) pair in the dataset:
  1. Selector picks k frame indices from the candidate pool.
  2. VLM answers the multiple-choice question using those k frames.
  3. We record correctness, runtime, and metadata.

Outputs:
  - per_example.csv: one row per QA pair (for debugging)
  - summary.json: aggregated metrics
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from frame_selectors import FrameSelector, validate_selection
from vlm import VLMJudge
from data import VideoQACache
from .metrics import compute_metrics, format_metrics


def run_evaluation(
    dataset: VideoQACache,
    selector: FrameSelector,
    judge: VLMJudge,
    k: int,
    output_dir: str,
    run_name: str,
    progress: bool = True,
) -> dict:
    """
    Evaluate a selector + VLM pair on the dataset.

    Args:
        dataset: LVBCache (possibly subsetted).
        selector: a FrameSelector instance.
        judge: a VLMJudge instance (Qwen2VLJudge, LLaVAOneVisionJudge, ...).
        k: number of frames the selector should pick.
        output_dir: where to write per_example.csv and summary.json.
        run_name: short identifier for this run (used in filenames).
        progress: show tqdm bar.

    Returns:
        The metrics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    iterator = tqdm(dataset, total=len(dataset), desc=run_name) if progress else dataset

    for ex in iterator:
        # ---- selector ----
        t0 = time.perf_counter()
        indices = selector.select(ex.frames, ex.question, k)
        sel_time = time.perf_counter() - t0
        validate_selection(indices, len(ex.frames), k)

        selected = ex.frames[indices]

        # ---- VLM ----
        t1 = time.perf_counter()
        pred_idx = judge.answer(
            selected,
            ex.question,
            ex.options,
            subtitle_text=ex.subtitle_text,
        )
        vlm_time = time.perf_counter() - t1

        rows.append({
            "video_id": ex.video_id,
            "question": ex.question[:120],
            "duration": ex.duration,
            "question_category": ex.question_category,
            "correct_idx": ex.correct_idx,
            "pred_idx": pred_idx,
            "selected_indices": ",".join(map(str, indices)),
            "n_candidates": len(ex.frames),
            "selector_time_s": round(sel_time, 4),
            "vlm_time_s": round(vlm_time, 4),
        })

    # ---- save and report ----
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"per_example_{run_name}.csv", index=False)

    metrics = compute_metrics(rows)
    metrics["selector"] = selector.name
    metrics["k"] = k
    metrics["run_name"] = run_name
    metrics["n_examples"] = len(rows)
    metrics["mean_selector_time_s"] = float(df["selector_time_s"].mean()) if len(df) else 0.0
    metrics["mean_vlm_time_s"] = float(df["vlm_time_s"].mean()) if len(df) else 0.0

    with open(output_dir / f"summary_{run_name}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(format_metrics(metrics))
    print(f"Mean selector time: {metrics['mean_selector_time_s']:.3f}s")
    print(f"Mean VLM time:      {metrics['mean_vlm_time_s']:.3f}s")
    return metrics
