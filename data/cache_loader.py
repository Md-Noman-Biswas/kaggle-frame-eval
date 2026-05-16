"""
Universal video QA cache loader.

Reads any dataset that was pre-processed into the standard cache layout:

    <cache_root>/
        metadata.parquet    # one row per QA pair — columns:
                            #   video_id, question, correct_choice (0-based int),
                            #   duration (seconds), question_category,
                            #   option0 .. option4
        frames/
            <video_id>.npy  # uint8 array (N, H, W, 3) of candidate frames

This layout is dataset-agnostic. EgoSchema, LongVideoBench, MVBench — any
dataset whose cache follows this schema is loadable without code changes.
The dataset name lives only in the cache folder, never in this file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
import numpy as np
import pandas as pd


@dataclass
class VideoQAExample:
    """One question-answer-video tuple as seen by the runner."""

    video_id: str
    question: str
    options: List[str]
    correct_idx: int        # 0-based index into options
    duration: float         # seconds (0.0 if unknown)
    question_category: str
    frames: np.ndarray      # uint8 (N, H, W, 3)


class VideoQACache:
    """
    Read-only loader over any pre-built video QA frame cache.

    Usage:
        # Load all examples
        cache = VideoQACache("/kaggle/input/egoschema-frames-cache")

        # Load a reproducible subset of 50
        cache = VideoQACache("/kaggle/input/lvb-frames-cache").subset(n=50, seed=0)

        # Iterate
        for example in cache:
            indices = selector.select(example.frames, example.question, k=8)
            ...
    """

    def __init__(
        self,
        cache_root: str,
        metadata_file: str = "metadata.parquet",
        frames_dir: str = "frames",
    ):
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            raise FileNotFoundError(f"Cache root not found: {self.cache_root}")

        meta_path = self.cache_root / metadata_file
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        self.df: pd.DataFrame = pd.read_parquet(meta_path)
        self.frames_root = self.cache_root / frames_dir

        # Drop rows whose .npy file is missing (partial cache builds).
        existing = self.df["video_id"].apply(
            lambda v: (self.frames_root / f"{v}.npy").exists()
        )
        n_missing = (~existing).sum()
        if n_missing > 0:
            print(f"[VideoQACache] {n_missing} rows dropped — frame file missing")
        self.df = self.df[existing].reset_index(drop=True)

        dataset_name = self.cache_root.name
        print(f"[VideoQACache] Loaded '{dataset_name}': {len(self.df)} examples")

    def __len__(self) -> int:
        return len(self.df)

    def subset(
        self,
        n: Optional[int] = None,
        seed: int = 0,
    ) -> "VideoQACache":
        """
        Return a new VideoQACache view restricted to n rows.

        If n is None or n >= len, returns self unchanged.
        Sampling is reproducible: same n + seed always gives the same rows.
        """
        if n is None or n >= len(self.df):
            return self
        new = VideoQACache.__new__(VideoQACache)
        new.cache_root = self.cache_root
        new.frames_root = self.frames_root
        new.df = self.df.sample(n=n, random_state=seed).reset_index(drop=True)
        return new

    def __iter__(self) -> Iterator[VideoQAExample]:
        for _, row in self.df.iterrows():
            frames = np.load(self.frames_root / f"{row['video_id']}.npy")
            options = [
                row[f"option{i}"]
                for i in range(5)
                if pd.notna(row.get(f"option{i}"))
            ]
            yield VideoQAExample(
                video_id=row["video_id"],
                question=row["question"],
                options=options,
                correct_idx=int(row["correct_choice"]),
                duration=float(row.get("duration", 0.0)),
                question_category=str(row.get("question_category", "unknown")),
                frames=frames,
            )
