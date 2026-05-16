"""
Random selection — the sanity floor.

Uniform should beat Random on enough samples to be statistically meaningful.
If it doesn't, the harness is broken (likely the VLM isn't sensitive to
frame quality on this dataset).
"""

from typing import List
import numpy as np
from .base import FrameSelector


class RandomSelector(FrameSelector):
    """Pick k frames at random. Seeded for reproducibility."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select(self, frames: np.ndarray, query: str, k: int) -> List[int]:
        n = len(frames)
        if k >= n:
            return list(range(n))
        idx = self.rng.choice(n, size=k, replace=False)
        return sorted(idx.tolist())
