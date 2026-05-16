"""
Uniform sampling — the baseline every frame selection paper compares against.
If your algorithm cannot beat this, it doesn't work.
"""

from typing import List
import numpy as np
from .base import FrameSelector


class UniformSelector(FrameSelector):
    """Pick k frames spaced evenly across the candidate pool."""

    name = "uniform"

    def select(self, frames: np.ndarray, query: str, k: int) -> List[int]:
        n = len(frames)
        if k >= n:
            return list(range(n))
        # np.linspace gives k evenly-spaced positions including endpoints
        return np.linspace(0, n - 1, k).round().astype(int).tolist()
