"""
The contract every frame selector must honor.

When you develop a new algorithm, subclass FrameSelector and implement
select(). The rest of the harness never needs to change.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class FrameSelector(ABC):
    """Abstract base class for frame selection algorithms."""

    name: str = "unnamed"

    @abstractmethod
    def select(
        self,
        frames: np.ndarray,
        query: str,
        k: int,
    ) -> List[int]:
        """
        Pick k frame indices from a pool of candidate frames.

        Args:
            frames: uint8 array, shape (N, H, W, 3). N candidate frames.
            query: the question text. Selectors that ignore the query
                   (uniform, random) should still accept it.
            k: number of frames to return.

        Returns:
            A list of k integer indices into `frames`, sorted ascending.
            Length must equal k. Indices must be unique and in [0, N).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


def validate_selection(indices: List[int], n_candidates: int, k: int) -> None:
    """Sanity check a selector's output. Call this in the runner."""
    if len(indices) != k:
        raise ValueError(f"Selector returned {len(indices)} indices, expected {k}")
    if len(set(indices)) != k:
        raise ValueError(f"Selector returned duplicate indices: {indices}")
    if any(i < 0 or i >= n_candidates for i in indices):
        raise ValueError(f"Selector returned out-of-range index in {indices} (n={n_candidates})")
    if list(indices) != sorted(indices):
        raise ValueError(f"Selector indices not sorted: {indices}")
