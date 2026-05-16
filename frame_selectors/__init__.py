"""
Registry for frame selectors.

Add a new entry to SELECTORS to make it discoverable from configs.
"""

from typing import Dict, Type
from .base import FrameSelector, validate_selection
from .uniform import UniformSelector
from .random_sel import RandomSelector
from .siglip_topk import SiglipTopKSelector


SELECTORS: Dict[str, Type[FrameSelector]] = {
    "uniform": UniformSelector,
    "random": RandomSelector,
    "siglip_topk": SiglipTopKSelector,
}


def build_selector(name: str, **kwargs) -> FrameSelector:
    """Look up a selector class by name and instantiate it."""
    if name not in SELECTORS:
        raise KeyError(
            f"Unknown selector '{name}'. Available: {list(SELECTORS.keys())}"
        )
    return SELECTORS[name](**kwargs)


__all__ = [
    "FrameSelector",
    "validate_selection",
    "UniformSelector",
    "RandomSelector",
    "SiglipTopKSelector",
    "SELECTORS",
    "build_selector",
]
