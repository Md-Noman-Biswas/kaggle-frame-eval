"""
Registry for frame selectors.

Add new entries to SELECTORS to make them discoverable from configs.

Built-in selectors:
  uniform               — evenly-spaced indices (mandatory baseline)
  random                — random pick (sanity floor)
  siglip_topk           — legacy alias for feature_topk(backbone='siglip_base')
  clip_topk             — feature_topk with CLIP ViT-L/14
  blip_topk             — feature_topk with BLIP
  feature_topk          — generic top-k; backbone is required as a kwarg
"""

from typing import Dict, Callable

from .base import FrameSelector, validate_selection
from .uniform import UniformSelector
from .random_sel import RandomSelector
from .feature_topk import FeatureTopKSelector
from .siglip_topk import SiglipTopKSelector


# Each entry is a factory that returns a configured selector instance.
# This lets us pre-bake common backbone choices as named selectors.
SELECTORS: Dict[str, Callable[..., FrameSelector]] = {
    "uniform":      lambda **kw: UniformSelector(**kw),
    "random":       lambda **kw: RandomSelector(**kw),

    # SigLIP family
    "siglip_topk":        lambda **kw: FeatureTopKSelector(backbone="siglip_base", **kw),
    "siglip_large_topk":  lambda **kw: FeatureTopKSelector(backbone="siglip_large", **kw),

    # CLIP family
    "clip_topk":          lambda **kw: FeatureTopKSelector(backbone="clip_large", **kw),
    "clip_base_topk":     lambda **kw: FeatureTopKSelector(backbone="clip_base", **kw),

    # BLIP family
    "blip_topk":          lambda **kw: FeatureTopKSelector(backbone="blip_base", **kw),

    # Catch-all: pass backbone= explicitly
    "feature_topk":       lambda **kw: FeatureTopKSelector(**kw),
}


def build_selector(name: str, **kwargs) -> FrameSelector:
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
    "FeatureTopKSelector",
    "SiglipTopKSelector",
    "SELECTORS",
    "build_selector",
]
