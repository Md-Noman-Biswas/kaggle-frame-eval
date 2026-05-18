"""
Registry for feature extractor backbones.

A backbone provides .encode_frames() and .encode_text(), both returning
L2-normalized embeddings. The FeatureTopKSelector consumes these.
"""

from typing import Callable, Dict

from .base import VideoTextBackbone
from .siglip import SigLIPBackbone
from .clip import CLIPBackbone
from .blip import BLIPBackbone


BACKBONES: Dict[str, Callable[..., VideoTextBackbone]] = {
    # SigLIP variants
    "siglip_base":  lambda **kw: SigLIPBackbone(
        model_id="google/siglip-base-patch16-384", **kw
    ),
    "siglip_large": lambda **kw: SigLIPBackbone(
        model_id="google/siglip-large-patch16-384", **kw
    ),

    # CLIP variants
    "clip_base":  lambda **kw: CLIPBackbone(
        model_id="openai/clip-vit-base-patch32", **kw
    ),
    "clip_large": lambda **kw: CLIPBackbone(
        model_id="openai/clip-vit-large-patch14", **kw
    ),

    # BLIP variants
    "blip_base": lambda **kw: BLIPBackbone(
        model_id="Salesforce/blip-itm-base-coco", **kw
    ),
}


def build_backbone(name: str, **kwargs) -> VideoTextBackbone:
    if name not in BACKBONES:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {list(BACKBONES.keys())}"
        )
    return BACKBONES[name](**kwargs)


__all__ = [
    "VideoTextBackbone",
    "SigLIPBackbone",
    "CLIPBackbone",
    "BLIPBackbone",
    "BACKBONES",
    "build_backbone",
]
