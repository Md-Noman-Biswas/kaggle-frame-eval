"""
Top-k selector parameterized by feature extractor backbone.

Generalization of the original SiglipTopKSelector. The selection algorithm
(top-k cosine similarity) is the same; what varies is the backbone used to
compute features. Add new backbones in frame_selectors/backbones/ without
touching this file.

Usage:
    sel = FeatureTopKSelector(backbone="siglip_base")
    sel = FeatureTopKSelector(backbone="clip_large")
    sel = FeatureTopKSelector(backbone="blip_base")

For backward compatibility, the legacy SiglipTopKSelector remains as a
thin alias — old experiments and notebooks keep working.
"""

from typing import List, Optional, Union
import numpy as np

from .base import FrameSelector
from .backbones import VideoTextBackbone, build_backbone


class FeatureTopKSelector(FrameSelector):
    """Pick k frames with highest cosine similarity to the query."""

    name = "feature_topk"

    def __init__(
        self,
        backbone: Union[str, VideoTextBackbone] = "siglip_base",
        **backbone_kwargs,
    ):
        """
        Args:
            backbone: Either a backbone name (string in BACKBONES registry)
                or an already-instantiated VideoTextBackbone.
            **backbone_kwargs: Forwarded to the backbone constructor when
                building from a name (e.g. device="cuda", batch_size=64).
        """
        if isinstance(backbone, str):
            self.backbone = build_backbone(backbone, **backbone_kwargs)
            self.name = f"feature_topk_{backbone}"
        else:
            self.backbone = backbone
            self.name = f"feature_topk_{backbone.name}"

    def select(self, frames: np.ndarray, query: str, k: int) -> List[int]:
        n = len(frames)
        if k >= n:
            return list(range(n))

        frame_emb = self.backbone.encode_frames(frames)   # (N, D)
        text_emb = self.backbone.encode_text(query)       # (D,)
        sims = (frame_emb @ text_emb).cpu().numpy()       # (N,)
        top_k = np.argpartition(-sims, k)[:k]
        return sorted(int(i) for i in top_k)
