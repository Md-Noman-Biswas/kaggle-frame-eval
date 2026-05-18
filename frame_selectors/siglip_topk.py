"""
SigLIP top-k frame selection.

DEPRECATED: this class is now a thin wrapper around the more general
FeatureTopKSelector(backbone="siglip_base"). Kept for backward compatibility
with existing notebooks and experiment results.

For new code, use:
    from frame_selectors import FeatureTopKSelector
    sel = FeatureTopKSelector(backbone="siglip_base")
"""

from .feature_topk import FeatureTopKSelector


class SiglipTopKSelector(FeatureTopKSelector):
    """Legacy alias — equivalent to FeatureTopKSelector(backbone='siglip_base')."""

    name = "siglip_topk"

    def __init__(self, **kwargs):
        # If user passed model_id (old API), forward it as backbone_kwargs
        super().__init__(backbone="siglip_base", **kwargs)
        # Override .name so the old "siglip_topk" label appears in reports
        self.name = "siglip_topk"
