"""
CLIP backbone — the classic image+text matching model.

Use this to ablate: does the choice of feature extractor matter more than
the selection strategy? Compare CLIP-TopK vs SigLIP-TopK at the same k.
"""

from typing import Optional
import numpy as np

from .base import VideoTextBackbone, _to_tensor


class CLIPBackbone(VideoTextBackbone):
    """CLIP ViT-L/14 by default; can switch to other CLIP variants."""

    name = "clip"

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        batch_size: int = 32,
        dtype: Optional[object] = None,
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._torch = torch
        self.model_id = model_id

        load_dtype = (dtype if dtype is not None else torch.float16) \
            if self.device == "cuda" else torch.float32
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(
            model_id, torch_dtype=load_dtype
        ).to(self.device).eval()

    def encode_frames(self, frames: np.ndarray):
        from PIL import Image
        torch = self._torch
        with torch.no_grad():
            pil = [Image.fromarray(f) for f in frames]
            feats = []
            for i in range(0, len(pil), self.batch_size):
                batch = pil[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                emb = _to_tensor(self.model.get_image_features(**inputs), torch)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb)
            return torch.cat(feats, dim=0)

    def encode_text(self, query: str):
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            emb = _to_tensor(self.model.get_text_features(**inputs), torch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb[0]
