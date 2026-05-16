"""
SigLIP top-k frame selection.

This is the strongest training-free baseline in the MDP^3 ablation table.
The idea is simple: encode each frame and the query with SigLIP, then pick
the k frames whose cosine similarity to the query is highest.

Two important details:

1) We load SigLIP once and reuse it across all videos (expensive to init,
   cheap to call). The harness keeps one instance of the selector alive.

2) For long videos we batch frames through the vision encoder to avoid
   blowing up VRAM. Batch size of 32 fits comfortably on a T4 in fp16.

Imports of torch and transformers are deferred so this module is safe to
import in environments without those packages (e.g. the cache-builder
notebook).
"""

from typing import List, Optional
import numpy as np

from .base import FrameSelector


def _to_tensor(emb, torch):
    """
    Normalise the output of get_image_features / get_text_features.

    Depending on the transformers version and model variant, these methods
    may return either a plain torch.Tensor or a ModelOutput object
    (e.g. BaseModelOutputWithPooling). Extract the embedding tensor in
    either case.
    """
    if isinstance(emb, torch.Tensor):
        return emb
    # ModelOutput — the pooled embedding is in pooler_output
    if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
        return emb.pooler_output
    # Last resort: first element of the object (image_embeds, etc.)
    return emb[0]


class SiglipTopKSelector(FrameSelector):
    """Pick k frames with highest SigLIP cosine similarity to the query."""

    name = "siglip_topk"

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-384",
        device: Optional[str] = None,
        batch_size: int = 32,
        dtype: Optional[object] = None,
    ):
        # Deferred imports — only load torch when the selector is built.
        import torch
        from transformers import AutoModel, AutoProcessor
        from PIL import Image  # noqa: F401 (used in select())

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._torch = torch

        load_dtype = (dtype if dtype is not None else torch.float16) \
            if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=load_dtype
        ).to(self.device).eval()

    def _encode_frames(self, frames: np.ndarray):
        """Return L2-normalised image embeddings, shape (N, D), on self.device."""
        from PIL import Image
        torch = self._torch
        with torch.no_grad():
            pil_frames = [Image.fromarray(f) for f in frames]
            feats = []
            for i in range(0, len(pil_frames), self.batch_size):
                batch = pil_frames[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                emb = _to_tensor(self.model.get_image_features(**inputs), torch)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb)
            return torch.cat(feats, dim=0)

    def _encode_text(self, query: str):
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(
                text=[query],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)
            emb = _to_tensor(self.model.get_text_features(**inputs), torch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb[0]

    def select(self, frames: np.ndarray, query: str, k: int) -> List[int]:
        n = len(frames)
        if k >= n:
            return list(range(n))

        frame_emb = self._encode_frames(frames)        # (N, D)
        text_emb = self._encode_text(query)            # (D,)
        sims = (frame_emb @ text_emb).cpu().numpy()    # (N,)
        top_k = np.argpartition(-sims, k)[:k]
        return sorted(int(i) for i in top_k)
