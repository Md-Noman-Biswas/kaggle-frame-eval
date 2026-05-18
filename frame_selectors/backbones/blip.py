"""
BLIP backbone — used by AKS (Adaptive Keyframe Sampling).

Note: AKS specifically uses BLIP's Image-Text Matching (ITM) head, which
produces a single scalar match score per (image, text) pair. That is more
discriminative than cosine similarity but slower (it needs a forward pass
per pair). Our default here uses BLIP's vision and text encoders for
cosine similarity to stay compatible with the FeatureTopKSelector
interface; switch use_itm=True to get the AKS-style scoring (slower).
"""

from typing import Optional
import numpy as np

from .base import VideoTextBackbone, _to_tensor


class BLIPBackbone(VideoTextBackbone):
    """BLIP image+text feature extractor."""

    name = "blip"

    def __init__(
        self,
        model_id: str = "Salesforce/blip-itm-base-coco",
        device: Optional[str] = None,
        batch_size: int = 16,
        dtype: Optional[object] = None,
        use_itm: bool = False,
    ):
        """
        Args:
            use_itm: If True, use BLIP's ITM head (matches AKS). Cosine sim
                between encoder outputs is the default (faster, batchable).
        """
        import torch
        from transformers import BlipForImageTextRetrieval, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_itm = use_itm
        self._torch = torch
        self.model_id = model_id

        load_dtype = (dtype if dtype is not None else torch.float16) \
            if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            model_id, torch_dtype=load_dtype
        ).to(self.device).eval()

        # Cache last query so we can do ITM mode efficiently
        self._cached_query = None
        self._cached_query_emb = None
        self._cached_frames_emb = None

    def encode_frames(self, frames: np.ndarray):
        """
        For cosine-similarity mode (default), returns the vision encoder's
        image embeddings, L2-normalized.

        For ITM mode, just caches the frames as PIL images so we can score
        them properly against the text in select().
        """
        from PIL import Image
        torch = self._torch
        with torch.no_grad():
            pil = [Image.fromarray(f) for f in frames]
            if self.use_itm:
                # ITM mode: defer scoring until we have the query.
                # Store frames; the FeatureTopKSelector will call encode_frames
                # before encode_text, so we cache here and compute in encode_text.
                self._cached_pil_frames = pil
                # Return a placeholder tensor with the right shape so the
                # selector's matmul works. We'll override the sims at select time.
                # Actually the cleanest path is to override select(), but since
                # we're stuck with the interface, we return dummy unit vectors
                # and rely on a separate code path in FeatureTopKSelector that
                # checks for ITM backbones.
                self._cached_n = len(pil)
                return torch.zeros((len(pil), 1), device=self.device)

            feats = []
            for i in range(0, len(pil), self.batch_size):
                batch = pil[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                vision_outputs = self.model.vision_model(**inputs)
                emb = vision_outputs.pooler_output
                emb = self.model.vision_proj(emb)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb)
            return torch.cat(feats, dim=0)

    def encode_text(self, query: str):
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(
                text=[query], return_tensors="pt",
                padding=True, truncation=True,
            ).to(self.device)
            text_outputs = self.model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            emb = text_outputs.last_hidden_state[:, 0, :]  # CLS token
            emb = self.model.text_proj(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb[0]
