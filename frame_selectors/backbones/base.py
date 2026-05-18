"""
Abstract interface every feature extractor backbone implements.

A backbone provides two methods:
  - encode_frames(frames) -> (N, D) tensor of L2-normalized embeddings
  - encode_text(query) -> (D,) tensor of L2-normalized embedding

The FeatureTopKSelector calls these two methods and computes cosine
similarity to pick top-k frames. Adding a new feature extractor means
writing a new class that subclasses VideoTextBackbone.
"""

from abc import ABC, abstractmethod
import numpy as np


class VideoTextBackbone(ABC):
    """Image+text feature extractor (CLIP-style)."""

    name: str = "unnamed"

    @abstractmethod
    def encode_frames(self, frames: np.ndarray):
        """
        Encode a batch of frames.

        Args:
            frames: uint8 array (N, H, W, 3).

        Returns:
            Torch tensor of shape (N, D), L2-normalized, on self.device.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, query: str):
        """
        Encode the query text.

        Returns:
            Torch tensor of shape (D,), L2-normalized, on self.device.
        """
        raise NotImplementedError


def _to_tensor(emb, torch):
    """
    Normalize the output of get_image_features / get_text_features.

    Depending on the transformers version and model variant, these methods
    may return either a plain torch.Tensor or a ModelOutput object
    (e.g. BaseModelOutputWithPooling). Extract the embedding tensor either way.
    """
    if isinstance(emb, torch.Tensor):
        return emb
    if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
        return emb.pooler_output
    return emb[0]
