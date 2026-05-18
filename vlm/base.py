"""
Abstract interface every VLM judge must honor.

A judge takes selected frames + a multiple-choice question and returns the
predicted option index. The runner calls only the methods defined here.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

LETTERS = ["A", "B", "C", "D", "E"]


class VLMJudge(ABC):
    """All VLM judges (Qwen2-VL, LLaVA-OneVision, ...) implement this."""

    name: str = "unnamed"

    @abstractmethod
    def answer(
        self,
        frames: np.ndarray,
        question: str,
        options: List[str],
        subtitle_text: Optional[str] = None,
    ) -> int:
        """
        Run the VLM on selected frames and return the predicted option index.

        Args:
            frames: uint8 array (K, H, W, 3) of the selected frames.
            question: question text.
            options: list of option strings (length 2-5).
            subtitle_text: Optional subtitles. Only used if the judge was
                constructed with use_subtitles=True.

        Returns:
            Integer index into `options` of the predicted answer.
        """
        raise NotImplementedError


def parse_letter(text: str, n_options: int) -> int:
    """Parse a model's reply into an option index in [0, n_options)."""
    import re
    valid = LETTERS[:n_options]
    text = text.strip().upper()
    m = re.search(r"\b([A-E])\b", text)
    if m and m.group(1) in valid:
        return LETTERS.index(m.group(1))
    if text and text[0] in valid:
        return LETTERS.index(text[0])
    return 0


def build_prompt(
    question: str,
    options: List[str],
    subtitle_text: Optional[str] = None,
    use_subtitles: bool = False,
    max_subtitle_chars: int = 4000,
) -> str:
    """Shared prompt format across all judges."""
    lines = []
    if use_subtitles and subtitle_text:
        lines.append("Subtitles:")
        lines.append(subtitle_text[:max_subtitle_chars])
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Options:")
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer with the single letter of the correct option only.")
    return "\n".join(lines)
