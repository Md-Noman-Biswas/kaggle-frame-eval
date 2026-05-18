"""
Registry for VLM judges.

Pick a judge by short name and forward kwargs to its constructor.

Usage:
    judge = build_judge("qwen2_vl_2b")
    judge = build_judge("qwen2_vl_7b_4bit")
    judge = build_judge("llava_ov_7b_4bit", use_subtitles=True)
    judge = build_judge("llava_ov_05b")
"""

from typing import Callable, Dict

from .base import VLMJudge
from .qwen2_vl import Qwen2VLJudge
from .llava_onevision import LLaVAOneVisionJudge


# Each entry: short name -> factory closure that returns a configured judge.
# Factories accept **kwargs for things like use_subtitles, max_new_tokens, etc.
JUDGES: Dict[str, Callable[..., VLMJudge]] = {
    # Qwen2-VL family
    "qwen2_vl_2b": lambda **kw: Qwen2VLJudge(
        model_id="Qwen/Qwen2-VL-2B-Instruct", load_4bit=False, **kw
    ),
    "qwen2_vl_7b_4bit": lambda **kw: Qwen2VLJudge(
        model_id="Qwen/Qwen2-VL-7B-Instruct", load_4bit=True, **kw
    ),

    # LLaVA-OneVision family
    "llava_ov_05b": lambda **kw: LLaVAOneVisionJudge(
        model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        load_4bit=False, **kw,
    ),
    "llava_ov_7b_4bit": lambda **kw: LLaVAOneVisionJudge(
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        load_4bit=True, **kw,
    ),
}


def build_judge(name: str, **kwargs) -> VLMJudge:
    """Look up a judge by short name and instantiate it."""
    if name not in JUDGES:
        raise KeyError(
            f"Unknown judge '{name}'. Available: {list(JUDGES.keys())}"
        )
    return JUDGES[name](**kwargs)


__all__ = [
    "VLMJudge",
    "Qwen2VLJudge",
    "LLaVAOneVisionJudge",
    "JUDGES",
    "build_judge",
]
