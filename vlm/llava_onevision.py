"""
LLaVA-OneVision-7B judge.

This is the primary baseline used in MDP^3 Table 1. Reproducing this model's
numbers gives you the strongest published comparison point.

Distributed weights expect 4-bit quantization to fit on a T4 (16 GB).

Architecture notes:
- Uses LlavaOnevisionForConditionalGeneration
- Multi-image native (each frame is a separate image in the conversation)
- Uses its own chat template via apply_chat_template
- Greedy decoding for reproducibility
"""

from typing import List, Optional
import numpy as np

from .base import VLMJudge, parse_letter, build_prompt


class LLaVAOneVisionJudge(VLMJudge):
    """LLaVA-OneVision-7B (or 0.5B for smoke tests)."""

    name = "llava_onevision"

    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        device: Optional[str] = None,
        dtype: Optional[object] = None,
        max_new_tokens: int = 8,
        use_subtitles: bool = False,
        max_subtitle_chars: int = 4000,
        load_4bit: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace model id. Recommended values:
              - "llava-hf/llava-onevision-qwen2-7b-ov-hf"      (primary, ~5 GB in 4-bit)
              - "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"    (smoke tests)
            load_4bit: Default True. 7B does NOT fit on T4 in fp16.
                Set False only on CPU or for 0.5B variant.
        """
        import torch
        from transformers import (
            LlavaOnevisionForConditionalGeneration,
            AutoProcessor,
        )

        self._torch = torch
        self._Image = __import__("PIL.Image", fromlist=["Image"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.use_subtitles = use_subtitles
        self.max_subtitle_chars = max_subtitle_chars
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)

        load_kwargs = {}
        if load_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_dtype = (dtype if dtype is not None else torch.float16) \
                if self.device == "cuda" else torch.float32
            load_kwargs["torch_dtype"] = load_dtype
            load_kwargs["device_map"] = "auto" if self.device == "cuda" else None

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        ).eval()

    def answer(
        self,
        frames: np.ndarray,
        question: str,
        options: List[str],
        subtitle_text: Optional[str] = None,
    ) -> int:
        torch = self._torch
        Image = self._Image
        with torch.no_grad():
            prompt_text = build_prompt(
                question, options, subtitle_text,
                self.use_subtitles, self.max_subtitle_chars,
            )

            # LLaVA-OneVision chat: each frame is a separate image entry.
            content = [{"type": "image"} for _ in frames]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            image_inputs = [Image.fromarray(f) for f in frames]
            inputs = self.processor(
                text=text,
                images=image_inputs,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            reply = self.processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            return parse_letter(reply, len(options))
