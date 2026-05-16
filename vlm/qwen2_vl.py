"""
Qwen2-VL-2B-Instruct as the frozen VideoLLM judge.

We feed K selected frames + question + multiple-choice options to the model
and parse the predicted letter (A, B, C, D, or E for LongVideoBench).

Why this model:
- ~4 GB in fp16, fits T4 easily with room for SigLIP loaded alongside.
- Same architecture family as Qwen2-VL-7B used in AKS (paper baseline).
- Multi-image input is native — no need to reconstruct a video file.

Notes:
- We use greedy decoding (do_sample=False) for reproducibility.
- max_new_tokens is small (8) because we only need the letter.

Torch / transformers are imported lazily so this module is safe to import
in CPU-only environments.
"""

from typing import List, Optional
import re
import numpy as np


# LongVideoBench has up to 5 options (A-E).
LETTERS = ["A", "B", "C", "D", "E"]


class Qwen2VLJudge:
    """Frozen Qwen2-VL-2B used to answer multiple-choice video QA."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        dtype: Optional[object] = None,
        max_new_tokens: int = 8,
    ):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self._torch = torch
        self._Image = __import__("PIL.Image", fromlist=["Image"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_id)
        load_dtype = (dtype if dtype is not None else torch.float16) \
            if self.device == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
            device_map="auto" if self.device == "cuda" else None,
        ).eval()

    def _build_prompt(self, question: str, options: List[str]) -> str:
        lines = [f"Question: {question}", "Options:"]
        for letter, opt in zip(LETTERS, options):
            lines.append(f"{letter}. {opt}")
        lines.append("Answer with the single letter of the correct option only.")
        return "\n".join(lines)

    def _parse_letter(self, text: str, n_options: int) -> int:
        valid = LETTERS[:n_options]
        text = text.strip().upper()
        m = re.search(r"\b([A-E])\b", text)
        if m and m.group(1) in valid:
            return LETTERS.index(m.group(1))
        if text and text[0] in valid:
            return LETTERS.index(text[0])
        return 0

    def answer(
        self,
        frames: np.ndarray,
        question: str,
        options: List[str],
    ) -> int:
        """
        Run the VLM on selected frames and return the predicted option index.

        Args:
            frames: uint8 array (K, H, W, 3) of the selected frames.
            question: question text.
            options: list of option strings (length 2-5).

        Returns:
            Integer index into `options` of the predicted answer.
        """
        torch = self._torch
        Image = self._Image
        with torch.no_grad():
            prompt_text = self._build_prompt(question, options)

            content = [
                {"type": "image", "image": Image.fromarray(f)} for f in frames
            ]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs = [Image.fromarray(f) for f in frames]
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            generated = output_ids[:, inputs.input_ids.shape[1] :]
            reply = self.processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            return self._parse_letter(reply, len(options))
