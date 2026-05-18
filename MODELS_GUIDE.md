# Models Guide

Quick reference for the VLM judges and feature extractor backbones that ship
with this harness.

## VLM Judges

The judge is the model that takes selected frames + a question and answers
multiple-choice. Pick one and pass its short name to `build_judge()`.

| Short name | Size | VRAM (T4) | When to use |
|---|---|---|---|
| `qwen2_vl_2b` | 2B | ~4 GB fp16 | Fast iteration, smoke tests |
| `qwen2_vl_7b_4bit` | 7B | ~5 GB 4-bit | Paper-matching, primary baseline |
| `llava_ov_05b` | 0.5B | ~1.5 GB | Smoke tests only |
| `llava_ov_7b_4bit` | 7B | ~5 GB 4-bit | **MDP³ primary baseline** — best for paper comparison |

Usage:

```python
from vlm import build_judge

# Default — what your earlier experiments used
judge = build_judge("qwen2_vl_2b")

# Paper-matching upgrade
judge = build_judge("llava_ov_7b_4bit")

# With subtitles
judge = build_judge("qwen2_vl_7b_4bit", use_subtitles=True)
```

## Feature Extractor Backbones

These power the `FeatureTopKSelector`. They produce embeddings; the selector
takes top-k by cosine similarity to the query.

| Backbone name | Size | When to use |
|---|---|---|
| `siglip_base` | ~0.9 GB | What MDP³ uses; the default |
| `siglip_large` | ~3 GB | Stronger SigLIP |
| `clip_base` | ~0.6 GB | Smaller alternative |
| `clip_large` | ~1.7 GB | What many papers use |
| `blip_base` | ~0.5 GB | What AKS uses |

Usage:

```python
from frame_selectors import FeatureTopKSelector, build_selector

# Direct construction
sel = FeatureTopKSelector(backbone="clip_large")

# Via registry — these have pre-baked names
sel = build_selector("clip_topk")      # = clip_large
sel = build_selector("blip_topk")      # = blip_base
sel = build_selector("siglip_topk")    # = siglip_base (legacy alias)
```

## Recommended experiments

### "Does my selector beat baselines?" — the core comparison

```python
JUDGE = "llava_ov_7b_4bit"      # paper baseline VLM
SELECTORS = [
    "uniform",                  # mandatory baseline
    "random",                   # sanity floor
    "siglip_topk",              # strong baseline (MDP³ ablation reference)
    "clip_topk",                # ablation: does backbone choice matter?
    "blip_topk",                # what AKS uses
    "your_algo",                # your contribution
]
```

### "Does the VLM choice change conclusions?" — VLM robustness check

Run the same selector list against multiple judges:
- `qwen2_vl_2b` (your old number)
- `qwen2_vl_7b_4bit`
- `llava_ov_7b_4bit`

If your selector beats uniform across all three, the gain is real and not
just a VLM artifact.

## What's NOT included (and why)

- **MiniCPM-V-2.6, Ovis2-8B**: trust_remote_code models with custom APIs.
  Adding them is straightforward but each one needs its own wrapper.
- **GPT-4o, Gemini**: paid APIs.
- **Video-LLaVA, LongVA, Video-XL**: older or specialized models.

Adding any of these is a single-file addition (write the class, register
it in vlm/__init__.py) following the same pattern as qwen2_vl.py and
llava_onevision.py.
