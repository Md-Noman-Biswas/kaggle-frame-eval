# kaggle-frame-eval

A plug-and-play harness for evaluating frame selection algorithms on LongVideoBench, designed for Kaggle's free GPU tier (T4 16GB, ~30 hr/week).

## What it does

```
LongVideoBench QA (cached)
      ↓
128 candidate frames per video (cached as .npy)
      ↓
[Your Frame Selector]  ←──── plug in your algorithm here
      ↓
k=8 selected frames
      ↓
Qwen2-VL-2B (frozen judge)
      ↓
Multiple-choice accuracy
```

Match the MDP³ / AKS evaluation protocol from the ICCV 2025 papers, but small enough to iterate on Kaggle.

## Project layout

```
kaggle-frame-eval/
├── frame_selectors/        # the swappable component
│   ├── base.py             # the contract every algorithm honors
│   ├── uniform.py          # mandatory baseline
│   ├── random_sel.py       # sanity floor
│   └── siglip_topk.py      # strong literature baseline
├── vlm/
│   └── qwen2_vl.py         # frozen Qwen2-VL-2B judge
├── data/
│   └── lvb_loader.py       # reads the cached LongVideoBench
├── eval/
│   ├── runner.py           # the main loop
│   └── metrics.py          # accuracy, per-duration, per-category
└── notebooks/
    ├── 00_build_lvb_cache.ipynb   # one-time: download + decode + cache
    └── 01_run_experiment.ipynb    # rerun this for every experiment
```

## Setup on Kaggle (one-time)

### Step 1 — Get an HF token, accept LVB terms
1. Create an HF account if you don't have one.
2. Visit `https://huggingface.co/datasets/longvideobench/LongVideoBench` and accept the terms.
3. Create a read token at `https://huggingface.co/settings/tokens`.
4. On Kaggle: `Add-ons → Secrets → Add a new secret`. Name: `HF_TOKEN`. Value: your token.

### Step 2 — Upload this code as a Kaggle Dataset
1. Zip the repo (excluding `notebooks/`) and upload it to Kaggle as a new dataset named `kaggle-frame-eval`.
   Alternatively, push to a GitHub repo and `!git clone` it in the notebook's first cell.

### Step 3 — Build the LongVideoBench cache (one-time)
1. Create a new Kaggle notebook. Upload `notebooks/00_build_lvb_cache.ipynb`.
2. Enable GPU (any tier — used for headroom, not compute).
3. Run all cells. Default config caches 30 QA pairs (~5-10 unique videos). Takes ~30 min including download.
4. When it finishes: `Save Version` → wait for commit → open the version's output → click `New Dataset`. Name it `lvb-cache`.

You now have two reusable Kaggle Datasets: the code (`kaggle-frame-eval`) and the data (`lvb-cache`).

## Running an experiment

1. Create a new Kaggle notebook. Upload `notebooks/01_run_experiment.ipynb`.
2. **Attach both Kaggle Datasets** as inputs: `kaggle-frame-eval` and `lvb-cache`.
3. Enable GPU T4 x2 (or P100).
4. In cell 2, set `CODE_DIR` and `CACHE_DIR` to the mounted paths (defaults usually work).
5. In cell 3, set `SELECTOR_NAME` to one of `"uniform"`, `"random"`, `"siglip_topk"`.
6. Run all cells.

Smoke run (10 examples, ~3 minutes after the VLM finishes downloading the first time):

```python
SELECTOR_NAME = "uniform"
K_FRAMES      = 8
N_EXAMPLES    = 10
```

Dev run (100 examples, ~30 minutes):

```python
N_EXAMPLES = 100
```

## Plugging in your own algorithm

1. Create `frame_selectors/my_algo.py`:

   ```python
   from .base import FrameSelector
   import numpy as np
   from typing import List

   class MyAlgo(FrameSelector):
       name = "my_algo"

       def __init__(self, **hparams):
           # load any models/state you need
           pass

       def select(self, frames: np.ndarray, query: str, k: int) -> List[int]:
           # your logic here
           # MUST return exactly k unique sorted indices in [0, len(frames))
           ...
   ```

2. Register it in `frame_selectors/__init__.py`:

   ```python
   from .my_algo import MyAlgo
   SELECTORS["my_algo"] = MyAlgo
   ```

3. Upload a new version of the `kaggle-frame-eval` dataset. Set `SELECTOR_NAME = "my_algo"` in the experiment notebook. Rerun.

The data cache, VLM, metrics, and runner never need to change.

## Sanity checks built in

- `validate_selection(indices, n, k)` runs after every `select()` call. Catches duplicates, wrong count, out-of-range, or unsorted indices immediately.
- The runner records per-example details to a CSV so you can debug specific failures.
- `random` selector with a fixed seed gives identical picks across runs.
- The VLM uses `do_sample=False` (greedy decoding) so two runs of the same selector produce the same accuracy.

## Quota-saving tips

- The cache is built once. Don't rebuild for every experiment.
- The VLM downloads to `/root/.cache/huggingface/hub` on first use. After the first session, model loading is fast (~30s).
- Use `N_EXAMPLES=10` for smoke runs while debugging. Only go to 100+ once a selector clearly works.
- Run `random` and `uniform` once each and reuse those numbers as your baseline table — they don't change.

## What to look for

A frame selector is **working** if it consistently beats `uniform` on:
- Overall accuracy on the dev set (n=100+).
- The long-video buckets (`3m_10m`, `15m_60m`) — that's where selection matters most. Short clips (`8s_15s`) are mostly noise.

A selector beating `random` but losing to `uniform` is broken (or solving a different problem than video QA).
