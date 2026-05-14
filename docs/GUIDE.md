# Project Guide — How to Explore and Run the Pipeline

> If you are joining the team today and want to understand the code, run it, and find your way around: start here.

This guide assumes you have read [docs/FINAL_DESIGN.md](FINAL_DESIGN.md). If you haven't, do that first — it explains *what* we are building and *why*. This guide is about *where the code lives* and *how to run it*.

---

## 1. The two-minute orientation

```
thumbnail-consistency/
├── README.md              ← high-level intro, run order, recent updates
├── docs/
│   ├── FINAL_DESIGN.md    ← architecture + decisions
│   ├── GUIDE.md           ← (this file) how to navigate + run
│   └── research/          ← background notes
├── src/                   ← all reusable Python (import from here)
├── configs/               ← YAML hyperparameters
├── tests/                 ← pytest unit tests (run these first)
├── notebooks/             ← experiments, training, evaluation, walkthrough
├── scripts/               ← one-off utilities (notebook patcher)
└── outputs/               ← produced at runtime (gitignored)
```

**Rule of thumb:** logic lives in `src/`. Notebooks *import* from `src/` and orchestrate runs. Never duplicate logic in a notebook — if you need a function in two places, it goes in `src/`.

---

## 2. First-time setup

```powershell
# 1. Install dependencies
python -m pip install torch transformers pillow opencv-python numpy `
                     pandas matplotlib pyyaml tqdm pytest scipy

# 2. Run unit tests — should be 18/18 passing
python -m pytest tests/ -q

# 3. (Optional) Read the walkthrough notebook before touching real data
jupyter notebook notebooks/pipeline_walkthrough.ipynb
```

If the unit tests don't pass, **stop**. Something in `src/scoring/` is broken — fix that before anything else.

---

## 3. Reading order — where to start in `src/`

Read the files in this order. Each one is short (50–200 lines).

| # | File | What it does | Why read it |
|---|---|---|---|
| 1 | [src/utils/io.py](../src/utils/io.py) | JSONL / YAML load + save | Used everywhere. Trivial. |
| 2 | [src/utils/seed.py](../src/utils/seed.py) | Deterministic seeding | Trivial. |
| 3 | [src/scoring/aggregate.py](../src/scoring/aggregate.py) | Pure-math: pool, per-chunk, multi-query agg | **★ The heart of the project.** No GPU, fully testable. |
| 4 | [src/scoring/chunking.py](../src/scoring/chunking.py) | Sliding 75-clip windows for >150s videos | Tiny — read once. |
| 5 | [src/eval/metrics.py](../src/eval/metrics.py) | AUC, Cohen's d, Welch's t | Tiny — produces all reported numbers. |
| 6 | [src/models/encoders.py](../src/models/encoders.py) | Cached CLIP / GPT-2 loaders | Singleton pattern. Lazy load. |
| 7 | [src/models/clipcap.py](../src/models/clipcap.py) | ★ The trained M2 (prefix MLP + frozen GPT-2) | **★ The only trained component.** Read this once carefully. |
| 8 | [src/models/moment_detr.py](../src/models/moment_detr.py) | M1 wrapper — feature path + raw-video path | Two classes, clear API. |
| 9 | [src/scoring/consistency.py](../src/scoring/consistency.py) | End-to-end `ConsistencyScorer` | Glues everything together. |
| 10 | [src/data/m2_dataset.py](../src/data/m2_dataset.py) | PyTorch Dataset for M2 (frame, query) pairs | Used by training notebook only. |

If you have one hour: read files 3, 7, 9. Those are 80% of the project's logic.

---

## 4. The notebooks — what each one does

| Notebook | Purpose | Inputs | Outputs |
|---|---|---|---|
| [data_collection_preparation.ipynb](../notebooks/data_collection_preparation.ipynb) | Build all data artifacts | QVHighlights JSONL + raw videos | `processed/*.jsonl`, `processed/*.npz`, `qvh_frames/`, `qvh_thumbnails/` |
| [pipeline_walkthrough.ipynb](../notebooks/pipeline_walkthrough.ipynb) | ★ **Explanatory tour of every `src/` module** | Nothing (synthetic data) | Console output only |
| [m1_baseline.ipynb](../notebooks/m1_baseline.ipynb) | Sanity-check M1 official checkpoint | M1 ckpt + features | Saliency figures |
| [train_clipcap.ipynb](../notebooks/train_clipcap.ipynb) | Train M2 | `m2_train_pairs.jsonl`, `m2_val_pairs.jsonl` | `outputs/checkpoints/clipcap_best.pt`, `outputs/predictions/m2_generated.jsonl` |
| [sanity_consistency.ipynb](../notebooks/sanity_consistency.ipynb) | ★ **GT-query smoke test (R1 gate)** | M1 ckpt + `consistency_eval.jsonl` | `outputs/figures/score_distribution_gt.png` |
| [run_evaluation.ipynb](../notebooks/run_evaluation.ipynb) | Full ablation (5 query sources) | M1 ckpt + `m2_generated.jsonl` | `outputs/figures/ablation_table.csv`, `score_distribution.png` |

---

## 5. The full run order (end-to-end)

This is the sequence to produce the V1 results. Each step depends on the previous.

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1 — Read & understand                                        │
│   • docs/FINAL_DESIGN.md                                          │
│   • docs/GUIDE.md (you are here)                                  │
│   • notebooks/pipeline_walkthrough.ipynb  ← run this on CPU first │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2 — Build the data (Person C)                                │
│   notebooks/data_collection_preparation.ipynb                     │
│   Run sections 1 → 18 in order.                                   │
│   Outputs: processed/m2_train_pairs.jsonl, m2_val_pairs.jsonl,    │
│            consistency_eval.jsonl, m2_image_emb.npz, ...          │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3 — Verify M1 works (Person A)                               │
│   notebooks/m1_baseline.ipynb (Step 1 + Step 7 — sanity only)     │
│   Confirms the official checkpoint loads and produces saliency.   │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4 — ★ RISK GATE 1: GT-query smoke test (Person D)            │
│   notebooks/sanity_consistency.ipynb                              │
│   PASS if AUC ≥ 0.75. Otherwise, debug score formula BEFORE       │
│   training M2.                                                    │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5 — Train M2 (Person B)                                      │
│   notebooks/train_clipcap.ipynb                                   │
│   Trains ClipCap, early-stops on CLIP-text similarity.            │
│   Generates K=3 queries for every eval pair.                      │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 6 — Full evaluation + ablation (Person D)                    │
│   notebooks/run_evaluation.ipynb                                  │
│   Runs the 5-source ablation, produces the headline AUC table.    │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 7 — Manual gold labels + writeup (all)                       │
│   50-pair manual label split across team.                         │
│   Qualitative figures.                                            │
│   docs/FINAL_REPORT.md.                                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Outputs — where do results land?

All produced artifacts go into `outputs/`. **Never commit them** (they're gitignored).

```
outputs/
├── checkpoints/
│   └── clipcap_best.pt              # best M2 by val CLIP-text-sim
├── predictions/
│   ├── m2_generated.jsonl           # 3 queries per eval pair
│   ├── consistency_gt_query.jsonl   # GT-query scores (smoke test)
│   ├── scores_gt_query.jsonl        # ablation rows
│   ├── scores_clipcap_k3.jsonl
│   ├── scores_clipcap_k1.jsonl
│   ├── scores_thumb_only.jsonl
│   └── scores_random_query.jsonl
├── figures/
│   ├── score_distribution.png
│   ├── score_distribution_gt.png
│   ├── ablation_table.csv           # ← the headline number lives here
│   ├── saliency_qualitative_*.png
│   ├── m2_qualitative.png
│   └── frame_vs_thumbnail_gap.png
└── training_log.json                # M2 epoch-by-epoch metrics
```

Open `outputs/figures/ablation_table.csv` to see the headline AUC numbers per query source. That's the result.

---

## 7. Common failure modes — debug entry points

| Symptom | Likely cause | Where to look |
|---|---|---|
| `pytest` fails on a `cosine` or `softmax` test | Math regression in `aggregate.py` | [src/scoring/aggregate.py](../src/scoring/aggregate.py) |
| `M2` produces only `"a man walks"` for every input | Mode collapse — LR too high, or prefix_len too small | `configs/clipcap.yaml`: drop LR to 1e-4, raise prefix_len to 16 |
| Smoke-test AUC ≈ 0.5 with GT queries | Formula broken OR `vid` mismatch with feature dir | Inspect `consistency_eval.jsonl` — does `video_vid` exist in `features/clip_features/`? |
| `FileNotFoundError: features/clip_features/<vid>.npz` | Feature dir not unpacked | Run section 14 of the data notebook (it audits this) |
| `extract_frame` returns 22% success | Video files missing from `VIDEO_DIR` | Re-run section 8 (download cell) and check `download_log.json` |
| Score distribution looks flat with no separation | M1 saliency is uniform → either bad query embedding or wrong video | Print `e_clip_w` and check L2-norm; print top-3 saliency clips |
| `RuntimeError: CUDA out of memory` | Too many ClipCap forward passes batched | Drop `batch_size` in `configs/clipcap.yaml`; close other notebooks |
| `clip_score is None` everywhere | Expected after v2 patch (center-frame). | Not a bug. Filter on `frame_path` instead. |

---

## 8. The `src/` ↔ `notebooks/` contract

To keep the team productive:

```
┌─────────────────────────────────────────────────────────────────┐
│  GOOD                              BAD                          │
│  ─────                             ───                          │
│  notebook imports from src.*       notebook redefines           │
│                                    consistency_score            │
│                                                                 │
│  bugfix lands in src/              bugfix lands in 2 notebooks  │
│                                                                 │
│  configs/*.yaml for hyperparams    hardcoded numbers in cells   │
│                                                                 │
│  outputs/ for artifacts            artifacts in notebook output │
│                                    cells (huge git diffs)       │
│                                                                 │
│  one person owns src/scoring/      4 people edit aggregate.py   │
│  (avoid merge conflicts on core)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. How to add a new experiment

Worked example: "I want to ablate `top_p ∈ {0.15, 0.25, 0.5}`."

1. **Don't** edit `consistency.yaml`'s `top_p` and rerun blindly.
2. **Do** open `run_evaluation.ipynb`, copy the eval cell, and parametrize `top_p` via `ScoringConfig(top_p=p)` in a loop.
3. Save each result to `outputs/predictions/scores_topP{p}.jsonl`.
4. Add the resulting figure to `outputs/figures/topP_sensitivity.png`.

The point: **don't mutate canonical configs in place for experiments.** Write a separate cell that overrides them locally.

---

## 10. Commands cheat sheet

```powershell
# Run all unit tests
python -m pytest tests/ -q

# Run just the math tests
python -m pytest tests/test_aggregate.py::test_cosine_identical_is_one -v

# Re-apply the data-notebook patch (idempotent)
python scripts/patch_data_notebook.py

# Open a notebook
jupyter notebook notebooks/pipeline_walkthrough.ipynb

# Quick sanity import (does `src/` import cleanly?)
python -c "from src.scoring.consistency import ConsistencyScorer; print('OK')"

# Quick AUC sanity (uses only pure-numpy code)
python -c "from src.eval.metrics import roc_auc; print(roc_auc([0.9,0.8,0.1,0.2],[1,1,0,0]))"
```

---

## 11. When you get stuck

1. Re-read the relevant section of [docs/FINAL_DESIGN.md](FINAL_DESIGN.md).
2. Check whether the symptom matches §7 above.
3. Run [pipeline_walkthrough.ipynb](../notebooks/pipeline_walkthrough.ipynb) on the same machine — every step has a small synthetic example. If those work, the bug is in your data or config; if they don't, the bug is in `src/`.
4. Ask the owner of that part of the code (see [README.md §Per-role ownership](../README.md)).

---

## 12. What this guide does NOT cover

- How Moment-DETR is trained internally (see [docs/research/moment_detr_architecture.md](research/moment_detr_architecture.md))
- The QVHighlights data schema (see [docs/research/m1_training_pipeline.md](research/m1_training_pipeline.md))
- Why we picked ClipCap (see [docs/research/m2_thumbnail_query_generation.md](research/m2_thumbnail_query_generation.md))
- Strategic Q&A (see [docs/research/research_direction_guidance.md](research/research_direction_guidance.md))

Those are all in `docs/research/` — read them if you need the *why*. This guide is only the *how*.
