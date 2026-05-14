# Thumbnail-Video Semantic Consistency

Retrieval-grounded scoring of how well a YouTube thumbnail matches the video it represents.
Built on QVHighlights + Moment-DETR + a ClipCap-style M2.

> **Master design document:** [docs/FINAL_DESIGN.md](docs/FINAL_DESIGN.md) — read this first.
> **Status:** Day 1 scaffold landed. M2 training + consistency smoke test ready to run.
> **Timeline:** 3 days. 4 people.

---

## TL;DR

```
Thumbnail T
  ├── CLIP-img (frozen) ─────────────────────────► e_T ───┐
  └── ClipCap-M2 ★ trained ─► query q ─► CLIP-text ─► e_q │
                                                          ▼
              Moment-DETR (official ckpt, frozen) ──► saliency s[t]
                                                          │
                              saliency-softmax pool ──► e_clip_w
                                                          │
              C = mean_k max_w [0.5·cos(e_q, e_clip) + 0.5·cos(e_T, e_clip)]
```

- Only **two** components train: the M2 prefix MLP (~12M params) and (optional) a tiny consistency head.
- Everything else is frozen pretrained.
- Eval: AUC on 600 synthetic positive/hard-negative pairs + 50-pair manual gold set.

---

## What just landed (recent update)

| Date | Change | Where |
|---|---|---|
| 2026-05-14 | Master design doc | [docs/FINAL_DESIGN.md](docs/FINAL_DESIGN.md) |
| 2026-05-14 | Data notebook patched (B1–B6 bug fixes + 12 new pipeline cells) | [notebooks/data_collection_preparation.ipynb](notebooks/data_collection_preparation.ipynb) |
| 2026-05-14 | Idempotent patch script (backed up the original) | [scripts/patch_data_notebook.py](scripts/patch_data_notebook.py) |
| 2026-05-14 | `src/` package — utils / models / scoring / eval | [src/](src/) |
| 2026-05-14 | YAML configs for ClipCap / consistency / eval | [configs/](configs/) |
| 2026-05-14 | Unit tests for the scoring math (18/18 pass) | [tests/test_aggregate.py](tests/test_aggregate.py) |
| 2026-05-14 | Three new notebooks: training, smoke test, full eval | [notebooks/](notebooks/) |

---

## Repository layout

```
thumbnail-consistency/
├── README.md                                 # this file
├── docs/
│   ├── FINAL_DESIGN.md                       # ★ master design doc (read first)
│   └── research/                             # background notes (read for context)
├── configs/
│   ├── clipcap.yaml                          # M2 hyperparams
│   ├── consistency.yaml                      # α, τ, top_p, paths
│   └── eval.yaml                             # ablation source list
├── src/
│   ├── utils/
│   │   ├── io.py                             # utf-8 JSONL / YAML helpers
│   │   └── seed.py                           # deterministic seeds
│   ├── data/
│   │   └── m2_dataset.py                     # PyTorch Dataset for M2 pairs
│   ├── models/
│   │   ├── encoders.py                       # cached CLIP / GPT-2 loaders
│   │   ├── clipcap.py                        # ★ trained M2 (prefix MLP + frozen GPT-2)
│   │   └── moment_detr.py                    # M1 wrapper (feature path + raw-video path)
│   ├── scoring/
│   │   ├── aggregate.py                      # pool, per-chunk, multi-query agg
│   │   ├── chunking.py                       # >150s video chunking
│   │   └── consistency.py                    # end-to-end ConsistencyScorer
│   └── eval/
│       └── metrics.py                        # AUC, Cohen's d, Welch's t
├── tests/
│   └── test_aggregate.py                     # pure-math unit tests
├── notebooks/
│   ├── data_collection_preparation.ipynb     # data pipeline (patched)
│   ├── m1_baseline.ipynb                     # M1 sanity (existing)
│   ├── train_clipcap.ipynb                   # ★ M2 training (NEW)
│   ├── sanity_consistency.ipynb              # ★ R1 GT-query smoke test (NEW)
│   ├── run_evaluation.ipynb                  # ★ full ablation eval (NEW)
│   └── model 2/                              # legacy v1/v2/v3 (deprecated)
├── scripts/
│   └── patch_data_notebook.py                # idempotent notebook patcher
├── Moment_DETR/                              # upstream Moment-DETR code (vendored)
└── outputs/                                  # produced at runtime
    ├── checkpoints/clipcap_best.pt
    ├── predictions/*.jsonl
    └── figures/*.png
```

---

## Quick start

### 1. Install

```powershell
python -m pip install -r requirements.txt
# or, minimal:
python -m pip install torch transformers pillow opencv-python numpy pandas matplotlib pyyaml tqdm pytest scipy
```

### 2. Verify everything parses

```powershell
python -m pytest tests/test_aggregate.py -q
# expected: 18 passed
```

### 3. Run the data pipeline

Open [notebooks/data_collection_preparation.ipynb](notebooks/data_collection_preparation.ipynb) on Colab and run sections 1 → 18 in order. The patched cells produce:

```
processed/
├── m1_train_subset.jsonl          # 2000-sample QVH-format subset
├── m1_val.jsonl                   # full official val
├── m2_train_pairs.jsonl           # ~1608 (frame, query) pairs
├── m2_val_pairs.jsonl             # ~300 val pairs (early-stop)
├── consistency_eval.jsonl         # ★ 600 pos+neg pairs (PRIMARY EVAL SET)
├── thumbnail_index.jsonl          # real YouTube thumbnails (domain-gap)
├── m2_image_emb.npz               # cached CLIP-img embeddings
├── thumbnail_image_emb.npz        # cached embeddings for real thumbnails
├── category_map.json
└── download_log.json
```

### 4. Run the Day-1 evening smoke test (CRITICAL)

```
notebooks/sanity_consistency.ipynb
```

This scores the 600-pair eval set using **ground-truth QVH queries** (no M2 needed).
**Gate before Day 2:**

| Outcome | Action |
|---|---|
| AUC ≥ 0.75 | Formula works. Continue with M2 training. |
| 0.6 ≤ AUC < 0.75 | Borderline. Adjust α / τ / top_p in `configs/consistency.yaml`. |
| AUC < 0.6 | Formula broken. **Stop and debug** before training. |

### 5. Train M2

```
notebooks/train_clipcap.ipynb
```

Reads `configs/clipcap.yaml`, trains the prefix MLP on `m2_train_pairs.jsonl`, early-stops on **CLIP-text similarity** on `m2_val_pairs.jsonl`, saves to `outputs/checkpoints/clipcap_best.pt`. Also generates K=3 nucleus-sampled queries for every eval pair into `outputs/predictions/m2_generated.jsonl`.

### 6. Run full evaluation + ablation

```
notebooks/run_evaluation.ipynb
```

Computes AUC + score separation for all 5 query sources in `configs/eval.yaml`:

```
ablation_query_sources:
  - gt_query        # GT QVH query (ceiling)
  - clipcap_k3      # 3 nucleus-sampled M2 queries
  - clipcap_k1      # single beam-3 M2 query
  - thumb_only      # α=0 (no query path)
  - random_query    # control
```

Outputs `outputs/figures/ablation_table.csv` and `score_distribution.png`.

---

## Data pipeline — what changed

### Bug fixes applied to the data notebook

| ID | Cell | Fix |
|---|---|---|
| B1 | `load_jsonl`, `save_jsonl` | Explicit `encoding="utf-8"`, `ensure_ascii=False` |
| B2 | `infer_category` | Added `VLOG_KEYWORDS`, `"other"` bucket; writes `category_map.json` |
| B3 | `download_clip` | Fallback `player_client` chain (android→web→ios), stderr captured, deterministic `video_downloaded` flag |
| B4 | `extract_frame` | MSEC-based seeking (VFR-safe) |
| B5 | `extract_frame` | `clip_score` populated on both fresh-extract **and** cached paths |
| B6 | `extract_frame` | Counter reflects "frames-with-score" (not just frame existence) |

### New cells added (sections 13–18)

| Section | Output | Purpose |
|---|---|---|
| 13. Thumbnail download | `qvh_thumbnails/*.jpg` + `thumbnail_index.jsonl` | Real YouTube thumbnails via `yt-dlp --write-thumbnail` |
| 14. M1 subset builder | `m1_train_subset.jsonl`, `m1_val.jsonl` | Official QVH-schema files (M1 audit / optional retrain) |
| 15. M2 val builder | `m2_val_pairs.jsonl` | ~300 stratified val pairs for early-stopping |
| 16. Consistency eval | `consistency_eval.jsonl` | **Primary eval**: 300 positives + 300 hard negatives (CLIP-text-cos < 0.6) |
| 17. Embedding cache | `m2_image_emb.npz`, `thumbnail_image_emb.npz` | Avoid re-encoding during evaluation loop |
| 18. Manifest | Console print | Sanity checklist |

All new cells start with the sentinel `# [PIPELINE_PATCH_v1]` so re-running the patch script is idempotent.

---

## Architecture — what's frozen, what's trained

| Component | State | Params | Trainable? |
|---|---|---|---|
| CLIP-ViT-B/32 (visual) | Pretrained | 150M | ❌ Frozen |
| CLIP-text | Pretrained | 63M | ❌ Frozen |
| GPT-2 small | Pretrained | 117M | ❌ Frozen |
| SlowFast | Pre-extracted features only | — | ❌ Not used at inference |
| Moment-DETR transformer | Official checkpoint | ~8M | ❌ Frozen (we do NOT retrain) |
| **M2 prefix MLP** | **★ TRAINED** | ~12M | ✅ Yes |
| (optional V1.1) consistency head | TRAINED | ~100 | ✅ Yes (if time) |
| Aggregation module | Deterministic math | 0 | ❌ Pure functions |

> "Why don't we retrain M1?" — The official checkpoint was trained on the full 7,218-sample QVHighlights train set and achieves R1@0.5 = 53%. Retraining on our 2,000 subset would give ~45% — strictly worse. M1 is a baseline; the novelty is the consistency formulation, not the retrieval model.

---

## Scoring math (recap from FINAL_DESIGN Appendix A)

**Per-chunk pooled video feature** (`src.scoring.aggregate.saliency_weighted_pool`):

```
e_clip_w = Σ_t softmax(s[t]/τ) · v[t],   τ=0.1, top 25% by saliency only
```

**Per-(query, chunk) score** (`src.scoring.aggregate.per_chunk_score`):

```
c(k, w) = α · cos(e_qk, e_clip_w) + (1-α) · cos(e_T, e_clip_w),   α=0.5
```

**Final aggregation** (`src.scoring.aggregate.aggregate_consistency`):

```
C(T, V) = (1/K) · Σ_k max_w c(k, w)        # mean over queries, max over chunks
```

These are pure NumPy functions and are unit-tested in `tests/test_aggregate.py`.

---

## Per-role day-1 to day-3 ownership

| Role | Modules owned | Day 1 task | Day 2 task | Day 3 task |
|---|---|---|---|---|
| **A — M1** | [src/models/moment_detr.py](src/models/moment_detr.py) | Verify official ckpt loads; cache video features for eval vids | Run saliency sanity on 5 examples | Build saliency-heatmap figures (5 cases) |
| **B — M2** | [src/models/clipcap.py](src/models/clipcap.py), [src/data/m2_dataset.py](src/data/m2_dataset.py), [notebooks/train_clipcap.ipynb](notebooks/train_clipcap.ipynb) | Train ClipCap on 1608 pairs | Pick best ckpt; generate K=3 queries for eval | M2 qualitative table (10 cases) |
| **C — Data** | [notebooks/data_collection_preparation.ipynb](notebooks/data_collection_preparation.ipynb) | Run patched notebook end-to-end; produce all `processed/*` artifacts | Help with manual labeling (50 pairs) | Build frame-vs-real-thumbnail gap figure |
| **D — Scoring** | [src/scoring/](src/scoring/), [src/eval/metrics.py](src/eval/metrics.py), [notebooks/sanity_consistency.ipynb](notebooks/sanity_consistency.ipynb), [notebooks/run_evaluation.ipynb](notebooks/run_evaluation.ipynb) | **Run R1 smoke test with GT queries** | Run full ablation; produce AUC table | τ / top_p sensitivity sweep |

---

## Risk gates

```
┌─────────────────────────────────────────────────────────────┐
│ Gate 1 (end of Day 1): GT-query AUC ≥ 0.75 on 600 pairs    │
│   → If yes: proceed with M2 training                        │
│   → If no:  the score formula is broken — fix first         │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Gate 2 (mid Day 2): M2 val CLIP-text-sim ≥ 0.55             │
│   → If yes: generate K=3 queries and run full ablation      │
│   → If no:  fall back to GT-query results (still publish)   │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Gate 3 (end of Day 2): clipcap_k3 AUC ≥ random_query AUC    │
│   → Confirms M2 contributes signal beyond noise             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configs

All hyperparameters live in `configs/` — no code changes needed for ablations.

### `configs/clipcap.yaml`
- `prefix_len: 10`, `freeze_gpt2: true`
- `lr: 2e-4`, `epochs: 20`, `early_stop_patience: 3`
- Generation: `top_p=0.9, temperature=1.1, num_return_sequences=3, no_repeat_ngram_size=2`

### `configs/consistency.yaml`
- `alpha: 0.5`, `tau: 0.1`, `top_p: 0.25`
- `chunk_window: 75`, `chunk_stride: 60`
- Paths to feature dirs and the M1 checkpoint

### `configs/eval.yaml`
- The 5-source ablation list
- Output paths

---

## How to run the tests

```powershell
python -m pytest tests/ -q
```

Current status: **18/18 passing** (cosine endpoints, saliency pool top-p, multi-query aggregation, ROC-AUC, Cohen's d, chunk indexing).

---

## Where to find more

| Topic | Document |
|---|---|
| Architecture decisions + open questions | [docs/FINAL_DESIGN.md](docs/FINAL_DESIGN.md) |
| Why we picked ClipCap | [docs/research/m2_thumbnail_query_generation.md](docs/research/m2_thumbnail_query_generation.md) |
| Moment-DETR internals | [docs/research/moment_detr_architecture.md](docs/research/moment_detr_architecture.md) |
| M1 training data spec | [docs/research/m1_training_pipeline.md](docs/research/m1_training_pipeline.md) |
| Known limitations | [docs/research/current_limitations.md](docs/research/current_limitations.md) |
| Ablation plan | [docs/research/ablation_ideas.md](docs/research/ablation_ideas.md) |
| Strategic Q&A | [docs/research/research_direction_guidance.md](docs/research/research_direction_guidance.md) |

---

## Notebook backup / rollback

The data patcher creates a backup on first run:

```
notebooks/data_collection_preparation.ipynb       # patched (current)
notebooks/data_collection_preparation.ipynb.bak   # original (untouched after first patch)
```

To restore the original:

```powershell
Copy-Item notebooks/data_collection_preparation.ipynb.bak `
          notebooks/data_collection_preparation.ipynb -Force
```

To re-apply the patch (idempotent — won't duplicate cells):

```powershell
python scripts/patch_data_notebook.py
```

---

## Conventions

- All file IO uses `encoding="utf-8"` and `ensure_ascii=False`.
- All random seeds: `2026` (set via `src.utils.seed.set_seed`).
- Notebooks **import** from `src.*` — never duplicate logic.
- All shared math goes through `src.scoring.aggregate.*` (the only "single source of truth" for the score).
- All produced artifacts go to `outputs/{checkpoints,predictions,figures}/`.

---

## Open questions (resolved or pending)

See [docs/FINAL_DESIGN.md §8](docs/FINAL_DESIGN.md). Headline items:

| # | Question | Status |
|---|---|---|
| Q1 | subs_train pretraining for M2? | Pending (Day 1 spare time only) |
| Q2 | Train a learned consistency head? | Pending (Day 3 if time) |
| Q3 | Include SlowFast in M1 input? | Resolved: **No** — official ckpt is CLIP-only |
| Q4 | Manual label count: 50 vs 100? | Resolved: **50**, split across team |
| Q5 | Beam-3 vs nucleus for M2 inference? | Resolved: **nucleus default**, beam as ablation |
| Q6 | α weight (sim_qv vs sim_Tv)? | Resolved: **0.5 / 0.5**, sensitivity sweep on Day 3 |
