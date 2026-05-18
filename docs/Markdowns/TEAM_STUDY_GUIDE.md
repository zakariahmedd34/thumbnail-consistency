# Team Study Guide

> Onboarding for two parallel teams working on the thumbnail-video consistency
> pipeline. Each team has a specific scope; the **shared layer** is what
> connects them.

---

## The teams

| Team | Owns | Headline question |
|---|---|---|
| **M1 Team** (Moment-DETR + temporal grounding) | M1 inference, features, saliency pooling | "Given a query and a video, where in time does the moment happen?" |
| **M2 Team** (Query generation) | ClipCap + VLMs + query rewriting | "Given a thumbnail image, what query best describes it?" |
| **Shared** | Scoring, eval, data, integration | "How well do M1 + M2 agree about a (thumbnail, video) pair?" |

---

## The big picture — how M1 and M2 plug together

```
                              ┌─────────────────────────────┐
                              │     M2 TEAM TERRITORY        │
                              │                              │
   Thumbnail (image)  ───────►│   M2 Generator              │
                              │   (ClipCap | Qwen2-VL |     │
                              │    LLaVA-1.6 | GT | random)  │
                              │                              │
                              └──────────────┬──────────────┘
                                             │
                                Generated Query string(s)
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │       SHARED LAYER           │
                              │   CLIP-text encoder (frozen) │
                              └──────────────┬──────────────┘
                                             │
                                       e_q (512-d)
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │     M1 TEAM TERRITORY        │
                              │                              │
   Video features (CLIP) ───►│   Moment-DETR (frozen)       │
                              │   ───────────────────         │
                              │   span regression head       │
                              │   foreground classifier      │
                              │   saliency projection head   │
                              │                              │
                              └──────────────┬──────────────┘
                                             │
                              saliency s[t] per 2-sec clip
                              + predicted spans + fg-prob
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │       SHARED LAYER           │
                              │                              │
                              │   Saliency-weighted pooling  │
                              │   e_clip = Σ_t w[t] · v[t]   │
                              │                              │
                              │   Consistency formula        │
                              │   C = α·cos(e_q, e_clip)     │
                              │     + (1-α)·cos(e_T, e_clip) │
                              │                              │
                              │   AUC / Cohen's d / figures  │
                              └─────────────────────────────┘
```

Each box is independent. M1 doesn't need to know which M2 generated the query.
M2 doesn't need to know how M1 produces saliency. They meet at the
**generated query string** and the **scoring math** in the middle.

---

# M1 TEAM — Your study list

You own everything that happens **after** the query is generated:
encoding it, retrieving moments, pooling features, contributing to the score.

## What to study (in order)

### Foundational — read these first (1-2 hours)

| # | File | Why |
|---|---|---|
| 1 | [docs/Markdowns/research/moment_detr_architecture.md](research/moment_detr_architecture.md) | Deep technical reference for Moment-DETR internals |
| 2 | [docs/Markdowns/research/m1_training_pipeline.md](research/m1_training_pipeline.md) | How M1 was trained, feature formats, expected results |
| 3 | [docs/Markdowns/research/paper_notes.md](research/paper_notes.md) | QVHighlights paper notes — important context for the dataset M1 uses |
| 4 | `Moment_DETR/moment_detr/moment_detr/model.py` | The actual transformer architecture (read it!) |
| 5 | `Moment_DETR/moment_detr/moment_detr/start_end_dataset.py` | How features + queries get loaded and padded |

### Core code you own (1 hour)

| File | What |
|---|---|
| [src/models/moment_detr.py](../../src/models/moment_detr.py) | Wrapper around the upstream model — two predictors: feature-path + raw-video |
| [src/scoring/chunking.py](../../src/scoring/chunking.py) | Splits >150s videos into 75-clip windows for M1 |
| [src/scoring/aggregate.py](../../src/scoring/aggregate.py) | Saliency-softmax pooling math — `saliency_weighted_pool()` is yours |
| `src/data/m2_dataset.py` | (skim only — owned by M2 team but uses your features) |

### Notebooks you should run end-to-end

| Notebook | Purpose |
|---|---|
| [notebooks/m1_baseline.ipynb](../../notebooks/m1_baseline.ipynb) | If extant — runs M1 on QVH subset, baseline numbers |
| [notebooks/sanity_consistency.ipynb](../../notebooks/sanity_consistency.ipynb) | R1 smoke test — exercises M1 with GT queries |
| [notebooks/final_data_collection_preparation.ipynb](../../notebooks/final_data_collection_preparation.ipynb) | Section 2.5 (feature extraction) is YOURS — make sure you understand it |

## Key concepts to master

1. **Why "feature path" vs "raw-video path"** — you almost always use feature path
   (CLIP visual features are pre-extracted to `.npz`); raw-video is for ad-hoc demos.

2. **TEF (Temporal Endpoint Features)**: `[t_start / L, t_end / L]` appended to each
   clip — gives M1 explicit position info. CLIP+TEF = 514-d input.

3. **Three output heads**:
   - `pred_spans (10, 2)` — center + width, normalized to [0, 1]
   - `pred_logits (10, 2)` — foreground (0) vs background (1) classifier
   - `saliency_scores (L_vid,)` — per-clip score, **this is what the consistency pipeline uses**

4. **Hungarian matching loss** — bipartite assignment between GT spans and the 10
   decoder queries.

5. **Saliency margin loss** — pairwise ranking, not regression. Positives must
   beat negatives by `margin=0.2`.

6. **Short-moment failure** — official paper reports 3.11% mAP on <10s moments.
   Important known limitation.

## Questions you should be able to answer (self-test)

- [ ] What's the difference between `predict_from_features()` and `predict()`?
- [ ] Why do we override `opt.device = self.device` in `MomentDETRFeaturePredictor.__init__`?
- [ ] What does the v_feat L2-normalization protect against?
- [ ] In `saliency_weighted_pool`, why use top-p mask before softmax?
- [ ] What's the chunking strategy for videos longer than 150s, and why this stride?
- [ ] If `saliency_scores` is all zeros, what does that mean?
- [ ] How do you cache M1 outputs so a re-run of the same (vid, query) is instant?

## Your acceptance tests

```python
# Test 1: M1 loads on both CPU and GPU
from src.models.moment_detr import MomentDETRFeaturePredictor
m1 = MomentDETRFeaturePredictor(ckpt_path="...", repo_root=".")
assert m1.device in ("cuda", "cpu")

# Test 2: Feature inference returns the right shape
import numpy as np
v = np.random.randn(75, 512).astype(np.float32)
preds = m1.predict_from_features(v_feat=v, t_feats=[np.random.randn(1, 512).astype(np.float32)])
assert len(preds[0]["pred_saliency_scores"]) == 75

# Test 3: Saliency-weighted pool concentrates on top-p clips
from src.scoring.aggregate import saliency_weighted_pool
v = np.zeros((10, 4)); v[5] = [1, 0, 0, 0]
s = np.array([0]*5 + [10] + [0]*4)
pool = saliency_weighted_pool(v, s, tau=0.1, top_p=0.25)
# expect pool to be very close to v[5]
assert np.dot(pool, v[5]) / (np.linalg.norm(pool) + 1e-8) > 0.99
```

---

# M2 TEAM — Your study list

You own everything that happens **before** the query reaches CLIP-text: how
thumbnails get turned into natural-language queries.

## What to study (in order)

### Foundational — read these first (1-2 hours)

| # | File | Why |
|---|---|---|
| 1 | [docs/Markdowns/research/m2_thumbnail_query_generation.md](research/m2_thumbnail_query_generation.md) | Why we picked ClipCap, vocabulary bottleneck issue, alternatives |
| 2 | [docs/Markdowns/research/current_limitations.md](research/current_limitations.md) | Especially L2 (single-frame temporal events) and L3 (vocabulary) |
| 3 | [docs/Markdowns/INTEGRATION_PLAN.md](INTEGRATION_PLAN.md) | The unified `M2Generator` interface — you'll add new generators inside this contract |
| 4 | ClipCap paper: arXiv:2111.09734 | Prefix tuning is the core idea |
| 5 | Qwen2-VL paper, LLaVA-1.6 page | Quick skim — understand they're vision-language models |

### Core code you own

| File | What |
|---|---|
| [src/m2_generators/base.py](../../src/m2_generators/base.py) | The contract every generator implements |
| [src/m2_generators/__init__.py](../../src/m2_generators/__init__.py) | Registry — decorator-based |
| [src/m2_generators/clipcap_generator.py](../../src/m2_generators/clipcap_generator.py) | Wraps the trained ClipCap |
| [src/m2_generators/qwen_vl_generator.py](../../src/m2_generators/qwen_vl_generator.py) | Qwen2-VL simple + aspect variants |
| [src/m2_generators/llava_generator.py](../../src/m2_generators/llava_generator.py) | LLaVA-1.6 aspect-then-query |
| [src/m2_generators/aspect_rewriter.py](../../src/m2_generators/aspect_rewriter.py) | Text-only post-processor wrapping any upstream |
| [src/m2_generators/ground_truth_generator.py](../../src/m2_generators/ground_truth_generator.py) | GT lookup for ceiling baseline |
| [src/m2_generators/random_generator.py](../../src/m2_generators/random_generator.py) | Random control |
| [src/models/clipcap.py](../../src/models/clipcap.py) | The actual ClipCap module (prefix MLP + frozen GPT-2) |
| [src/data/m2_dataset.py](../../src/data/m2_dataset.py) | PyTorch Dataset for ClipCap training |

### The three "starter" VLM notebooks (already studied if you wrote them)

| File | Content now lives in |
|---|---|
| `vlms/vlm_query_generation.ipynb` | `QwenVLSimpleGenerator` |
| `vlms/vlm_aspect_extraction_test.ipynb` (Qwen branch) | `QwenVLAspectGenerator` |
| `vlms/vlm_aspect_extraction_test.ipynb` (LLaVA branch) | `LLaVAAspectGenerator` |
| `vlms/aspect_extraction_pipeline.ipynb` | `aspect_rewriter.py` + `aspect_utils.py` |

### Notebooks you should run end-to-end

| Notebook | Purpose |
|---|---|
| [notebooks/train_clipcap.ipynb](../../notebooks/train_clipcap.ipynb) | Train the prefix MLP from scratch |
| [notebooks/m2_score_single.ipynb](../../notebooks/m2_score_single.ipynb) | Score one pair with any M2, see all intermediates |
| [notebooks/m2_ablate.ipynb](../../notebooks/m2_ablate.ipynb) | Multi-M2 ablation table (your headline result table) |
| [notebooks/compare_m2_generators.ipynb](../../notebooks/compare_m2_generators.ipynb) | Side-by-side qualitative figures |

## Key concepts to master

1. **What trains vs what's frozen**: M2's ClipCap *prefix MLP* trains; everything
   else (GPT-2, CLIP-image, CLIP-text) is frozen. VLMs (Qwen, LLaVA) are 100%
   frozen — they're inference-only.

2. **Prefix tuning**: ClipCap maps `(B, 512)` CLIP-image embedding to
   `(B, prefix_len, 768)` GPT-2 token embeddings. The "prefix" prepends GPT-2's
   normal input and conditions generation.

3. **Decoding strategies**:
   - **Greedy** — argmax at each step, deterministic, boring
   - **Beam search** — best for fluency, but k=3 beams gives near-paraphrases
   - **Nucleus (top-p)** — what we use for K>1: gives semantically diverse queries

4. **Aspect extraction**: Take a free-form query → parse into structured
   {living_being, action, object, location, other} → optionally re-emit as a
   normalized sentence. Useful for ablations and style normalization.

5. **Multi-frame augmentation**: For each annotated record, extract K=3 frames
   evenly spaced across `relevant_clip_ids` → 3× the training data.

6. **VRAM costs**: ClipCap ~700 MB, Qwen2-VL 7B (4-bit) ~5 GB, LLaVA-1.6 ~6 GB.
   T4 = 16 GB total; load one VLM at a time.

## Questions you should be able to answer (self-test)

- [ ] How does `ClipCapM2.generate()` produce K different queries from the same image?
- [ ] What's the difference between `qwen2vl_simple` and `qwen2vl_aspect` prompts?
- [ ] Why does `aspect_rewriter.py` need its own LLM (Qwen2.5-3B) on top of the VLM?
- [ ] What happens to consistency AUC if M2 always outputs "a man talks to camera"?
- [ ] Why is GT (ground-truth lookup) a useful baseline — it doesn't even look at the thumbnail?
- [ ] How do you add a new generator (e.g., Florence-2)? Walk me through the steps.

## Your acceptance tests

```python
# Test 1: Every registered M2 generator implements the contract
from src.m2_generators import available, build_m2
from src.m2_generators.base import M2Generator
for name in available():
    # Lightweight ones we can instantiate; heavy ones we only check class
    pass

# Test 2: GT generator returns the right query
m2 = build_m2("gt", lookup_jsonl="data/processed/consistency_eval.jsonl")
queries = m2.generate("any.jpg", k=3, qid=707)
assert all(q == queries[0] for q in queries)  # deterministic

# Test 3: Random is reproducible
m2a = build_m2("random", pool_jsonl="data/processed/consistency_eval.jsonl", seed=42)
m2b = build_m2("random", pool_jsonl="data/processed/consistency_eval.jsonl", seed=42)
assert m2a.generate("x.jpg", k=10) == m2b.generate("x.jpg", k=10)
```

---

# SHARED LAYER — Both teams must understand

This is where M1's output meets M2's output and produces the final score.
Both teams should be able to read this code; one person from each team
should own changes here.

## Files in the shared layer

| File | What it does |
|---|---|
| [src/scoring/aggregate.py](../../src/scoring/aggregate.py) | Pure-math: pool, per_chunk_score, consistency_score |
| [src/scoring/consistency.py](../../src/scoring/consistency.py) | End-to-end `ConsistencyScorer` class |
| [src/scoring/chunking.py](../../src/scoring/chunking.py) | Long-video windowing |
| [src/eval/metrics.py](../../src/eval/metrics.py) | AUC, Cohen's d, Welch's t |
| [src/models/encoders.py](../../src/models/encoders.py) | Cached CLIP / GPT-2 loaders, defensive bypass for HF version quirks |
| [src/utils/io.py](../../src/utils/io.py), [seed.py](../../src/utils/seed.py) | Plumbing |
| [tests/test_aggregate.py](../../tests/test_aggregate.py) | 18 unit tests covering the math |

## The scoring formula (memorize this)

```
e_clip_w = Σ_t softmax(s[t] / τ) · v[t]            (saliency-weighted pooled feature for chunk w)
                                                    (only top-p of clips by saliency contribute)

c(k, w)  = α · cos(e_qk, e_clip_w)  +  (1-α) · cos(e_T, e_clip_w)
            ↑ query path                ↑ thumbnail-direct path

C(T, V)  = mean_k [ max_w c(k, w) ]                 (mean over K queries, max over chunks)
```

| Symbol | What | Source |
|---|---|---|
| `e_T` | CLIP-image embedding of thumbnail | M2 team's thumbnail goes through CLIP-img |
| `e_qk` | CLIP-text embedding of generated query k | M2 team's query → CLIP-text (shared) |
| `e_clip_w` | Pooled video feature for chunk w | M1 team's saliency × video features |
| `s[t]` | Saliency at clip t | M1 team |
| `v[t]` | CLIP feature at clip t | M1 team (extracted features) |
| `α`, `τ`, `top_p` | Hyperparameters | Shared config |

## The three eval sets (everyone must know)

| Eval JSONL | Negative type | Tests |
|---|---|---|
| `consistency_eval.jsonl` | thumb_A + query_A + **video_B** (different video) | Pipeline survives easy case |
| `consistency_eval_hard.jsonl` | thumb_A + query_A + video_A, **frame OUTSIDE moment** | **Temporal grounding** ★ research claim |
| `consistency_eval_query_mismatch.jsonl` | thumb_A + **WRONG query** + video_A | **Query path** contributes |

## Cross-team checkpoints

A point where M1 and M2 must agree:

| Check | Where it lives | Why it matters |
|---|---|---|
| CLIP-text dim = M1 text-feat dim | Both teams' code | M2's CLIP-text-encoded queries flow into M1 — dims must match (both 512) |
| M1 saliency vector length = chunk length | M1's wrapper | Otherwise pooling math breaks |
| `e_clip_w` shape = `e_q` shape = `e_T` shape | All in 512 | Cosine similarity requires same dim |
| Eval JSONL `vid` field matches `clip_features/{vid}.npz` filenames | Data team / M1 | If they diverge, no features = no scores |

---

# Recommended team rituals

### Week-0 onboarding (each team, ~3 hours)

1. Read your team's "What to study" list
2. Open the team's primary notebook on Colab, Run All
3. Tweak one parameter (e.g., M1 team changes `tau=0.05`, M2 team swaps to `qwen2vl_aspect`)
4. Observe the AUC delta
5. Write a one-paragraph "what I learned" note

### Daily standup

| Round | M1 team | M2 team |
|---|---|---|
| What I did | M1 inference / saliency / features | M2 training / generation / VLM eval |
| Blocked on | M2 team's queries / features missing | M1 saliency / scorer output |
| Today | Per-team focus | Per-team focus |

### Integration sync (3× per project)

Once at the start, once after first results, once before the writeup. Topics:

- Are we feeding the right query format into CLIP-text?
- Are the three eval sets being interpreted the same way by both teams?
- Are the cached embeddings + saliency outputs aligned?

---

# Where to look first when something breaks

| Symptom | Likely owner | First place to look |
|---|---|---|
| AUC drops on a specific eval set | Shared | Check eval set integrity, then scoring formula |
| Generated queries are gibberish | M2 | ClipCap collapse / VLM parsing failure |
| Saliency is uniform across clips | M1 | M1 ckpt or `opt.device` mismatch |
| `mat1 and mat2 shapes cannot be multiplied` | Shared | CLIP encoder bypass — `src/models/encoders.py` |
| `No module named moment_detr` | M1 | `_ensure_md_on_path(repo_root)` failed |
| Random AUC ≠ 0.5 | Shared | Seed pollution or eval-set imbalance |
| Cross-video AUC > 0.95 with CLIP-only > 0.9 | Shared | Visual leak in eval — known issue, see FINAL_REPORT §4.2 |

---

# Reading order in priority

For each team, if you only have 2 hours:

### M1 team — 2-hour speedrun

1. `docs/Markdowns/research/moment_detr_architecture.md` (40 min)
2. `src/models/moment_detr.py` + `Moment_DETR/moment_detr/moment_detr/model.py` (40 min)
3. `notebooks/sanity_consistency.ipynb` Run All on Colab + read cell outputs (40 min)

### M2 team — 2-hour speedrun

1. `docs/Markdowns/research/m2_thumbnail_query_generation.md` (30 min)
2. `src/m2_generators/{base.py, clipcap_generator.py, qwen_vl_generator.py}` (40 min)
3. `notebooks/m2_ablate.ipynb` Run All with M2_LIST=['gt', 'clipcap'] (50 min)

Both teams meet at the end of week-0: each side presents one figure (their
team's notebook output) and answers 3 questions from the other team.

---

# When you're ready to contribute

| You are | First contribution should be |
|---|---|
| M1 team newcomer | Add caching for M1 saliency outputs (see INTEGRATION_PLAN §7.3) |
| M2 team newcomer | Add a new generator (Florence-2 / SmolVLM) — copy `qwen_vl_generator.py` template |
| Shared layer | Write a test in `tests/test_score_schema.py` validating the output JSON |

Pick a small task. Don't try to understand every line on day 1. The codebase
has good abstraction boundaries — you can be productive in one box (M1 or M2)
without understanding the other.
