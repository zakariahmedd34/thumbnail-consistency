# Thumbnail-Video Semantic Consistency
## Comprehensive Project Report

> Retrieval-grounded consistency scoring for YouTube thumbnails.
> Built on QVHighlights + Moment-DETR + pluggable M2 query generators
> (ClipCap, Qwen2-VL, LLaVA-1.6).

---

## 0. Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Background & Related Work](#3-background--related-work)
4. [System Architecture](#4-system-architecture)
5. [Moment-DETR (M1) — Deep Dive](#5-moment-detr-m1--deep-dive)
6. [M2 Query Generators](#6-m2-query-generators)
7. [Consistency Score Formulation](#7-consistency-score-formulation)
8. [Data Preparation](#8-data-preparation)
9. [Evaluation Methodology](#9-evaluation-methodology)
10. [Results](#10-results)
11. [Limitations & Discussion](#11-limitations--discussion)
12. [Future Work](#12-future-work)
13. [Reproducibility](#13-reproducibility)
14. [References](#14-references)

---

## 1. Executive Summary

### What problem we solve

YouTube thumbnails frequently misrepresent the videos they preview ("clickbait").
We propose a method for **automatic detection of thumbnail-video inconsistency**
that requires **no clickbait labels**: instead, we use the video's own
annotated structure (from the QVHighlights dataset) as supervision and ask
whether the thumbnail's content is temporally retrievable inside the video.

### What we built

A modular pipeline with three stages:

```
Thumbnail  ──►  M2 (query generator)  ──►  query
                                              │
                                              ▼
              CLIP-text  →  M1 (Moment-DETR)  →  per-clip saliency
                                              │
                                              ▼
                          saliency-weighted pooling
                                              │
                                              ▼
                                  consistency score C(T, V)
```

The **M2 stage** is interchangeable: we benchmark ClipCap (trained), Qwen2-VL
(zero-shot VLM), LLaVA-1.6, an aspect-extraction wrapper, ground-truth lookup
(ceiling), and random (control).

### Headline findings

| Eval | Negative type | Best M2 AUC | CLIP-only AUC | M1 contribution |
|---|---|---|---|---|
| Cross-video | random different video | 0.939 (GT) | 0.968 | -0.03 (eval is too easy) |
| **Within-video** ★ | out-of-window frame, same video | **0.638** (GT) | 0.540 | **+0.10** |
| **Query-mismatch** ★ | same frame+video, wrong query | **0.726** (GT) | 0.500 | **+0.23** |

**α-sweep on query-mismatch** (where positives and negatives share frame+video
and differ ONLY in query): α=0.0→0.497, α=0.5→0.752, **α=1.0→0.972**. This
proves the query path is the dominant signal when visual content is held
constant.

---

## 2. Problem Statement

### 2.1 The clickbait problem

A YouTube thumbnail is a 1280×720 image that previews a video. Channel owners
have economic incentives to make thumbnails maximally engaging — even when
this means depicting events that don't occur in the video. Examples:

- A thumbnail of a person with a shocked face → the video is calm
- A thumbnail with text "I LOST MY HOUSE!" → the video is about losing a hat
- A thumbnail of a dramatic explosion → the video shows none

Existing detection approaches fall into three families, all with limitations:

| Approach | Limitation |
|---|---|
| **Engagement-metric proxy** (click-through-rate, dwell time) | Post-hoc only; can't moderate before deployment |
| **Hand-labeled clickbait classifiers** | Labels are expensive, biased, language-specific |
| **Title/thumbnail text classification** | Misses purely visual misleading-ness |

### 2.2 Our formulation

We frame consistency detection as a **retrieval-grounded scoring task**:

> Given thumbnail **T** and video **V**, the consistency score **C(T, V)**
> measures whether the visual content depicted in **T** is **temporally
> retrievable** inside **V** — i.e., does **V** contain a moment whose
> CLIP-feature representation aligns with **T** and with a natural-language
> query that describes **T**?

This formulation has three properties that make it tractable:

1. **No clickbait labels required** — we use moment-retrieval annotations
   (QVHighlights) as supervision
2. **Continuous output** — `C ∈ [-1, 1]` is more informative than a binary
   clickbait/not flag
3. **Interpretable** — the score decomposes into a query-grounding component
   and a direct visual-similarity component

### 2.3 Formal task specification

Inputs:
- **T**: thumbnail image (224×224 RGB)
- **V**: video, represented by **L** CLIP-feature vectors at 2-second
  granularity (L ≈ 75 for a 150-second QVHighlights segment)

Output:
- **C(T, V) ∈ [-1, 1]**

Constraints:
- No fine-tuning of CLIP, GPT-2, or Moment-DETR (they remain frozen)
- No clickbait labels in training
- Inference must complete in seconds per pair on a T4 GPU

### 2.4 Why this matters academically

The combination of:
1. Vision-language **query generation** from images
2. **Cross-modal moment retrieval** in video
3. **Consistency scoring** as a downstream task

...is not well-studied. Most prior work treats either captioning (image →
text) or retrieval (text → moment) independently. Composing them into a
**consistency probe** is the contribution.

---

## 3. Background & Related Work

### 3.1 CLIP (Contrastive Language-Image Pretraining)

Radford et al. (2021). Vision encoder + text encoder trained on 400M
image-text pairs to produce a shared 512-d embedding space where
semantically aligned images and captions have high cosine similarity.

**Our use**:
- CLIP-image: encode the thumbnail → `e_T` (512-d, L2-normalized)
- CLIP-text: encode generated queries → `e_q` (512-d, L2-normalized)
- CLIP video features: pre-extracted per 2-sec clip → feed M1

### 3.2 Moment-DETR

Lei et al. (NeurIPS 2021). Adapts DETR (Carion et al.) from object detection
to **temporal moment localization in video**. Takes a free-form query and a
video, predicts temporal spans containing the moment + a per-clip saliency
score.

Key facts:
- 8M params, trained on QVHighlights
- Uses CLIP visual features + CLIP text features
- Outputs: 10 candidate spans + foreground probs + per-clip saliency
- Reports R1@0.5 = 53.2%, R1@0.7 = 34.0%, mAP = 30.6% on QVH val

**Our use**:
- Frozen official checkpoint — we do NOT retrain
- Feed it our generated query + video features
- Use the **saliency head** as the primary signal (more reliable than span
  regression for short moments)

### 3.3 ClipCap

Mokady et al. (arXiv 2021). Bridges CLIP and GPT-2 by training only a small
prefix MLP that maps CLIP image embeddings to GPT-2's token embedding space.
The frozen GPT-2 then continues generation conditioned on this prefix,
producing captions for arbitrary images.

**Our use**:
- One of our M2 generators
- ~12M trainable params (the prefix MLP only)
- Frozen GPT-2 + frozen CLIP-image
- Trained on ~9,000 (frame, query) pairs from QVHighlights multi-frame
  augmentation

### 3.4 Vision-Language Models (VLMs)

A newer family of models (Qwen2-VL, LLaVA-1.6, Florence-2) that fuse a
vision encoder with a large language model (3B-70B params), trained on
multi-modal instruction data.

**Properties**:
- Zero-shot prompting (no task-specific training)
- World-knowledge priors from LLM pretraining
- Higher computational cost (~5-7s per image on T4)
- Higher VRAM (~5-6 GB for 7B models in 4-bit)

**Our use**:
- Qwen2-VL 7B and LLaVA-1.6 Mistral 7B as alternative M2 generators
- 4-bit quantization (bitsandbytes nf4) for T4 compatibility

### 3.5 QVHighlights dataset

Lei et al. (NeurIPS 2021). 10,148 videos with:
- Natural-language queries
- Annotated relevant time windows
- Per-clip saliency scores (3 annotators × {0,1,2,3,4})

**Our use**:
- Source of training and evaluation data
- We use the videos, queries, and saliency annotations
- We construct three evaluation sets from the structure (Section 9.2)

---

## 4. System Architecture

### 4.1 End-to-end pipeline

```
                ┌──────────────────────────────────────────────────┐
                │                  INPUT                            │
                │  Thumbnail T   (224×224 RGB)                      │
                │  Video V       (pre-extracted CLIP features:      │
                │                 75 × 512, L2-normalized, 2s/clip) │
                └──────────────────────────┬───────────────────────┘
                                           │
                  ┌────────────────────────┴────────────────────────┐
                  │                                                  │
                  ▼ T                                                ▼ T
        ┌───────────────────┐                          ┌─────────────────────┐
        │ CLIP-image (frozen)│                          │   M2 Generator       │
        │ ViT-B/32           │                          │  (pluggable):        │
        │ → e_T (512-d)      │                          │  - ClipCap (trained) │
        └────────┬───────────┘                          │  - Qwen2-VL 7B       │
                 │                                       │  - LLaVA-1.6 7B      │
                 │  e_T                                  │  - aspect_rewriter   │
                 │                                       │  - GT lookup         │
                 │                                       │  - Random control    │
                 │                                       └──────────┬───────────┘
                 │                                                  │
                 │                                                  │  K = 3 queries
                 │                                                  ▼
                 │                                       ┌─────────────────────┐
                 │                                       │ CLIP-text (frozen)   │
                 │                                       │ → e_q1, e_q2, e_q3   │
                 │                                       │   (each 512-d)       │
                 │                                       └──────────┬───────────┘
                 │                                                  │
                 │                                                  │
                 │                                                  ▼
                 │                                       ┌─────────────────────┐
                 │  V (clip features)─────────────────►  │ Moment-DETR (frozen)│
                 │                                       │ For each (q_k, V):   │
                 │                                       │  → saliency s_k[t]  │
                 │                                       │  → pred_spans       │
                 │                                       │  → fg_prob          │
                 │                                       └──────────┬───────────┘
                 │                                                  │
                 │                                       ▼ saliency-weighted pool
                 │                                       e_clip_w = Σ_t softmax(s/τ)·v[t]
                 │                                                  │
                 │                                                  │
                 └──────────────────────────────┐                   │
                                                ▼                   ▼
                                       ┌──────────────────────────────────────┐
                                       │  Per-query / per-chunk score          │
                                       │  c(k, w) = α·cos(e_qk, e_clip_w)      │
                                       │          + (1-α)·cos(e_T, e_clip_w)   │
                                       └──────────────┬───────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────────────┐
                                       │  Aggregation                          │
                                       │  C(T, V) = mean_k max_w c(k, w)       │
                                       └──────────────┬───────────────────────┘
                                                      │
                                                      ▼
                                       Final consistency score C ∈ [-1, 1]
```

### 4.2 What trains vs what's frozen

| Component | State | Params | Why |
|---|---|---|---|
| CLIP-ViT-B/32 visual | Pretrained, **frozen** | 150M | Web-scale alignment already learned |
| CLIP-text | Pretrained, **frozen** | 63M | Same reason |
| GPT-2 small | Pretrained, **frozen** | 117M | English language model |
| Moment-DETR | Pretrained on full QVH, **frozen** | 8M | Retraining on 2K subset would degrade — official ckpt is the strongest baseline |
| Qwen2-VL 7B / LLaVA-1.6 7B | Pretrained, **frozen** | 7B each | Prompted, not fine-tuned |
| **M2 ClipCap prefix MLP** | **★ TRAINED** | ~12M | Cross-modal alignment from CLIP space to GPT-2 prefix space |

**Total trainable parameters in V1: ~12M** (M2 prefix MLP only).

This makes the project a **method paper**, not a model paper: the contribution
is the formulation and pipeline, not a new architecture.

### 4.3 Pluggable M2 — the key abstraction

Every M2 generator implements the same interface:

```python
class M2Generator(ABC):
    @abstractmethod
    def generate(self, thumbnail, k=1) -> List[str]:
        """Return K query strings for the thumbnail."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier."""
```

A registry pattern allows runtime selection:

```python
from src.m2_generators import build_m2
m2 = build_m2("qwen2vl_simple")        # or 'clipcap', 'llava16_aspect', etc.
queries = m2.generate(thumbnail, k=3)   # interchangeable
```

The downstream scoring code is generator-agnostic — only the M2 changes
across experiments.

---

## 5. Moment-DETR (M1) — Deep Dive

### 5.1 Why Moment-DETR fits

The thumbnail-consistency task needs: "given a query, where in the video is
this content?" That's exactly what moment retrieval is. Among alternatives:

| Model | Pros | Cons | Verdict |
|---|---|---|---|
| 2D-TAN | Strong on long moments | Anchor-based; needs span enumeration | Slower |
| VSLNet | Span-prediction-based | Simpler but lower accuracy | Weaker |
| **Moment-DETR** | DETR-style set prediction; saliency head; SOTA when published | Short-moment localization weak (3% mAP <10s) | ★ Chosen |

### 5.2 Architecture diagram

```
Video features  (L, 512)   ──┐
+ TEF (L, 2)                 │
                             ├──► Input projection (linear)
Query features (Q, 512)    ──┘    each modality → 256-d
                                            │
                                            ▼
                                  ┌──────────────────────┐
                                  │  Transformer Encoder  │ (2 layers, 8 heads)
                                  │  self-attention over  │
                                  │  concatenated [V; Q]  │
                                  └────────────┬─────────┘
                                               │ memory (V+Q tokens, 256-d)
                                               ▼
                              ┌────────────────┴────────────────┐
                              │                                  │
                              ▼                                  ▼
                  ┌───────────────────────┐         ┌──────────────────────┐
                  │  Transformer Decoder   │         │  Saliency projection │
                  │  10 learnable queries  │         │  Linear(256, 1)       │
                  │  cross-attend to       │         │  on video positions   │
                  │  encoder memory        │         │  → s[t] for t=1..L    │
                  │  (2 layers, 8 heads)   │         └──────────────────────┘
                  └────────────┬───────────┘
                               │ hs (10, 256)
                               │
                       ┌───────┴────────┐
                       ▼                ▼
              ┌────────────────┐  ┌───────────────┐
              │ span_embed MLP │  │ class_embed   │
              │ → (10, 2)      │  │ Linear(256, 2)│
              │ (center, width)│  │ → fg/bg logits│
              └────────────────┘  └───────────────┘
```

### 5.3 Inputs and dimensions

| Component | Dimension | Source |
|---|---|---|
| Video CLIP features | (L_vid, 512), L2-norm | Pre-extracted, 1 per 2-sec clip |
| Temporal Endpoint Features (TEF) | (L_vid, 2) = [t_start/L, t_end/L] | Computed on-the-fly |
| Video input to encoder | (L_vid, 514) | Concatenation, L_vid ≤ 75 |
| Query CLIP text features | (L_q, 512), L2-norm | Pre-extracted per-token, L_q ≤ 32 |
| Internal hidden dim | 256 | After input projection |
| Decoder queries | (10, 256) | Learned at training time |

### 5.4 Three output heads — and why we use only one

**Head 1: span_embed (MLP, depth=3)**

```python
pred_spans (10, 2)  # (center_normalized, width_normalized) in [0, 1]
```

Used by paper for the R1@0.5 / R1@0.7 metrics. We **don't** use this directly
because it's unstable for short moments (3.1% mAP on <10s clips).

**Head 2: class_embed (Linear)**

```python
pred_logits (10, 2)  # foreground / background classifier
```

We use this as an **optional confidence damping** factor in the consistency
score (currently disabled in V1).

**Head 3: saliency_proj (Linear)** ★ THE ONE WE USE

```python
saliency_scores (L_vid,)  # one score per 2-second clip
```

This is what feeds into our saliency-weighted pooling. It's the more reliable
signal because:

1. It's **per-clip**, so it has natural temporal resolution (no span regression
   instability)
2. It was trained with a **margin ranking loss**, which is robust to label noise
3. It's **query-conditioned** — different queries on the same video produce
   different saliency vectors

### 5.5 Loss functions during M1 training (for reference)

| Loss | Weight | What it teaches |
|---|---|---|
| L1 span | 10 | Predicted center/width close to GT |
| GIoU span | 1 | Overlap-aware regression |
| Class CE | 4 | Foreground / background |
| Saliency margin | 1 | Positives score higher than negatives by margin=0.2 |

We don't retrain M1, but it helps to know what supervision baked these heads.

### 5.6 What gets fed in (concretely)

For each (query, video) pair our scorer evaluates:

```python
# Inputs to M1
src_vid       : (1, n_clips, 514)   # video features + TEF
src_vid_mask  : (1, n_clips)         # all ones for valid clips
src_txt       : (1, 1, 512)          # the query's CLIP-text embedding, broadcast
src_txt_mask  : (1, 1)               # single token

# Outputs we use
outputs['saliency_scores']  : (1, n_clips)   # ★ this is everything
outputs['pred_logits']      : (1, 10, 2)     # used for optional damping
```

### 5.7 Known M1 weaknesses (limitations §11)

- **Short-moment failure**: mAP 3.11% on <10s moments
- **Saliency supervision noise**: 3 annotators often disagree
- **Fixed 10 queries**: hard ceiling on simultaneous moments
- **No cross-modal pretraining**: trained from scratch on QVH only

---

## 6. M2 Query Generators

We benchmark **six** interchangeable M2 strategies, plus two baselines.

### 6.1 The seven approaches

```
                          M2 Generator Family
                          ───────────────────

   ┌─ Trained on QVH ─────────────────────────────┐
   │                                              │
   │  ClipCap (★ V1's only trained component)     │
   │  ─ CLIP-image → prefix MLP → frozen GPT-2    │
   │  ─ 12M trainable params                      │
   │  ─ ~50 ms/image, 700 MB VRAM                 │
   │                                              │
   └──────────────────────────────────────────────┘

   ┌─ Zero-shot VLMs (4-bit) ──────────────────────┐
   │                                               │
   │  Qwen2-VL-simple                              │
   │  ─ Single sentence prompt                     │
   │  ─ Qwen2-VL 7B (4-bit), ~5 GB VRAM            │
   │  ─ ~4 sec/image                               │
   │                                               │
   │  Qwen2-VL-aspect                              │
   │  ─ Aspect-extraction-then-query prompt        │
   │  ─ Same model, richer output                  │
   │  ─ ~6 sec/image                               │
   │                                               │
   │  LLaVA-1.6-aspect                             │
   │  ─ Same prompt, different visual backbone     │
   │  ─ LLaVA-1.6 Mistral 7B (4-bit), ~6 GB VRAM   │
   │  ─ ~6 sec/image                               │
   │                                               │
   └───────────────────────────────────────────────┘

   ┌─ Composite / post-processing ─────────────────┐
   │                                               │
   │  AspectRewriter(upstream)                     │
   │  ─ Runs Qwen2.5-3B text-only on upstream's    │
   │    query → extract aspects → reconstruct      │
   │  ─ +1 sec, +3 GB on top of upstream           │
   │                                               │
   └───────────────────────────────────────────────┘

   ┌─ Baselines / controls ────────────────────────┐
   │                                               │
   │  GT (ground-truth lookup)                     │
   │  ─ Returns the human-annotated query          │
   │  ─ CPU-only, ~0 ms                            │
   │  ─ ★ Ceiling baseline                         │
   │                                               │
   │  Random (control)                             │
   │  ─ Returns a query from a pool                │
   │  ─ AUC should be ~0.5                         │
   │                                               │
   └───────────────────────────────────────────────┘
```

### 6.2 ClipCap — the trained component (deep dive)

ClipCap is the one place where learning happens in this project. Architecture:

```
   CLIP image features  e_T  (B, 512)
                        │
                        ▼
          ┌──────────────────────────────┐
          │  Prefix MLP (★ TRAINED)       │
          │  Linear(512, 768·prefix_len)  │
          │  Tanh                          │
          │  Linear(768·prefix_len, ...)   │
          └──────────────┬───────────────┘
                         │ (B, prefix_len, 768)
                         │
                         ▼
                  prepend to token embs
                         │
                         ▼
          ┌──────────────────────────────┐
          │  GPT-2 small (frozen)         │
          │  117M params, no fine-tune    │
          │  generates next-token logits  │
          └──────────────┬───────────────┘
                         │
                         ▼
                  Query string
```

**Training data**: ~9,000 (frame, query) pairs from K=3 multi-frame
augmentation. Each video contributes 3 frames spread across its relevant_window.

**Training loss**: cross-entropy on GT query tokens, label-smoothing 0.05.

**Why this is the right size**:
- Too few params → underfits
- Too many params → would need much more data (>100K pairs)
- 12M prefix MLP fits on T4 in batch 32 + gradient accumulation
- Trains in ~3 hours

**Inference**:
- K=3 nucleus-sampled queries per thumbnail
- top_p=0.9, temperature=1.1, no_repeat_ngram_size=2

### 6.3 VLM approaches — zero-shot, no training

For Qwen2-VL and LLaVA-1.6 we just send a structured prompt and parse the
response. Two prompt strategies:

**Simple prompt (Qwen2-VL-simple)**:
```
Describe what is happening in this image in one short sentence.
Start with a subject. Only describe what is visually present.
```
→ Output: single query string.

**Aspect prompt (Qwen2-VL-aspect, LLaVA-1.6-aspect)**:
```
Step 1 - Extract aspects: living_being, action, object, location, other
Step 2 - Write one concise query from those aspects

<aspects>{"living_being": [...], ...}</aspects>
<query>YOUR QUERY HERE</query>
```
→ Output: structured aspects dict + query.

The aspect prompt is more interpretable (we can see what the VLM "saw") and
enables an ablation: how much does the structured intermediate help?

### 6.4 Aspect Rewriter — composite generator

Wraps any upstream M2 (e.g. Qwen2-VL-simple) and post-processes its output:

```
Upstream M2 → "a man in a red hoodie stretches on a sidewalk"
        ↓
Qwen2.5-3B  → {living_being: ["man"], action: ["stretches"],
                object: [], location: ["sidewalk"],
                other: ["red", "hoodie"]}
        ↓
Reconstructor → "man stretches in sidewalk with red, hoodie"
```

**Hypothesis**: normalizing the style across M2s reduces variance and may help
the consistency formula. Whether it does is an empirical question for the
ablation table.

### 6.5 Generator comparison

| | ClipCap | Qwen2-VL-simple | Qwen2-VL-aspect | LLaVA-1.6-aspect | GT | Random |
|---|---|---|---|---|---|---|
| Reads thumb? | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Reads video? | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Trained? | ✅ (prefix MLP) | ❌ | ❌ | ❌ | ❌ | ❌ |
| GPU req? | ✅ (T4 ok) | ✅ (T4 ok, 4-bit) | ✅ | ✅ | ❌ | ❌ |
| Time/image | ~50 ms | ~4 s | ~6 s | ~6 s | ~0 | ~0 |
| VRAM | 700 MB | 5 GB | 5 GB | 6 GB | 0 | 0 |
| World knowledge | Limited (QVH vocab) | Strong | Strong | Strong | N/A | N/A |
| Output style | Generic, QVH-flavored | Fluent, varied | Structured + query | Verbose, varied | Authoritative | Noise |
| Failure mode | "a man talks to camera" | Confident hallucination | JSON parse failure | Verbose drift | None (perfect) | All match → random |

---

## 7. Consistency Score Formulation

### 7.1 The formula

For thumbnail **T**, video **V** with **W** chunks (W=1 when L ≤ 75), and
**K** generated queries:

**Per-chunk pooled video feature** — saliency-softmax over top-p clips:

```
e_clip_w =  Σ_t softmax(s_k[t] / τ) · v[t]              (1)

       where s_k[t] is the saliency for clip t when M1 is given query k,
             v[t] is the CLIP feature of clip t,
             τ = 0.1,
             only the top 25% of clips by saliency contribute (rest masked to -inf)
```

**Per-(query, chunk) score**:

```
c(k, w) =  α · cos(e_qk, e_clip_w)  +  (1 - α) · cos(e_T, e_clip_w)   (2)

       where e_qk is the CLIP-text embedding of query k,
             e_T is the CLIP-image embedding of the thumbnail,
             α = 0.5
```

**Aggregation**:

```
per_query[k] = max_w c(k, w)                                          (3)
                  (best chunk for query k)

C(T, V)      = mean_k per_query[k]                                    (4)
                  (average over K queries — the deliberate "agreement" check)
```

### 7.2 Design rationale

| Choice | Why |
|---|---|
| Saliency-softmax pool, not mean pool | Concentrates on the most relevant moments |
| Top-p mask before softmax | Avoids dilution across irrelevant clips |
| α=0.5 default | Balances query-grounded retrieval and direct visual similarity |
| Max over chunks | A consistent thumbnail only needs *some* part of the video to match |
| Mean over queries | Conservative: the thumbnail's meaning should be supported by **most** queries, not just one |
| K=3 nucleus-sampled queries | Diversity > beam search paraphrases |

### 7.3 What α controls (ablation)

| α | Score uses... | Behavior |
|---|---|---|
| 0.0 | Only `cos(e_T, e_clip)` | Pure visual similarity, ignores query |
| 0.5 | Balanced | V1 default |
| 1.0 | Only `cos(e_q, e_clip)` | Pure query path |

Our query-mismatch sweep (Section 10) shows α=1.0 achieves AUC 0.972 vs
α=0.0 at 0.497, **proving the query path is the dominant signal** when
visual content is held constant.

---

## 8. Data Preparation

### 8.1 Source — QVHighlights

| Stat | Value |
|---|---|
| Total videos | 10,148 |
| Train/val/test | 7,218 / 1,550 / 1,380 |
| Avg duration | 150 s |
| Avg query length | 12 words |
| Avg relevant window | 25 s |

For each train record:
- 1 natural-language query
- 1+ relevant_windows (continuous time intervals)
- ~5-30 `relevant_clip_ids` (2-sec clips inside the windows)
- 3 annotators' saliency scores per relevant clip (0..4)

### 8.2 Our prepared artifacts

| File | Size | Purpose |
|---|---|---|
| `train_processed.jsonl` | ~3,100 records | All QVH train with frame paths + selection metadata |
| `m1_train_subset.jsonl` | 2,000 | QVH-format subset (for optional M1 retrain audit) |
| `m1_val.jsonl` | 1,550 | Full QVH val in official format |
| `m2_train_pairs.jsonl` | ~2,800 | Single-frame (frame, query) for M2 training |
| **`m2_train_pairs_multi.jsonl`** | **~8,400** | **K=3 multi-frame aug — default M2 training set** |
| `m2_val_pairs.jsonl` | ~300 | M2 val for early-stopping |
| **`consistency_eval.jsonl`** | **600** | **E3 cross-video eval** |
| **`consistency_eval_hard.jsonl`** | **584** | **E4 within-video eval ★** |
| **`consistency_eval_query_mismatch.jsonl`** | **600** | **E5 query-mismatch eval ★** |
| `thumbnail_index.jsonl` | 1,733 | Real YouTube thumbnails for domain-gap probe |
| `m2_image_emb.npz` | ~3 MB | Cached CLIP-img embeddings for training frames |
| `manual_gold_template.jsonl` | 50 | Template for optional manual labeling |

### 8.3 Frame extraction strategy

We use **center-frame extraction** of the highest-saliency 2-second clip,
with tie-breaking by annotator agreement (lowest variance), then temporal
position. This is implemented in `select_best_frame()` in the data notebook.

For multi-frame augmentation, K=3 frames are sampled evenly across all
`relevant_clip_ids` and each (frame, query) becomes a separate training pair.

---

## 9. Evaluation Methodology

### 9.1 Why we need multiple eval sets

A single eval set can hide pipeline failures. We construct **three** sets that
each test a different aspect of the consistency claim, with construction
choices designed to expose different failure modes:

```
              Negative type     What it tests        CLIP-only AUC expected
              ─────────────     ─────────────        ──────────────────────
   E3   Cross-video             pipeline works         ~0.5 (no leak)
        (different video,         on easy cases        OR
         different topic)                              ~0.9+ (visual leak)
                                                       ★ we measured 0.968 → LEAK

   E4   Within-video             temporal              ~0.5 (visual continuity
        (same video,             grounding              within video)
         frame outside the
         relevant window)

   E5   Query-mismatch           query path             0.5 by construction
        (same frame + same       contribution           (positives and negatives
         video, wrong query)                            share frame+video, differ
                                                        only in query)
```

### 9.2 Eval set construction (formal)

For each eval set we sample 300 positive pairs and 300 negative pairs
balanced 1:1.

**Positive (all three sets)**:
```
thumbnail_frame  = qvh_frames/{vid_A}.jpg
query            = QVHighlights query for vid_A
video            = vid_A
label            = 1
```

**E3 cross-video negative**:
```
thumbnail_frame  = qvh_frames/{vid_A}.jpg
query            = QVHighlights query for vid_A
video            = vid_B  ★ different video, cos(q_A, q_B) < 0.6
label            = 0
```

**E4 within-video negative** (the most important):
```
thumbnail_frame  = qvh_frames_within_neg/{vid_A}_clip{c}.jpg
                   ★ c is OUTSIDE the relevant_window of vid_A
                   ★ chosen at distance ≥5 from any relevant clip
query            = QVHighlights query for vid_A
video            = vid_A  (same video!)
label            = 0
```

**E5 query-mismatch negative**:
```
thumbnail_frame  = qvh_frames/{vid_A}.jpg
query            = QVHighlights query for vid_B  ★ where cos(q_A, q_B) < 0.4
video            = vid_A
label            = 0
```

### 9.3 Metrics

| Metric | Formula | What it tells us |
|---|---|---|
| ROC-AUC | Rank-based, 0.5 = random, 1.0 = perfect | Primary headline metric |
| Cohen's d | (μ_pos - μ_neg) / s_pooled | Effect size |
| Welch's t / p | Two-sample t-test | Statistical significance |
| Confusion at median | TP/FP/TN/FN/Acc/F1 at threshold = median(scores) | Operating point |

### 9.4 Ablation design

Five query sources × three eval sets = **15 cells** in the headline table:

| Source | Description |
|---|---|
| `gt` | Ground-truth lookup (ceiling) |
| `clipcap_k3` | 3 nucleus-sampled ClipCap queries |
| `qwen2vl_simple_k3` | 3 nucleus-sampled Qwen2-VL queries |
| `llava16_aspect_k3` | 3 LLaVA-1.6 queries |
| `random_query` | Control |

Plus the **α sweep on E5**: α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.

---

## 10. Results

### 10.1 Headline AUC table (GT queries)

| Eval | Full pipeline AUC | CLIP-only AUC | M1 lift | Cohen's d |
|---|---|---|---|---|
| E3 Cross-video | 0.939 | **0.968** | **-0.029** | 2.25 |
| **E4 Within-video** | **0.638** | 0.540 | **+0.098** | 0.45 |
| **E5 Query-mismatch** | **0.726** | 0.500 | **+0.226** | 0.84 |

### 10.2 α-sweep on query-mismatch eval (clean test of formula)

| α | AUC | Interpretation |
|---|---|---|
| 0.0 (thumb-only) | 0.497 | Random (frame+video identical pos vs neg) ✓ |
| 0.5 (balanced) | 0.752 | Default — full pipeline |
| **1.0 (query-only)** | **0.972** | **Near-perfect — proves query path works** |

### 10.3 Key finding — visual leak in cross-video eval

The naive "cross-video evaluation" inflates apparent AUC because the
thumbnail (extracted from the positive's video) is visually distant from
random unrelated videos. **CLIP-only on this eval gives AUC = 0.968**, almost
the same as the full pipeline (0.939). The full pipeline is not contributing.

**This is a methodology finding**: cross-video synthetic negatives are too
easy. Reporting them as evidence of pipeline performance is misleading.

### 10.4 The honest research claim

The **within-video** AUC = 0.638 (vs CLIP-only 0.540, **+0.10 lift from M1**)
is the genuine measurement. The pipeline contributes meaningful but modest
temporal-grounding signal.

The **query-mismatch** AUC = 0.726 (vs CLIP-only 0.500, **+0.23 lift from
query path**) confirms M1's saliency genuinely responds to the query content,
not just the visual.

### 10.5 M2 ablation (pending)

To be filled in after the multi-M2 ablation finishes via
`notebooks/m2_ablate.ipynb`. Expected structure:

```
                       AUC on consistency_eval_hard.jsonl (within-video)
M2 generator            with GT query    with this M2's queries
──────────────────      ─────────────    ─────────────────────
gt (ceiling)               0.638                  —
clipcap_k3                   —                   0.??
qwen2vl_simple               —                   0.??
qwen2vl_aspect               —                   0.??
llava16_aspect               —                   0.??
random_query                 —                   ~0.50
```

The interesting comparisons:
- Best M2 vs GT: how much does the trained-from-scratch ClipCap lose vs a
  zero-shot VLM?
- VLM-simple vs VLM-aspect: does the structured intermediate help?
- Aspect-rewriter wrapping the best upstream: net positive or net negative?

---

## 11. Limitations & Discussion

### 11.1 Cross-video eval has visual leak

(See Section 10.3.) Mitigated by reporting E4 (within-video) and E5
(query-mismatch) as primary metrics. Cross-video reported as a methodology
finding, not a performance claim.

### 11.2 Within-video AUC is moderate (0.638)

M1's saliency provides useful but not decisive temporal signal on this hard
test. This is consistent with the **short-moment localization weakness** in
the original Moment-DETR paper (3.1% mAP on moments <10s). Many QVH events
are short; the per-clip saliency head is more reliable than span regression
on this regime, but is still noisy.

### 11.3 Real-thumbnail domain gap not measured in V1

M2 was trained on natural video frames; real YouTube thumbnails are
stylized (text overlays, color grading). We provide tooling for this
experiment in [docs/Markdowns/Label/OOD_EVALUATION.md](Label/OOD_EVALUATION.md)
§E8 (`scores_real_thumbnails.jsonl`) but did not run it.

### 11.4 No manual gold labels in V1

A 50-pair manual labeling tooling exists (`manual_gold_template.jsonl`,
`docs/Markdowns/Label/LABELING_GUIDE.md`) but labels weren't collected
within the V1 timeline. The α-sweep and CLIP-only diagnostics on each eval
set serve as validation that the synthetic eval reflects pipeline-level
behavior.

### 11.5 M1 is not fine-tuned for our domain

We use the official released checkpoint. Retraining on our 2K subset would
yield ~45% R1@0.5 vs the official 53% — strictly worse. So we don't retrain.

### 11.6 ClipCap is generic

Trained on ~9K (frame, query) pairs from QVH, ClipCap captures the
domain-specific vocabulary but lacks world knowledge. VLMs (Qwen, LLaVA)
bring world knowledge but at 100x compute cost.

---

## 12. Future Work

| # | Direction | Effort | Payoff |
|---|---|---|---|
| 1 | **Domain-gap experiment** (E8): re-run eval with real YouTube thumbnails | 1 hour | Quantifies the train/deploy mismatch — publishable number |
| 2 | **Manual gold labels** (50 pairs, 2 hours of labeling) | 2 hours | Validates synthetic eval against human judgment |
| 3 | **Within-video harder negatives**: pick clips at the edge of relevant_window to test fine-grained localization | Medium | Could lower the 0.638 AUC and expose stronger M1 weakness |
| 4 | **CapCLIP fine-tuning** as another M2 (in `Moment_DETR/run_on_video/`) | Medium | Trains an alternative to ClipCap |
| 5 | **Florence-2 / SmolVLM** as M2 — smaller VLM (2B params) | Easy (just register a new class) | Lower-cost VLM baseline |
| 6 | **Multi-VLM ensemble** — average scores from Qwen + LLaVA + ClipCap | Easy | Often gives small AUC boost |
| 7 | **Retrieval-in-the-loop ClipCap** — reward M2 for queries that improve M1 retrieval | Hard | Strongest training signal but complex |
| 8 | **Caching layer**: M1 saliency cache + CLIP-text emb dedup | 1-2 hours | 10x faster iteration on ablations |
| 9 | **Real-clickbait dataset collection** — 100-200 YouTube videos labeled "consistent / clickbait" | Weeks | Cross-validates pipeline against engagement signal |
| 10 | **Multi-modal consistency**: add OCR on thumbnail text + ASR on video → text-text consistency | Medium | Tests another consistency axis |

---

## 13. Reproducibility

### 13.1 Repository

- Repo: github.com/zakariahmedd34/thumbnail-consistency
- Branch: feature/v1-pipeline
- Commit at submission: [FILL IN]
- License: MIT

### 13.2 Environment

| Component | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | ≥ 2.0 |
| transformers | ≥ 4.45 (for LLaVA-Next) |
| qwen-vl-utils | latest |
| accelerate | ≥ 0.30 |
| bitsandbytes | ≥ 0.43 |
| Colab runtime | T4 GPU recommended |

### 13.3 Data

Committed to git (under `data/processed/`):

- `consistency_eval.jsonl` (250 KB)
- `consistency_eval_hard.jsonl` (210 KB)
- `consistency_eval_query_mismatch.jsonl` (250 KB)
- `m2_train_pairs.jsonl`, `m2_train_pairs_multi.jsonl`, `m2_val_pairs.jsonl`
- `m1_train_subset.jsonl`, `m1_val.jsonl`
- `manual_gold_template.jsonl`
- `category_map.json`, `m2_split_indices.json`

On Drive only (size constraints):
- `moment_detr_features.tar.gz` (7.5 GB) — official Moment-DETR features
- `qvh_videos/` (~5 GB) — raw .mp4 segments
- `qvh_frames/`, `qvh_frames_multi/`, `qvh_frames_within_neg/`, `qvh_thumbnails/`

### 13.4 Run order

| Step | Notebook | Time |
|---|---|---|
| 1 | `data_collection_preparation.ipynb` | 30-45 min (data already on Drive) |
| 2 | `train_clipcap.ipynb` | ~3 hours (T4 GPU) |
| 3 | `sanity_consistency.ipynb` | 10 min |
| 4 | `m2_ablate.ipynb` with `M2_LIST=['gt','clipcap','qwen2vl_simple','llava16_aspect','random']` | ~45 min |
| 5 | `compare_m2_generators.ipynb` | 10 min (for figures) |

Each notebook has a `# [COLAB_BOOTSTRAP_v2_modular]` cell at position 1 that
mounts Drive, extracts M1 features to LOCAL SSD (NOT Drive), and registers
`src/` on `sys.path`.

### 13.5 Seeds

All random seeds set to `2026` via `src/utils/seed.py`. Output is
deterministic up to floating-point GPU variation.

### 13.6 Acceptance tests

```bash
# Unit tests (math + registry)
python -m pytest tests/ -q
# expected: 26 passed (18 aggregate + 8 M2 registry)
```

---

## 14. References

- Carion, N., et al. (2020). **End-to-End Object Detection with Transformers**.
  ECCV.
- Lei, J., Berg, T. L., & Bansal, M. (2021). **QVHighlights: Detecting
  Moments and Highlights in Videos with Natural Language Queries**. NeurIPS.
- Mokady, R., Hertz, A., & Bermano, A. H. (2021). **ClipCap: CLIP Prefix
  for Image Captioning**. arXiv:2111.09734.
- Radford, A., et al. (2021). **Learning Transferable Visual Models From
  Natural Language Supervision**. ICML.
- Wang, P., et al. (2024). **Qwen2-VL: Enhancing Vision-Language Model's
  Perception of the World at Any Resolution**. arXiv:2409.12191.
- Liu, H., et al. (2024). **LLaVA-NeXT: Improved reasoning, OCR, and
  world knowledge**.
- Dettmers, T., et al. (2022). **LLM.int8(): 8-bit Matrix Multiplication
  for Transformers at Scale**. NeurIPS.
- Loshchilov, I., & Hutter, F. (2019). **Decoupled Weight Decay
  Regularization (AdamW)**. ICLR.

---

## Appendix A — Mathematical glossary

| Symbol | Meaning |
|---|---|
| **T** | Thumbnail image |
| **V** | Video |
| **C(T, V)** | Final consistency score in [-1, 1] |
| **e_T** | CLIP-image embedding of T (512-d, L2-norm) |
| **e_q** | CLIP-text embedding of query q (512-d, L2-norm) |
| **e_clip_w** | Saliency-pooled video feature for chunk w (512-d) |
| **s[t]** | Per-clip saliency score from M1 |
| **v[t]** | CLIP visual feature of clip t |
| **α** | Weight on query path in consistency formula |
| **τ** | Saliency softmax temperature |
| **top-p** | Fraction of clips kept by saliency before pooling |
| **K** | Number of generated queries per thumbnail |
| **W** | Number of video chunks (=1 for ≤150s videos) |

## Appendix B — File index

| Type | Files |
|---|---|
| Source | `src/m2_generators/*.py`, `src/models/*.py`, `src/scoring/*.py`, `src/eval/*.py`, `src/utils/*.py`, `src/cli/*.py` |
| Configs | `configs/m2/*.yaml`, `configs/consistency.yaml`, `configs/eval.yaml`, `configs/drive.yaml` |
| Notebooks | `notebooks/{final_data_collection_preparation, train_clipcap, sanity_consistency, run_evaluation, run_pipeline_allinone, m2_score_single, m2_ablate, compare_m2_generators}.ipynb` |
| Tests | `tests/test_aggregate.py` (18), `tests/test_m2_registry.py` (8) |
| Docs | `docs/Markdowns/{INTEGRATION_PLAN, FINAL_REPORT, COMPREHENSIVE_REPORT, TEAM_STUDY_GUIDE}.md` + `Label/`, `Plan/`, `research/` |
| Data | `data/DATA_README.md`, `data/processed/*.jsonl` |
