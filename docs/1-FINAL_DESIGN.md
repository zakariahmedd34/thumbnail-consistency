# Final Consolidated Design — Thumbnail-Video Semantic Consistency

**Status:** V1 execution document. 3-day timeline.
**Team:** 4 people, continuous work.
**Last updated:** 2026-05-14.

---

## TL;DR

- **Architecture:** Thumbnail → ClipCap-M2 → query → frozen CLIP-text → official Moment-DETR → saliency-weighted pooled video features → cosine-based consistency score.
- **Only two things train:** the M2 prefix MLP and (optionally, if time permits) a tiny consistency head.
- **No retraining of M1.** Use the official released checkpoint.
- **No retraining of CLIP / SlowFast / GPT-2.** All frozen.
- **Evaluation:** AUC on 600 synthetic positive/hard-negative pairs + 50-pair manual gold set + frame-vs-real-thumbnail domain-gap measurement.

---

## 0. Glossary

| Term | Meaning |
|---|---|
| **T** | Thumbnail image (224×224 RGB) |
| **V** | Video (75 CLIP-feature clips × 2s = 150s, or chunked if longer) |
| **q_k** | k-th generated query string (M2 output) |
| **e_T** | CLIP-image embedding of thumbnail (512-d) |
| **e_q** | CLIP-text embedding of query (512-d) |
| **e_clip_w** | Saliency-softmax pooled CLIP video feature for chunk w (512-d) |
| **C(T, V)** | Final consistency score in [-1, 1] |
| **M1** | Moment-DETR (official checkpoint) |
| **M2** | ClipCap-style thumbnail → query model |

---

## 1. Where Learning Happens

### 1.1 The honest answer

> The project is **mostly orchestration of pretrained systems**, with **one small learned component (M2 prefix MLP)** and **one explicit task formulation (the consistency score)**.
>
> That is still a valid research contribution — the contribution is the *formulation and evaluation of a new task* (thumbnail-video consistency as a retrieval-grounded score), not the invention of a new architecture.

### 1.2 Component-by-component status

| Component | State | Params | Supervision | What it learns |
|---|---|---|---|---|
| CLIP-ViT-B/32 (visual) | Pretrained, **frozen** | 150M | — | Web-scale image-text alignment (already learned) |
| CLIP-text | Pretrained, **frozen** | 63M | — | Web-scale text representation (already learned) |
| GPT-2 small | Pretrained, **frozen** | 117M | — | English language modeling (already learned) |
| SlowFast features | Pre-extracted, **not used at inference** | — | — | (Already encoded in feature files) |
| **M2 prefix MLP** | **TRAINED** | ~12M | QVHighlights query strings | Map CLIP image space → GPT-2 prefix space |
| Moment-DETR transformer | Pretrained (official), **frozen** | ~8M | (Already trained on full QVH) | Cross-modal moment retrieval |
| **Consistency head** *(optional V1.1)* | **TRAINED** | ~100 | 50-pair manual gold set | Logistic regression on score features |
| Aggregation module | Deterministic | 0 | — | Nothing — pure math |

### 1.3 "Where Learning Happens" diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       FROZEN (pretrained)                        │
│                                                                  │
│   CLIP-img  ─┐                              ┌─  CLIP-text        │
│              │                              │                    │
│              │                              │                    │
│              ▼                              ▼                    │
│        (e_T)                          (e_q)                      │
│              │                              │                    │
│              │                              ▼                    │
│              │                    ┌─────────────────┐            │
│              │                    │  Moment-DETR    │            │
│              │                    │  (OFFICIAL)     │            │
│              │                    │  → saliency     │            │
│              │                    └─────────────────┘            │
└──────────────┼──────────────────────────────────────────────────-┘
               │
               │       ╔════════════════════════════════════╗
               │       ║  ★ TRAINED (this is the learning)  ║
               │       ║                                    ║
               └────── ║   M2 Prefix MLP                    ║
                       ║   512 → (768 × 10) prefix tokens   ║
                       ║   Loss: cross-entropy on GT query  ║
                       ║   Data: 1608 (frame, query) pairs  ║
                       ║                                    ║
                       ║   (Optional) Consistency Head      ║
                       ║   4-d → 1 logistic regression      ║
                       ║   Loss: BCE on manual gold labels  ║
                       ║   Data: 50 (T, V, label) pairs     ║
                       ╚════════════════════════════════════╝
```

### 1.4 What is the research framing?

```
                  ┌──────────────────────┐
                  │   This project is:    │
                  └──────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Architecture         Metric              Task formulation
   invention           learning             & orchestration ★
   (NO)                (partial)            (YES — claim this)
```

**Claim in writeup:** *"We propose a retrieval-grounded consistency metric and a captioning-bridged pipeline that grounds thumbnails in temporal video evidence, without requiring labelled clickbait data."*

### 1.5 Supervision sources

| Supervision signal | Used by | Comes from |
|---|---|---|
| QVHighlights query strings | M2 training | Human annotators (QVH paper) |
| QVHighlights saliency scores | M1 (already trained) | Human annotators (QVH paper) |
| QVHighlights relevant_windows | M1 (already trained) | Human annotators (QVH paper) |
| Synthetic pos/neg pairs | Pipeline evaluation | Constructed from QVH structure |
| 50-pair manual labels | Optional consistency head + sanity | This team |

---

## 2. Data Scenarios — A / B / C / D

### 2.1 Comparison table

| | **A. 2000 subset only** | **B. Full QVH (~7K)** | **C. + subs_train (~150K ASR)** | **D. + external YouTube** |
|---|---|---|---|---|
| **M1 retrain?** | No (use official ckpt) | No | No | Would require, not feasible |
| **M2 benefits?** | Marginal — current state | Slightly (more pairs ~5K) | **Yes, large** — pretraining | Yes if labels exist, but no labels |
| **CLIP adapt?** | No | No | No | Tempting, breaks M1 |
| **SlowFast adapt?** | No | No | No | Not feasible |
| **Preprocessing cost** | 0 (done) | ~1h (extract more frames) | ~6h (extract 150K frames) | ~2 days (download + label) |
| **Expected gain** | Baseline | +2-3% AUC | +5-10% AUC | Unknown |
| **Engineering risk** | None | Low | Medium (disk, OOM) | High (yt-dlp, labels) |
| **Worth it for 3 days?** | YES (default) | OPTIONAL (Day 1 spare) | OPTIONAL (if Day 1 fast) | NO |

### 2.2 The "frozen models + more data" paradox — resolved

```
┌────────────────────────────────────────────────────────────┐
│  Q: If everything is frozen, why would more data help?     │
└────────────────────────────────────────────────────────────┘

Answer: Only data that flows through a TRAINABLE component helps.

  ┌─────────────────────────────────────────────────────────┐
  │  Data → CLIP (frozen) → ... → frozen ... → output       │
  │                                                         │
  │  No gradient → no learning → MORE DATA DOES NOTHING     │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  Data → CLIP (frozen) → M2-prefix (TRAINABLE) → ...     │
  │                              ▲                          │
  │                              │  gradient flows here     │
  │                                                         │
  │  → MORE DATA HELPS this component                       │
  └─────────────────────────────────────────────────────────┘
```

**Practical implication:**
- Case A/B: extra data only helps M2. Marginal because M2 is already capacity-matched.
- Case C: extra (frame, ASR) pairs *do* help M2 via pretraining. Best ROI if Day 1 finishes early.
- Case D: extra videos do nothing unless you also retrain something they flow through.

### 2.3 Recommended path

```
   Day 1 morning
       │
       ▼
   [Case A baseline]
   M2 trains on 1608 pairs
       │
       ▼
   ┌─────────────────────────────┐
   │ Day 1 finished early?       │
   └─────────────────────────────┘
       │            │
      Yes          No
       │            │
       ▼            ▼
  [Case C add]   STAY ON A
  Pretrain on
  subs_train
  150K pairs
       │
       ▼
   Fine-tune on
   1608 pairs
```

---

## 3. Final Pipeline Diagram

### 3.1 End-to-end (V1)

```
                       INPUT
                  ┌──────────────┐
                  │  Thumbnail T │  (224×224 RGB)
                  └──────┬───────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼ [FROZEN]                        ▼ [TRAINED+FROZEN]
  ┌─────────────┐                  ┌─────────────────┐
  │  CLIP-img   │                  │   M2 = ClipCap  │
  │  ViT-B/32   │                  │   ┌──────────┐  │
  └──────┬──────┘                  │   │  CLIP-img│  │ (frozen)
         │                         │   └─────┬────┘  │
         │ e_T (512)               │         ▼       │
         │                         │   ┌──────────┐  │
         │                         │   │ Prefix   │  │ ★ TRAINED
         │                         │   │ MLP      │  │
         │                         │   └─────┬────┘  │
         │                         │         ▼       │
         │                         │   ┌──────────┐  │
         │                         │   │  GPT-2   │  │ (frozen)
         │                         │   │  decode  │  │
         │                         │   └─────┬────┘  │
         │                         └─────────┼───────┘
         │                                   │
         │                            {q_1, q_2, q_3}
         │                            nucleus(p=0.9, T=1.1)
         │                                   │
         │                                   ▼ [FROZEN]
         │                            ┌──────────────┐
         │                            │  CLIP-text   │
         │                            └──────┬───────┘
         │                                   │
         │                            {e_q1, e_q2, e_q3}
         │                                   │
         │                                   ▼ [FROZEN]
         │             ┌──────────────────────────────────┐
         │             │   M1 = Moment-DETR (OFFICIAL)    │
         │             │                                  │
         │             │   Inputs: e_qk + video feats     │
         │             │   Outputs:                       │
         │             │     • saliency s_k[t] (per clip) │
         │             │     • spans (10, 2)              │
         │             │     • fg-prob (10,)              │
         │             └────────────────┬─────────────────┘
         │                              │
         │                              ▼ [DETERMINISTIC]
         │                ┌──────────────────────────────┐
         │                │  Saliency-softmax pooling    │
         │                │  e_clip_w =                  │
         │                │   Σ_t softmax(s_k[t]/τ)·v[t] │
         │                └──────────────┬───────────────┘
         │                               │
         │                               ▼
         │           ┌────────────────────────────────────────┐
         │           │ Per-(k, chunk w) score:                │
         └──────────▶│ c(k,w)=0.5·cos(e_qk,e_clip_w)          │
                     │        +0.5·cos(e_T,e_clip_w)          │
                     └─────────────────┬──────────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────────┐
                     │ Aggregation:                           │
                     │   per_q = max_w c(k, w)                │
                     │   C(T,V) = mean_k per_q                │
                     └─────────────────┬──────────────────────┘
                                       │
                                       ▼
                                  ┌──────────┐
                                  │ C(T, V)  │  ∈ [-1, 1]
                                  └────┬─────┘
                                       │
                       ┌───────────────┼───────────────┐
                       ▼               ▼               ▼
                  Synthetic        Manual gold     Frame-vs-
                  AUC (600)        AUC (50)        thumbnail gap
```

### 3.2 Legend

| Symbol | Meaning |
|---|---|
| `[FROZEN]` | Pretrained, no gradient, inference only |
| `[TRAINED+FROZEN]` | Contains trained part + frozen parts |
| `★ TRAINED` | The component being learned in this project |
| `[DETERMINISTIC]` | Math only, no parameters |

---

## 4. Modularity vs Rapid Implementation

### 4.1 Recommendation: **Hybrid — modules from Day 1, notebooks for experiments only.**

Pure-notebook is fastest for one person but breaks for a team of 4. Pure modularization wastes Day 1. The hybrid:

```
   src/   ←── stable code, imported everywhere, on git
   notebooks/  ←── experiments, plots, scratch
   configs/  ←── YAML for hyperparams
```

### 4.2 What goes where

| Code type | Location | Reason |
|---|---|---|
| Data loading (M1/M2) | `src/data/` | 4 people read the same loader |
| ClipCap model class | `src/models/clipcap.py` | Trained module, needs versioning |
| Consistency scorer | `src/scoring/consistency.py` | Tested by team, called by all |
| Aggregation math | `src/scoring/aggregate.py` | Pure functions, unit-testable |
| M1 inference wrapper | `src/models/moment_detr.py` | Called by scorer |
| Eval metrics | `src/eval/metrics.py` | AUC, separation, etc. |
| Configs | `configs/*.yaml` | Hyperparams without code edit |
| Training run | `notebooks/train_clipcap.ipynb` | One-off experiment |
| Eval run | `notebooks/run_evaluation.ipynb` | Plots + numbers |
| Qualitative figures | `notebooks/qualitative.ipynb` | Visual outputs |
| Sanity checks | `notebooks/sanity_*.ipynb` | Quick checks |

### 4.3 Recommended folder structure

```
thumbnail-consistency/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── m1_dataset.py        # official QVH JSONL loader
│   │   ├── m2_dataset.py        # (frame, query) loader
│   │   ├── consistency_eval.py  # pos/neg pair builder
│   │   └── thumbnails.py        # yt-dlp thumbnail downloader
│   ├── models/
│   │   ├── clipcap.py           # ★ trained M2
│   │   ├── moment_detr.py       # wrapper for official ckpt
│   │   └── encoders.py          # CLIP loaders (cached)
│   ├── scoring/
│   │   ├── aggregate.py         # per-chunk, per-query math
│   │   ├── consistency.py       # main score function
│   │   └── chunking.py          # video chunking for >150s
│   ├── eval/
│   │   ├── metrics.py           # AUC, Cohen's d, t-test
│   │   └── qualitative.py       # figure builders
│   └── utils/
│       ├── io.py                # jsonl load/save (utf-8)
│       └── seed.py
├── configs/
│   ├── clipcap.yaml
│   ├── consistency.yaml
│   └── eval.yaml
├── notebooks/
│   ├── data_collection_preparation.ipynb  # existing, fix bugs
│   ├── m1_baseline.ipynb                  # existing, sanity only
│   ├── train_clipcap.ipynb                # NEW
│   ├── run_evaluation.ipynb               # NEW
│   ├── qualitative_figures.ipynb          # NEW
│   └── sanity_consistency.ipynb           # NEW
├── docs/
│   ├── FINAL_DESIGN.md                    # this file
│   └── research/                          # existing notes
├── outputs/
│   ├── checkpoints/clipcap_best.pt
│   ├── predictions/*.jsonl
│   └── figures/*.png
└── tests/
    └── test_aggregate.py        # unit tests for scoring math
```

### 4.4 Productivity rules for the team

| Rule | Why |
|---|---|
| Never import across notebooks | Notebooks are scratch; circular deps kill |
| Pin random seeds in `src/utils/seed.py` | Reproducible AUC |
| Cache `e_T` and `e_clip` to `.npy` | Don't re-encode 600 times during scorer debugging |
| One person owns `src/scoring/` | Avoid merge conflicts on the core math |
| All numbers go to `outputs/predictions/` as JSONL | Easy to diff |

---

## 5. Explainability

### 5.1 Decision: **Defer formal XAI. Ship saliency + qualitative figures.**

```
   What V1 needs                       What V1 does NOT need
   ─────────────                       ─────────────────────
   ✓ M1 saliency curves                ✗ Attention rollout
   ✓ Top retrieved windows             ✗ Grad-CAM
   ✓ Generated query + score           ✗ Token-level attention maps
   ✓ Pos/neg score histograms          ✗ SHAP / LIME
   ✓ 5-10 qualitative case studies     ✗ Counterfactual explanations
```

### 5.2 What counts as "enough interpretability" for V1

| Artifact | Insight provided | Cost |
|---|---|---|
| Per-video saliency heatmap | *Where* M1 looked | 5 min/figure |
| Retrieved top-3 windows overlay | *What* M1 found | 5 min/figure |
| Generated query alongside score | *Why* score is high/low | Free (already produced) |
| Side-by-side `e_T`-only vs full score | *Did M2 help here?* | Free with ablation |
| Score distribution plot (pos vs neg) | Pipeline behavior | 10 min, one plot |

### 5.3 Decision tree

```
  Need explainability?
        │
        ▼
  Have time on Day 3?
   /         \
  Yes        No
   │          │
   ▼          ▼
  Add 5      Just produce
  qualitative  the score
  cases +     distribution
  saliency    histogram
  overlays
```

---

## 6. Three-Day Schedule

### 6.1 Role assignments

| Person | Owns |
|---|---|
| **A** | M1 wrapper, official-checkpoint inference, saliency caching |
| **B** | M2 (ClipCap class, training, generation) |
| **C** | Data: notebook fixes, thumbnail download, consistency_eval split |
| **D** | Scoring + evaluation + figures |

### 6.2 Day 1 — Foundations

```
┌─────────────────────────────────────────────────────────────────┐
│ DAY 1 — Everyone builds in src/, no shared state yet            │
└─────────────────────────────────────────────────────────────────┘

   A (M1)             B (M2)             C (Data)         D (Score)
   ─────              ──────             ────────         ─────────
   ↓ Load offical     ↓ Build ClipCap    ↓ Fix notebook   ↓ Implement
     ckpt to GPU       class (~30 LoC)     bugs B1-B6      aggregate.py
   ↓ Build           ↓ Switch to GPT-2   ↓ Add thumbnail  ↓ Implement
     moment_detr.py    tokenizer           download cell    consistency.py
     wrapper         ↓ Start training    ↓ Build          ↓ Smoke test on
   ↓ Cache video       on 1608 pairs       m2_val_pairs    5 hand pairs
     features for                          (300 from val)   with GT query
     all eval vids                       ↓ Cache e_T      ↓ Stub
                                          to .npy           consistency_eval
                                                            .jsonl builder

  End of Day 1: M2 training running, M1 callable, data ready, scorer skeleton.
```

### 6.3 Day 2 — Convergence

```
┌─────────────────────────────────────────────────────────────────┐
│ DAY 2 — Wire everything together, primary results               │
└─────────────────────────────────────────────────────────────────┘

   A (M1)             B (M2)             C (Data)         D (Score)
   ─────              ──────             ────────         ─────────
   ↓ Verify saliency  ↓ Pick best        ↓ Build          ↓ End-to-end
     outputs            checkpoint         consistency_       run on 600
     correct            (CLIP-text sim)    eval.jsonl         pairs
   ↓ Cache             ↓ Generate K=3      (300+300)        ↓ Compute AUC,
     saliency for       queries per       ↓ Start manual      Cohen's d
     all eval pairs     thumbnail          labeling (50)   ↓ Run ablation
                                                              table
                                                              (GT/Cap/eT/rand)

  End of Day 2: HEADLINE NUMBER (AUC) is in. Ablation table is in.
```

### 6.4 Day 3 — Analysis & writeup

```
┌─────────────────────────────────────────────────────────────────┐
│ DAY 3 — Figures, writeup, polish                                │
└─────────────────────────────────────────────────────────────────┘

   A                B                  C                  D
   ─                ─                  ─                  ─
   ↓ Saliency       ↓ M2 qualitative   ↓ Frame-vs-real-  ↓ Final score
     heatmaps         (10 cases:        thumbnail gap       distribution
     for 5 cases      success+fail)     experiment          plots
                                       ↓ Finish manual    ↓ τ / top-p
                                         labels             sensitivity
                                                            sweep

   ──────── All hands: writeup, figures, final report integration ────────

  End of Day 3: complete prototype + report.
```

### 6.5 What MUST work (in priority)

| Priority | Item | Fallback if it fails |
|---|---|---|
| P0 | M1 official checkpoint loads & runs inference | Drop the pipeline — must work |
| P0 | M2 trains and generates non-`<UNK>` queries | Fall back to GT query for eval (still publishable) |
| P0 | Consistency score computes on a (T, V) pair | Drop the pipeline — must work |
| P0 | Synthetic 600-pair AUC | Drop manual labels, keep synthetic |
| P1 | Manual gold AUC | Skip, keep synthetic only |
| P1 | Multi-query (K=3) | Fall back to K=1 |
| P2 | Frame-vs-thumbnail domain gap | Note as future work |
| P2 | Consistency head (learned) | Skip, use deterministic only |
| P3 | subs_train pretraining | Skip |
| P3 | Saliency qualitative figures | Skip if time short |

### 6.6 Highest-risk components

```
   Risk                                  Mitigation
   ────                                  ──────────
   1. M2 produces degenerate queries   → Monitor CLIP-text sim every epoch,
                                         early stop, fall back to GT
   2. Synthetic negatives accidentally → CLIP-text dissim < 0.6 filter,
      consistent                         spot-check 20 of them
   3. Moment-DETR ckpt incompat with   → Test ckpt loading in hour 1 of Day 1
      our environment
   4. yt-dlp 403 on thumbnail download → Use Wayback Machine fallback or
                                         drop the experiment
   5. AUC near 0.5 (pipeline doesn't  → Run with GT query first — if GT
      work)                              also gives ~0.5, the score formula
                                         itself is broken, not M2
```

---

## 7. FINAL DECISIONS

| # | Decision | Rationale |
|---|---|---|
| D1 | **Use official Moment-DETR checkpoint. Do not retrain.** | Best baseline; saves 4-6h; cleaner numbers |
| D2 | **M2 = ClipCap (frozen GPT-2 + trained prefix MLP)** | Eliminates vocab bottleneck; trainable with 1608 pairs |
| D3 | **Freeze CLIP visual, CLIP text, GPT-2, SlowFast** | Tuning them with 1608 samples → overfit; breaks M1 |
| D4 | **Multi-query K=3 via nucleus sampling** | Diversity > beam paraphrases |
| D5 | **Consistency = mean_q max_w [0.5·cos(e_q, e_clip) + 0.5·cos(e_T, e_clip)]** | Captures both M2 path and direct visual path |
| D6 | **Saliency-softmax pooling (τ=0.1, top-p=0.25)** | Saliency more stable than spans |
| D7 | **Evaluation = synthetic 600 AUC + 50 manual gold** | Replaces missing clickbait labels |
| D8 | **Download real YouTube thumbnails as metadata** (eval only) | Cheap; gives domain-gap measurement |
| D9 | **Hybrid src/ + notebooks layout** | Team productivity for 4 people |
| D10 | **Defer XAI; ship qualitative figures only** | Saliency + retrieval already interpretable |

---

## 8. OPEN QUESTIONS

| # | Question | When to decide | Default if undecided |
|---|---|---|---|
| Q1 | Use subs_train for M2 pretraining? | End of Day 1 | No — only if Day 1 is fast |
| Q2 | Train a learned consistency head? | Day 3 morning | No — keep deterministic |
| Q3 | Include SlowFast features in M1 input? | Day 1 hour 1 | No — official ckpt is CLIP-only |
| Q4 | Manual label N? (50 vs 100) | Day 2 | 50 (split across team) |
| Q5 | Beam-3 vs nucleus for M2 inference? | Day 2 morning | Both — report nucleus, ablate |
| Q6 | Weight `sim_qv` vs `sim_Tv`? (0.5 / 0.5) | Day 3 if time | Default 0.5 / 0.5 |

---

## 9. HIGH-RISK ITEMS

```
  ┌────────────────────────────────────────────────────────────┐
  │ R1: GT-query AUC must beat random                          │
  │     If pipeline doesn't separate pos/neg even with the     │
  │     GROUND-TRUTH query, the score formula is broken.       │
  │     ⇒ Test this Day 1 evening before anything else.        │
  └────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ R2: ClipCap collapse                                       │
  │     Frozen GPT-2 + tiny MLP can collapse to a constant     │
  │     ("a man walks") if LR too high or prefix_len wrong.    │
  │     ⇒ LR 2e-4, prefix_len=10, monitor diversity hourly.    │
  └────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ R3: Feature alignment bugs                                 │
  │     Official feature filenames vs subset vids must match.  │
  │     ⇒ Day 1 hour 1: set intersection check.                │
  └────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ R4: Hard-negative quality                                  │
  │     Random swap → some pairs are accidentally consistent.  │
  │     ⇒ CLIP-text dissimilarity < 0.6 filter, spot-check 20. │
  └────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ R5: Notebook chaos                                         │
  │     4 people editing the same .ipynb → merge hell.         │
  │     ⇒ All shared logic in src/. Notebooks import only.     │
  └────────────────────────────────────────────────────────────┘
```

---

## 10. FUTURE WORK (explicitly deferred)

| Item | Why deferred | When useful |
|---|---|---|
| CLIP fine-tuning on (thumbnail, frame) pairs | Needs collected data, risks M1 break | Future paper |
| Retrieval-in-the-loop M2 training | Differentiable retrieval is hard | Future paper |
| Engagement-metric proxy labels (CTR drop) | No data access in 3 days | Industrial extension |
| Cross-dataset generalization | Need ActivityNet / TaCoS | Future paper |
| Attention rollout / token-level XAI | Not required for V1 claim | If reviewer asks |
| Learned chunk aggregator (vs max) | Needs more labeled data | V2 |
| Multi-modal: thumbnail OCR text | Engineering only | V2 |
| Real clickbait dataset collection | Two-week labeling project | Full paper |

---

## 11. Quick reference — what to build first

```
   Hour 0 ────────────────────────────────────────► Hour 72
     │                                                │
     │ ┌───────────────────────────────────────┐      │
     │ │  P0 critical path                     │      │
     │ └───────────────────────────────────────┘      │
     │                                                │
     ▼                                                ▼
   ① Verify M1 ckpt loads (A, 1h)
   ② Cache video features keyed by vid (A, 2h)
   ③ ClipCap class + tokenizer swap (B, 3h)
   ④ Notebook bug fixes B1-B6 (C, 2h)
   ⑤ Consistency scorer skeleton (D, 3h)
   ⑥ Build consistency_eval.jsonl (C, 2h)
   ⑦ Smoke test: GT query → AUC sanity (D, 1h)
   ⑧ Start ClipCap training overnight (B, run)
                                          │
                                          ▼
                                       Day 2: end-to-end run, AUC, ablation
                                          │
                                          ▼
                                       Day 3: figures + writeup
```

---

## Appendix A — Equations

**Per-chunk pooled video feature:**

$$e_{\text{clip},w} = \sum_{t=1}^{L_w} \text{softmax}\!\left(\frac{s_k[t]}{\tau}\right) \cdot v_{w,t}, \quad \tau=0.1$$

(Take only top 25% by saliency before softmax for stability.)

**Per-(query, chunk) score:**

$$c(k, w) = \alpha \cos(e_{q_k}, e_{\text{clip},w}) + (1-\alpha) \cos(e_T, e_{\text{clip},w}), \quad \alpha=0.5$$

**Multi-query aggregation:**

$$C(T, V) = \frac{1}{K} \sum_{k=1}^{K} \max_w c(k, w)$$

**Optional confidence damping:**

$$C_{\text{final}} = C(T, V) \cdot \left(0.5 + 0.5 \cdot \max_{i \in \text{top-3 spans}} p_{\text{fg},i}\right)$$

---

## Appendix B — File checklist (end of project)

```
✓ outputs/checkpoints/clipcap_best.pt
✓ outputs/predictions/m2_generated.jsonl
✓ outputs/predictions/consistency_scores.jsonl
✓ outputs/figures/score_distribution.png
✓ outputs/figures/ablation_table.png
✓ outputs/figures/saliency_qualitative_*.png   (5 cases)
✓ outputs/figures/m2_qualitative.png            (10 cases)
✓ outputs/figures/frame_vs_thumbnail_gap.png
✓ docs/FINAL_REPORT.md                           (3-day writeup)
```
