# Evaluation Process — End-to-End

> How data prepared by `data_collection_preparation.ipynb` flows into the
> evaluation notebooks, what gets measured at each stage, and how to interpret
> the numbers.

---

## 0. TL;DR

The project produces **five evaluations**, in order of dependence:

| # | Eval | Notebook / cell | What it measures | Headline metric |
|---|---|---|---|---|
| E1 | M1 sanity | `m1_baseline.ipynb` or inline | M1 official checkpoint works | Saliency curves look sane |
| E2 | M2 intrinsic | end of `train_clipcap.ipynb` | M2 query quality vs GT | CLIP-text-sim, vocab coverage |
| E3 | R1 smoke (cross-video) | `run_pipeline_allinone.ipynb` (`consistency_eval.jsonl`) | Pipeline separates matched/unmatched videos | AUC ≈ 0.95 (ceiling) |
| E4 | Within-video hard | same notebook, `consistency_eval_hard.jsonl` | **Pipeline localizes the correct moment** | AUC 0.65–0.80 (★ research claim) |
| E5 | Full ablation | `run_evaluation.ipynb` | M2-generated vs GT vs random queries | 5-row AUC table |

The **headline of the writeup** is **E4 + E5**. E1/E2/E3 are supporting evidence.

---

## 1. The data prepared by `data_collection_preparation.ipynb`

The data notebook is the upstream producer for **every** evaluation step.

### 1.1 Files produced

```
processed/
├── train_processed.jsonl              # All 7K+ records with annotation + best frame
├── train_full_processed.jsonl         # Same, unfiltered backup
├── val_processed.jsonl
├── test_processed.jsonl
├── m1_train_subset.jsonl              # Official-QVH schema (for M1 audits)
├── m1_val.jsonl                       # ditto for val
├── m2_train_pairs.jsonl               # ~1.6K (frame, query) — single-frame M2 data
├── m2_train_pairs_multi.jsonl         # ~9-10K (frame, query) — multi-frame aug (3 per query)
├── m2_val_pairs.jsonl                 # ~300 val pairs for early stopping
├── consistency_eval.jsonl             # 600 pairs (300 pos / 300 cross-video neg)
├── consistency_eval_hard.jsonl        # 600 pairs (within-video neg, harder)
├── consistency_eval_query_mismatch.jsonl  # 600 pairs (same frame+video, wrong query)
├── thumbnail_index.jsonl              # Real YouTube thumbnails per record
├── m2_image_emb.npz                   # Cached CLIP-img for all M2 frames
├── thumbnail_image_emb.npz            # Cached CLIP-img for real thumbnails
├── category_map.json                  # qid -> category for stratification
└── manual_gold.jsonl                  # (optional) 50 hand-labeled pairs for gold AUC
```

### 1.2 What each eval set looks like

**E3 / cross-video positive**:
```json
{ "pair_id": "pos_707",
  "label": 1,
  "thumb_vid":  "Exr8FNpynKo_210.0_360.0",
  "frame_path": ".../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "thumb_query": "Man in red hoodie stretches on the street.",
  "video_vid":  "Exr8FNpynKo_210.0_360.0"  }                # same as thumb_vid
```

**E3 / cross-video negative**:
```json
{ "pair_id": "neg_707_8520",
  "label": 0,
  "thumb_vid":  "Exr8FNpynKo_210.0_360.0",
  "frame_path": ".../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "thumb_query": "Man in red hoodie stretches on the street.",  # query about thumb_vid
  "video_vid":  "FBcDXzzxjiA_510.0_660.0",                       # DIFFERENT video
  "neg_query_cos": 0.32 }                                         # cos(q_thumb, q_videoref)
```

**E4 / within-video hard negative**:
```json
{ "pair_id": "hardneg_707_clip60",
  "label": 0,
  "thumb_vid":  "Exr8FNpynKo_210.0_360.0",
  "frame_path": ".../qvh_frames_within_neg/Exr8FNpynKo_..._clip60.jpg",
  "thumb_query": "Man in red hoodie stretches on the street.",
  "video_vid":  "Exr8FNpynKo_210.0_360.0",                       # SAME video
  "out_of_window_clip_id": 60 }                                   # clip from OUTSIDE moment
```

**E5 / query-mismatch negative**:
```json
{ "pair_id": "qmiss_707_with_q9850",
  "label": 0,
  "thumb_vid":  "Exr8FNpynKo_210.0_360.0",
  "frame_path": ".../qvh_frames/Exr8FNpynKo_..._.jpg",              # SAME frame as positive
  "thumb_query": "Woman in red interviews a man speaking in Myanmar.",  # WRONG query
  "video_vid":  "Exr8FNpynKo_210.0_360.0",                              # SAME video
  "cos_correct_wrong": 0.18 }                                            # cos(correct_q, wrong_q)
```

### 1.3 Sanity tests for the prepared data

After running `data_collection_preparation.ipynb`, these checks must pass:

| Check | Where | Pass condition |
|---|---|---|
| All M1 features extracted | `Moment_DETR/moment_detr/features/clip_features/` | ≥9,000 .npz files |
| M2 train pairs aligned | `m2_train_pairs.jsonl` | every record has `frame_path` that exists on disk |
| M2 val pairs aligned | `m2_val_pairs.jsonl` | same |
| Cross-video eval balanced | `consistency_eval.jsonl` | ~300 pos + ~300 neg |
| Within-video eval populated | `consistency_eval_hard.jsonl` | ~300 pos + ~300 neg, frames in `qvh_frames_within_neg/` |
| Query-mismatch eval populated | `consistency_eval_query_mismatch.jsonl` | ~300 pos + ~300 neg, all `cos_correct_wrong < 0.4` |
| Real thumbnails downloaded | `qvh_thumbnails/` | ≥1,000 .jpg files |

---

## 2. E1 — M1 sanity (does the released checkpoint produce sensible saliency?)

**Purpose**: verify before using M1 in anything downstream.

**How**:
```python
m1 = MomentDETRFeaturePredictor(ckpt_path=CKPT, repo_root=...)
# Pick a random val record; encode its query and load its features
saliency = m1_predict_saliency(v_feat, gt_query)
# Plot saliency vs ground-truth relevant_window
```

**Acceptance**:
- Saliency values vary across clips (not uniform)
- The peak should lie inside or near the GT relevant_window for ≥ 80% of samples
- If saliency is uniform / random, M1 ckpt didn't load correctly → debug before continuing

---

## 3. E2 — M2 intrinsic evaluation (is the trained ClipCap producing useful queries?)

**Purpose**: validate M2 before plugging it into the consistency pipeline.

**Metrics** (computed at end of `train_clipcap.ipynb`):

| Metric | Definition | Target |
|---|---|---|
| CLIP-text-sim to GT | `cos(CLIP_text(M2_query), CLIP_text(GT_query))` averaged over val set | **≥ 0.65** |
| Vocab coverage | fraction of M2 tokens that decode to valid English (not gibberish) | **≥ 0.95** |
| Length distribution | mean / std of generated query lengths | Should roughly match GT (~10 ± 4 tokens) |
| BLEU-1 / BLEU-2 (legacy) | unigram / bigram overlap | Report but don't trust as primary |

**Acceptance**: CLIP-text-sim ≥ 0.55 means M2 is generating meaningfully similar queries to GT. < 0.45 indicates the model collapsed to repetitive captions.

**Qualitative**: print 20 (thumbnail, GT query, M2 query) triples. Eyeball them.

---

## 4. E3 — R1 cross-video smoke test (the baseline ceiling)

**Purpose**: confirm the score formula works when both query and visuals agree.

**Setup**: `consistency_eval.jsonl` (300 pos + 300 cross-video neg).

**Configuration**: `alpha=0.5, tau=0.1, top_p=0.25`, GT queries used (not M2).

**Metric**: AUC, Cohen's d.

**Expected**: AUC ≈ 0.85–0.95. (Achieved: **0.946**.)

**Diagnostic**: CLIP-only baseline must be ≈ 0.5 (random) — if higher, the eval set has a visual leak.

---

## 5. E4 — Within-video hard eval (the temporal-grounding claim)

**Purpose**: this is the research contribution — *the pipeline localizes the
correct moment within a video*.

**Setup**: `consistency_eval_hard.jsonl` (300 pos + 300 within-video neg). The
negatives use the SAME video as the positive, but a frame from OUTSIDE the
relevant_window.

**Configuration**: `alpha=0.5`, GT queries.

**Metric**: AUC, Cohen's d.

**Expected**: AUC ≈ **0.65–0.80**. This is the honest research number.

| Outcome | Interpretation |
|---|---|
| ≥ 0.75 | Strong temporal-grounding claim |
| 0.65–0.75 | Moderate claim; report with limitations |
| < 0.60 | M1 saliency isn't focusing well; report as a finding |

---

## 6. E5 — Full ablation (M2-generated queries vs ceiling vs random)

**Purpose**: the headline ablation table for the paper.

**Setup**: any one of the three eval sets (cross-video is most common for
ablation reporting).

**Five conditions**:

| Source | Description | Expected AUC | Role |
|---|---|---|---|
| `gt_query` | Ground-truth QVH annotation | 0.85–0.95 | Ceiling |
| `clipcap_k3` | 3 nucleus-sampled queries from M2, mean-aggregated | 0.75–0.85 | **Pipeline as deployed** |
| `clipcap_k1` | Single beam-3 query from M2 | 0.70–0.80 | Single-query baseline |
| `thumb_only` | α = 0 (no query path) | 0.65–0.75 | Visual-only baseline |
| `random_query` | Random query from the set | ≈ 0.50 | Control |

**Headline claim**: `clipcap_k3 AUC` minus `random_query AUC` = M2's contribution.

**Auxiliary**: alpha sweep on query-mismatch eval shows the query path is
genuinely informative (you already have: α=0 → 0.50, α=1 → 0.97, confirming
M1 saliency responds to the query).

---

## 7. Manual gold set (optional, Day-3 nice-to-have)

50 thumbnails hand-labeled `consistent / inconsistent`. Compute AUC, compare
against synthetic AUC. If both agree, the synthetic eval is a faithful proxy.

---

## 8. Frame-vs-real-thumbnail domain-gap probe (optional)

Re-run E3 using `qvh_thumbnails/{youtube_id}.jpg` (real YouTube thumbnails)
instead of extracted frames. AUC will drop — the difference quantifies the
distribution gap between training data and real inference.

---

## 9. Outputs landing on Drive

After running the full evaluation suite:

```
outputs/
├── checkpoints/clipcap_best.pt
├── predictions/
│   ├── consistency_gt_query.jsonl           # E3
│   ├── consistency_hard.jsonl               # E4
│   ├── consistency_qmismatch.jsonl          # E5 query-mismatch
│   ├── scores_gt_query.jsonl                # ablation rows
│   ├── scores_clipcap_k3.jsonl
│   ├── scores_clipcap_k1.jsonl
│   ├── scores_thumb_only.jsonl
│   └── scores_random_query.jsonl
└── figures/
    ├── score_distribution_gt.png            # E3
    ├── score_distribution_hard.png          # E4
    ├── score_distribution_qmismatch.png     # E5
    ├── ablation_table.csv                   # E5
    └── alpha_sweep.png                      # E5 alpha ablation
```

---

## 10. Reading guide: what to report in the paper

| Section of writeup | Use number from |
|---|---|
| "Score formula is well-calibrated" | E5 alpha sweep (α=0 → 0.50, α=1 → 0.97) |
| "Pipeline reliably distinguishes matched videos" | E3 (AUC ≈ 0.95) |
| "Pipeline performs temporal grounding" | **E4 (AUC ≈ 0.70–0.80)** ★ |
| "M2-generated queries contribute substantial signal" | E5 clipcap_k3 vs random_query (gap) |
| "Cost of using generated queries vs GT" | E5 clipcap_k3 vs gt_query (gap) |
| "Limitations: short moments, real thumbnails" | E8 domain gap + qualitative failure cases |
