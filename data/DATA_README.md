# Data README — Thumbnail-Video Consistency Project

> Every artifact produced by `notebooks/data_collection_preparation.ipynb`, what it
> looks like, how it was built, and where it's used downstream.

The actual data lives on Google Drive at
`/content/drive/MyDrive/thumbnail_project/`. This document is the
**reproducibility contract**: if you have the same JSONL files, the same eval
results follow.

---

## 0. Data root layout

```
/content/drive/MyDrive/thumbnail_project/
├── processed/                              ← canonical JSONL / NPZ artifacts (~30 MB)
├── qvh_videos/                             ← raw .mp4 clips (~5-10 GB)
├── qvh_frames/                             ← single-frame extractions (~30 MB)
├── qvh_frames_multi/                       ← multi-frame extractions (~120 MB)
├── qvh_frames_within_neg/                  ← out-of-window frames for hard eval (~5 MB)
├── qvh_thumbnails/                         ← real YouTube thumbnails (~150 MB)
├── moment_detr/                            ← upstream Moment-DETR (code + ckpt)
├── outputs/                                ← prediction outputs from eval (~50 MB)
├── checkpoints/                            ← trained ClipCap (~50 MB)
├── moment_detr_features.tar.gz             ← M1 features (~7.5 GB, packed)
└── clip_aug_text_features.zip              ← auxiliary, unused in V1
```

What goes to GitHub: **none of the data files** (they're gitignored). Only this
README + the recipe for rebuilding them.

---

## 1. `processed/train_processed.jsonl`

| | |
|---|---|
| **Source** | `notebooks/data_collection_preparation.ipynb` Section 5 + v11 reconcile |
| **Records** | ~3,100 (one per QVHighlights training sample) |
| **Used by** | Source of truth for every downstream cell — multi-frame aug, eval builders, M2 single-frame fallback |

### Schema

```json
{
  "qid": 707,
  "query": "Man in red hoodie stretches on the street.",
  "duration": 150.0,
  "vid": "Exr8FNpynKo_210.0_360.0",
  "youtube_id": "Exr8FNpynKo",
  "seg_start": 210.0,
  "seg_end": 360.0,
  "relevant_windows": [[20, 60]],
  "relevant_clip_ids": [10, 11, 12, ..., 30],
  "saliency_scores": [[3,3,3], [4,3,4], ...],
  "best_clip_id": 22,
  "best_timestamp": 45.0,
  "mean_saliency": 3.667,
  "selection_reason": "annotator_agreement",
  "category": "daily_vlog",
  "frame_path": "/.../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "video_downloaded": true,
  "clip_score": null
}
```

---

## 2. `processed/m1_train_subset.jsonl` and `m1_val.jsonl`

| | |
|---|---|
| **Source** | Section 14 (filters official `highlight_train_release.jsonl` by subset qids) |
| **Records** | Train: ~3,100. Val: 1,550 |
| **Used by** | Moment-DETR official training script (if you ever retrain M1) |

Schema is the **original QVHighlights format**, used for compatibility with
`moment_detr/train.py`:

```json
{
  "qid": 707,
  "query": "Man in red hoodie stretches on the street.",
  "duration": 150.0,
  "vid": "Exr8FNpynKo_210.0_360.0",
  "relevant_windows": [[20, 60]],
  "relevant_clip_ids": [10, 11, ..., 30],
  "saliency_scores": [[3,3,3], [4,3,4], ...]
}
```

---

## 3. `processed/m2_train_pairs.jsonl`

| | |
|---|---|
| **Source** | Section 10 (or v11 reconcile if frames were pre-extracted) |
| **Records** | ~2,824 — one per record that has a successfully extracted frame |
| **Used by** | M2 training (single-frame fallback) — `cfg['data']['train_jsonl']` in `configs/clipcap.yaml` |

### Schema

```json
{
  "qid": 707,
  "frame_path": "/.../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "query": "Man in red hoodie stretches on the street.",
  "vid": "Exr8FNpynKo_210.0_360.0",
  "best_timestamp": 45.0,
  "mean_saliency": 3.667,
  "clip_score": null
}
```

`clip_score` is `null` for center-frame extractions (v2/v3) and a float in
[0, 1] for CLIP-ranked (v1).

---

## 4. ★ `processed/m2_train_pairs_multi.jsonl` (RECOMMENDED training set)

| | |
|---|---|
| **Source** | v9 multi-frame augmentation cell |
| **Records** | ~8,400 — K=3 frames × ~2,800 records |
| **Used by** | M2 training — **★ default in `configs/clipcap.yaml`** (manually edit `cfg['data']['train_jsonl']`) |

For each annotated record we extract K=3 frames evenly spaced across the
`relevant_clip_ids`. Each (frame, query) becomes a separate training example.

### Schema

```json
{
  "qid": 707,
  "vid": "Exr8FNpynKo_210.0_360.0",
  "frame_path": "/.../qvh_frames_multi/Exr8FNpynKo_210.0_360.0_clip15.jpg",
  "query": "Man in red hoodie stretches on the street.",
  "clip_id": 15,
  "best_timestamp": 31.0,
  "mean_saliency": 3.667,
  "augment_id": "Exr8FNpynKo_210.0_360.0_clip15"
}
```

Three records share the same `qid` and `query` but have different `frame_path`
and `clip_id`. This is intentional — it's the augmentation signal.

---

## 5. `processed/m2_val_pairs.jsonl`

| | |
|---|---|
| **Source** | Section 15 (v7 self-contained builder) |
| **Records** | ~300 (stratified by `mean_saliency` tertile) |
| **Used by** | M2 early-stopping (CLIP-text similarity) in `train_clipcap.ipynb` |

Same schema as `m2_train_pairs.jsonl`. Drawn from the **val split** of QVH —
so M2 never sees these queries during training.

---

## 6. `processed/consistency_eval.jsonl` — E3 cross-video eval

| | |
|---|---|
| **Source** | Section 16 |
| **Records** | 600 (300 positives + 300 cross-video negatives) |
| **Used by** | R1 smoke test (`sanity_consistency.ipynb`), full ablation (`run_evaluation.ipynb`) |

The **default eval set**. Positives use a thumbnail frame against the same video.
Negatives use the same thumbnail frame against a different video (with
`cos(q_thumb, q_neg_video) < 0.6` to avoid accidental positives).

### Schema (positive)

```json
{
  "pair_id": "pos_707",
  "label": 1,
  "thumb_qid": 707,
  "thumb_vid": "Exr8FNpynKo_210.0_360.0",
  "frame_path": "/.../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "thumb_query": "Man in red hoodie stretches on the street.",
  "video_qid": 707,
  "video_vid": "Exr8FNpynKo_210.0_360.0",
  "video_query_for_ref": "Man in red hoodie stretches on the street."
}
```

### Schema (negative — cross-video)

```json
{
  "pair_id": "neg_707_8520",
  "label": 0,
  "thumb_qid": 707,
  "thumb_vid": "Exr8FNpynKo_210.0_360.0",
  "frame_path": "/.../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "thumb_query": "Man in red hoodie stretches on the street.",
  "video_qid": 8520,
  "video_vid": "FBcDXzzxjiA_510.0_660.0",
  "video_query_for_ref": "A couple is sitting in a bath together...",
  "neg_query_cos": 0.32
}
```

**Result we measured**: AUC = 0.946 with GT queries (the pipeline ceiling).

---

## 7. ★ `processed/consistency_eval_hard.jsonl` — E4 within-video eval

| | |
|---|---|
| **Source** | v14 hard builder cell |
| **Records** | 600 (300 positives + 300 within-video negatives) |
| **Used by** | The **research claim** — does the pipeline localize the correct moment WITHIN a video? |

Positives are the same as `consistency_eval.jsonl`. Negatives use the **same
video** as the positive, but a frame from OUTSIDE the `relevant_window`. Forces
the model to discriminate by **timing**, not by visual content.

### Schema (negative — within-video)

```json
{
  "pair_id": "hardneg_707_clip60",
  "label": 0,
  "thumb_qid": 707,
  "thumb_vid": "Exr8FNpynKo_210.0_360.0",
  "frame_path": "/.../qvh_frames_within_neg/Exr8FNpynKo_210.0_360.0_clip60.jpg",
  "thumb_query": "Man in red hoodie stretches on the street.",
  "video_qid": 707,
  "video_vid": "Exr8FNpynKo_210.0_360.0",
  "out_of_window_clip_id": 60
}
```

---

## 8. `processed/consistency_eval_query_mismatch.jsonl` — E5 query-mismatch eval

| | |
|---|---|
| **Source** | v14 qmiss builder cell |
| **Records** | 600 (300 positives + 300 query-mismatch negatives) |
| **Used by** | α-sweep analysis — isolates the query path's contribution |

Same frame, same video, **different query**. Tests whether the consistency
score genuinely depends on the query (rather than just thumbnail-video
visual similarity).

### Schema (negative — query mismatch)

```json
{
  "pair_id": "qmiss_707_with_q9850",
  "label": 0,
  "thumb_qid": 707,
  "thumb_vid": "Exr8FNpynKo_210.0_360.0",
  "frame_path": "/.../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "thumb_query": "Woman in red interviews a man speaking in Myanmar.",
  "video_qid": 707,
  "video_vid": "Exr8FNpynKo_210.0_360.0",
  "correct_query_was": "Man in red hoodie stretches on the street.",
  "wrong_query_from": 9850,
  "cos_correct_wrong": 0.18
}
```

**Result we measured**: α-sweep:
- α=0.0 (thumb-only): AUC=0.497 ≈ random ✓
- α=0.5 (balanced): AUC=0.752
- α=1.0 (query-only): AUC=0.972 ← proves the query path works

---

## 9. `processed/manual_gold.jsonl` (optional V1.1)

| | |
|---|---|
| **Source** | v12 gold builder + manual labeling (see [docs/LABELING_GUIDE.md](LABELING_GUIDE.md)) |
| **Records** | 50 (25 synthetic-positive + 25 synthetic-negative) |
| **Used by** | Validates synthetic eval against human judgment (v13 evaluator) |

Schema = `consistency_eval.jsonl` schema + two manual fields:

```json
{
  "pair_id": "pos_707",
  "label_synthetic": 1,
  "label_human": 1,          // 0=inconsistent, 1=consistent, 0.5=partial
  "confidence": 1,            // 1=sure, 2=somewhat, 3=guessing
  "note": "Frame and video match clearly.",
  "thumb_vid": "Exr8FNpynKo_210.0_360.0",
  "frame_path": "...",
  "thumb_query": "...",
  "video_vid": "Exr8FNpynKo_210.0_360.0",
  "video_query_for_ref": "..."
}
```

---

## 10. `processed/thumbnail_index.jsonl`

| | |
|---|---|
| **Source** | Section 13 (v4 direct-HTTP download from `i.ytimg.com`) |
| **Records** | ~1,733 (one per unique `youtube_id`) |
| **Used by** | Domain-gap experiment (E8) — substitutes real thumbnails for extracted frames |

```json
{
  "qid": 707,
  "vid": "Exr8FNpynKo_210.0_360.0",
  "youtube_id": "Exr8FNpynKo",
  "thumbnail_path": "/.../qvh_thumbnails/Exr8FNpynKo.jpg"
}
```

---

## 11. `processed/m2_image_emb.npz` and `thumbnail_image_emb.npz`

| | |
|---|---|
| **Source** | Section 17 (v5/v6 with defensive CLIP-image bypass) |
| **Records** | `m2_image_emb`: ~2,824 embeddings keyed by `qid`. `thumbnail_image_emb`: ~1,201 keyed by `youtube_id` |
| **Used by** | Speeds up evaluation by avoiding re-encoding images |

Schema (numpy):

```python
np.load('m2_image_emb.npz')
  → {
    'qids': np.array([707, 8520, ...]),       # (N,) int64
    'emb' : np.array([[...], [...], ...]),    # (N, 512) float32, L2-normalized
  }
```

---

## 12. `processed/category_map.json`

| | |
|---|---|
| **Source** | Section 6 |
| **Records** | qid → category dict for all train records |
| **Used by** | Stratified subset selection; optional reporting in writeup |

```json
{
  "707": "daily_vlog",
  "8520": "daily_vlog",
  "9850": "news",
  ...
}
```

Categories: `daily_vlog`, `travel_vlog`, `news`, `other`.

---

## 13. `processed/download_log.json`

Debug-only artifact from yt-dlp. Records which videos succeeded/failed during
Section 8 downloads. Not used by any downstream cell.

---

## Where each artifact appears in the pipeline

```
                                  ┌─ M2 single-frame ──── m2_train_pairs.jsonl
                                  │
                                  ├─ M2 multi-frame  ──── m2_train_pairs_multi.jsonl ★
M2 training (ClipCap) ────────────┤
                                  ├─ Early stopping ────── m2_val_pairs.jsonl
                                  │
                                  └─ Query generation ──── consistency_eval.jsonl (frame_path)

                                  ┌─ Saliency / spans ──── (M1 features from local SSD)
M1 inference (frozen ckpt) ───────┤
                                  └─ Text encoding ──────── thumb_query field of eval JSONL

                                  ┌─ E3 cross-video ────── consistency_eval.jsonl
Consistency scorer ───────────────┼─ E4 within-video ───── consistency_eval_hard.jsonl
                                  ├─ E5 query-mismatch ─── consistency_eval_query_mismatch.jsonl
                                  ├─ E6 manual gold ────── manual_gold.jsonl (optional)
                                  └─ E8 domain-gap ──────── thumbnail_index.jsonl + qvh_thumbnails/
```

---

## What's pushed to GitHub vs Drive

| Where | What | Reason |
|---|---|---|
| **GitHub** | This `DATA_README.md`, configs, src/, notebooks, tests | Reproducibility recipe |
| **Drive** | All JSONL/NPZ artifacts, videos, frames, features | Too large for git, lots of small files |
| **GitHub** (optional) | Small JSONLs (consistency_eval, m2_val) for offline reproducibility | If <10 MB total |
| **GitHub** ❌ | Videos (.mp4), all .npz, all frame .jpg files | Always too large |

If you want to push small JSONLs anyway, edit `.gitignore` to un-ignore them:

```
# In .gitignore, remove these or scope them:
processed/qvh_*           # ignore frames, not jsonls
outputs/checkpoints/      # ignore .pt files
outputs/predictions/*.jsonl  # but keep predictions/  ← controversial
```

For V1 the safest answer: **only push this README**. Anyone wanting to reproduce
runs the data_collection notebook with QVHighlights and gets identical files.

---

## Reproducing from scratch

```
Prerequisite: QVHighlights annotations at moment_detr/data/highlight_*_release.jsonl
              moment_detr_features.tar.gz on Drive (provided by upstream)

1. Open notebooks/data_collection_preparation.ipynb
2. Run Sections 1-7  → train_processed, val_processed, test_processed
3. Run Section 8     → downloads ~2,800-3,100 videos (1-2 hours, yt-dlp dependent)
4. Run Section 9     → extracts single-frame per video (~10 min)
5. Run Section 10    → m2_train_pairs.jsonl
6. Run v11 reconcile → rebuilds m2_pairs from disk if frames pre-existed
7. Run v9 multi-frame aug → m2_train_pairs_multi.jsonl (~10 min)
8. Run Section 13    → downloads real thumbnails (~2 min)
9. Run Section 14    → M1 official-format JSONL
10. Run Section 15   → m2_val_pairs.jsonl (~10-20 min, includes new val downloads)
11. Run Section 16   → consistency_eval.jsonl
12. Run v14          → consistency_eval_hard.jsonl + consistency_eval_query_mismatch.jsonl
13. Run Section 17   → m2_image_emb.npz + thumbnail_image_emb.npz
14. Run v12          → manual_gold_template.jsonl
15. (Manual)         → label gold set, save as manual_gold.jsonl
16. Run v13          → human vs synthetic agreement
17. Run Section 18   → manifest check, expect 9-11 OK
```

Total wall time on Colab T4 with already-downloaded videos: ~40 min.
First-time setup (with video downloads): ~2-3 hours.

---

## Acceptance test (paste in fresh Colab cell)

```python
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive/thumbnail_project')

checks = {
    'train_processed.jsonl'                  : 2000,
    'm1_train_subset.jsonl'                  : 2000,
    'm1_val.jsonl'                           : 1500,
    'm2_train_pairs.jsonl'                   : 1500,
    'm2_train_pairs_multi.jsonl'             : 4500,
    'm2_val_pairs.jsonl'                     : 200,
    'consistency_eval.jsonl'                 : 500,
    'consistency_eval_hard.jsonl'            : 500,
    'consistency_eval_query_mismatch.jsonl'  : 500,
    'thumbnail_index.jsonl'                  : 800,
    'manual_gold_template.jsonl'             : 49,
    'category_map.json'                      : 0,
    'm2_image_emb.npz'                       : 0,
    'thumbnail_image_emb.npz'                : 0,
}
ok = True
for name, min_lines in checks.items():
    p = DRIVE / 'processed' / name
    n = sum(1 for _ in open(p)) if (p.exists() and name.endswith('.jsonl')) else 0
    sz = p.stat().st_size if p.exists() else 0
    if name.endswith('.jsonl'):
        passed = n >= min_lines
        print(f"  {name:43s}  {n:>7,d} lines  {'OK' if passed else 'FAIL'}")
        ok = ok and passed
    else:
        print(f"  {name:43s}  {sz/1024:>7,.0f} KB  {'OK' if sz > 0 else 'MISSING'}")
print(f"\nData acceptance: {'PASS' if ok else 'FAIL'}")
```

---

## Citing the data

If publishing, cite both QVHighlights (the source annotations) and ClipCap (for
the M2 architecture):

```bibtex
@inproceedings{lei2021qvhighlights,
  title     = {QVHighlights: Detecting Moments and Highlights in Videos with Natural Language Queries},
  author    = {Lei, Jie and Berg, Tamara L and Bansal, Mohit},
  booktitle = {NeurIPS},
  year      = {2021}
}

@article{mokady2021clipcap,
  title   = {ClipCap: CLIP Prefix for Image Captioning},
  author  = {Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal = {arXiv:2111.09734},
  year    = {2021}
}
```
