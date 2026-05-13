# M1 Training Pipeline — Practical Reference

**Last updated:** 2026-05-07

---

## Feature Files You Have

```
Moment_DETR_REPO/
  moment_detr_features.tar.gz      ← SlowFast + CLIP video features
  clip_aug_text_features.zip        ← CLIP text/query embeddings (augmented)
```

**Critical first step before any training:** Inspect the feature files to understand their exact structure.

```bash
# Unpack and inspect
tar -xzf moment_detr_features.tar.gz
ls features/
# Expected: slowfast_features/, clip_features/, clip_text_features/

# Check a sample video feature
python3 -c "
import numpy as np, os
f = np.load('features/clip_features/RoripwjYFp8_360.0_510.0.npz')
print(f.files)          # should be ['features']
print(f['features'].shape)  # (n_clips, 512) — varies per video
"

# Check a text feature
python3 -c "
import numpy as np
f = np.load('features/clip_text_features/qid0.npz')
print(f.files)          # ['pooler_output', 'last_hidden_state']
print(f['last_hidden_state'].shape)   # (max_q_l, 512)
print(f['pooler_output'].shape)       # (512,)
"
```

---

## Data Loading Pipeline (Exactly How It Works)

From `start_end_dataset.py`:

### Per-Sample Loading (`__getitem__`)
```python
# 1. Load query features
q_feat = np.load(f"clip_text_features/qid{qid}.npz")["last_hidden_state"]
# shape: (≤32, 512), L2-normalized, trimmed to max_q_l=32

# 2. Load video features (multiple dirs if concat)
v_feat = np.load(f"clip_features/{vid}.npz")["features"]
# shape: (n_clips, 512), L2-normalized, trimmed to max_v_l=75

# 3. Append TEF (Temporal Endpoint Features)
tef_st = arange(0, n_clips) / n_clips          # [0/75, 1/75, ..., 74/75]
tef_ed = tef_st + 1.0 / n_clips
tef = stack([tef_st, tef_ed], dim=1)           # (n_clips, 2)
video_feat = cat([v_feat, tef], dim=1)         # (n_clips, 514)

# 4. Build span labels
# windows = [[26, 36]] seconds
# normalized: center = (26+36)/2 / 150 = 0.207, width = 10/150 = 0.067
# stored as [0.207, 0.067]

# 5. Build saliency labels
# From relevant_clip_ids and saliency_scores, pick hard_pos/hard_neg/easy_pos/easy_neg
```

### Saliency Label Construction (Critical Detail)
```python
agg_scores = sum(scores, axis=1)     # sum over 3 annotators per clip
sort_indices = argsort(agg_scores)   # ascending

hard_pos = relevant_clip_ids[-1]     # highest-scored relevant clip
hard_neg = relevant_clip_ids[0]      # lowest-scored relevant clip
easy_pos = random.choice(relevant_clip_ids)
easy_neg = random.choice(outside_relevant_clips)

# Final saliency labels: [hard_pos, easy_pos], [hard_neg, easy_neg]
# Shape: (batch, 2) for both pos and neg
```

This means the saliency loss receives 2 positive-negative pairs per sample. The loss penalizes `neg_score + margin > pos_score`.

---

## Running M1 on Your Subset

After inspecting features, adapt the training command:

```bash
# Minimal working command for 2000-sample subset
PYTHONPATH=. python moment_detr/train.py \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path data/highlight_train_release.jsonl \
  --eval_path data/highlight_val_release.jsonl \
  --eval_split_name val \
  --v_feat_dirs features/clip_features \
  --v_feat_dim 512 \
  --t_feat_dir features/clip_text_features/ \
  --t_feat_dim 512 \
  --bsz 32 \
  --n_epoch 200 \
  --lr 1e-4 \
  --data_ratio 0.286 \    # 2000/7000 ≈ 0.286 for subset
  --results_root results \
  --exp_id clip_only_subset_2000
```

**Important:** `data_ratio` takes the first N% of records — it's not the same as your stratified subset. For reproducibility, pass a custom `train_path` pointing to your processed `train_processed.jsonl`.

---

## Expected Performance Degradation (Subset vs Full)

Based on the official numbers and typical scaling behavior:

| Split Size | Expected R1@0.5 | Expected mAP |
|---|---|---|
| Full (~7000) | 53.23% | 30.58% |
| 2000 subset | ~42–47% | ~22–26% |
| 500 samples | ~30–35% | ~15–20% |

The degradation is steep for moment retrieval because the model needs diverse query-video pairs to generalize the cross-modal attention. Your stratified sampling (daily/travel/news) reduces distribution shift within the subset.

---

## Feature Dimensions Reference

| Feature | Dim | Notes |
|---|---|---|
| SlowFast | 2304 | slow (2048) + fast (256) |
| CLIP visual | 512 | ViT-B/32, L2-norm |
| CLIP+SlowFast | 2816 | concatenated, L2-norm each separately |
| CLIP+SlowFast+TEF | 2818 | append 2 TEF dims |
| CLIP+TEF | 514 | official CLIP-only config |
| CLIP text | 512 | per-token embeddings |

---

## What `clip_aug_text_features.zip` Contains

This is a feature file not in the standard release. "aug" likely refers to query augmentation — the same query text encoded with minor perturbations (dropout, paraphrase). Used in some improved Moment-DETR variants to regularize text encoding. You can use these as a drop-in replacement for `clip_text_features` if they follow the same `qid{N}.npz` naming convention.

---

## Evaluation Metrics Explained

### Moment Retrieval (MR)
- **R1@0.5:** Recall@1 with IoU threshold 0.5. Is the top-ranked prediction overlapping ≥50% with ground truth?
- **R1@0.7:** Stricter IoU threshold. Penalizes approximate localizations.
- **mAP:** Mean Average Precision across IoU thresholds [0.5, 0.55, ..., 0.95]. Most comprehensive single metric.

### Highlight Detection (HL)
- **HL Hit1:** Is the predicted top-1 saliency clip within the ground-truth relevant window?
- **HL mAP:** Saliency ranking quality across all clips. Evaluated at "Fair" (score≥1), "Good" (≥2), "VeryGood" (≥3) thresholds.

For your project, **HL mAP (Good)** is the most useful metric because it measures whether your model correctly scores the truly salient clips higher — which is what your consistency pipeline depends on.

---

## Common Failure Modes to Anticipate

### 1. Feature File Mismatch
The `vid` strings in your processed JSONL must exactly match the feature filenames. Check:
```python
# vid = "RoripwjYFp8_360.0_510.0"
# feature file = "features/clip_features/RoripwjYFp8_360.0_510.0.npz"
```
Float formatting issues (e.g., `360.0` vs `360`) will silently fail.

### 2. Missing Features for Subset Videos
Your 2000-sample subset was selected from the full train set. The feature files must exist for ALL 2000 videos. If some are missing, the dataloader will raise `FileNotFoundError` during training.

### 3. ctx_l Mismatch
`ctx_l = len(video_feat)` is set per sample. Saliency labels (`relevant_clip_ids`) must all be < `ctx_l`. If any `relevant_clip_id >= 75`, it gets clamped. This is noted in the code as a known minor bug.

### 4. Gradient Explosion
`grad_clip=0.1` is aggressive. If you see NaN losses early in training, the loss scale is too high. Check that features are L2-normalized and that span labels are in [0, 1].
