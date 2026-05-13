# Moment-DETR Architecture — Deep Technical Reference

**Source:** `moment_detr/model.py`, `start_end_dataset.py`, `matcher.py`, `config.py`, `train.sh`  
**Last updated:** 2026-05-07

---

## High-Level Architecture

Moment-DETR adapts DETR (DEtection TRansformer) from object detection to **temporal moment localization**. Instead of detecting bounding boxes in 2D space, it detects temporal windows in 1D time.

```
Video features  (L_vid, D_vid)  ──┐
                                  ├──► Input Projection ──► Concat ──► Transformer Encoder+Decoder
Text features   (L_txt, D_txt)  ──┘                                           │
                                                                              ▼
Learnable query embeddings (10, 256)  ──────────────────────────►      Decoder queries
                                                                               │
                                                                    ┌──────────┴──────────┐
                                                                    ▼                     ▼
                                                             span_embed (MLP)     class_embed (Linear)
                                                                    │                     │
                                                             pred_spans (10,2)    pred_logits (10,2)
                                                             (center, width)    (foreground/background)

                                                          + saliency_proj: vid_memory → (L_vid,) scores
```

---

## Input Representations

### Video Features
- **Format:** Pre-extracted per 2-second clip. Each clip → one feature vector.
- **SlowFast:** 2304-dim (slow pathway 2048 + fast pathway 256)
- **CLIP visual:** 512-dim (ViT-B/32 image encoder, frame-averaged per clip)
- **Combined:** 2304 + 512 = **2816-dim** (full model) or **512-dim** (CLIP-only model)
- **TEF (Temporal Endpoint Features):** 2-dim [start/L, end/L] appended → +2 to video dim
- **Max sequence length:** 75 clips (150 seconds at 2s/clip)
- **Normalization:** L2-normalized per feature vector

### Text / Query Features
- **Format:** Pre-extracted CLIP text embeddings, one per query
- **CLIP text:** 512-dim per token (last_hidden_state), up to 32 tokens
- **Normalization:** L2-normalized
- **txt_drop_ratio:** Optional token dropout during training (augmentation, not used in official)

### Key Dimension Mapping
```
ctx_mode = "video_tef"
v_feat_dim = 512 (CLIP-only) + 2 (TEF) = 514  ← actual input_vid_proj input dim
t_feat_dim = 512
hidden_dim = 256  ← all internal representations
```

---

## Transformer Architecture

```python
# 2 encoder layers + 2 decoder layers
enc_layers = 2
dec_layers = 2
hidden_dim = 256
nheads = 8
dim_feedforward = 1024
dropout = 0.1
input_dropout = 0.5  # high dropout on input projections
```

### Input Projections
Before entering the transformer, both modalities are projected to `hidden_dim=256`:
```python
# 2-layer MLP with LayerNorm + ReLU + dropout
input_vid_proj: (D_vid) → 256  [2 LinearLayer blocks]
input_txt_proj: (D_txt) → 256  [2 LinearLayer blocks]
```
Note: `input_dropout=0.5` is very aggressive. This is regularization against feature noise.

### Positional Encoding
- Video: **Sine positional encoding** on clip indices (temporal position)
- Text: **Zero position** (positional encoding disabled for text by default, `use_txt_pos=False`)

This is an important design choice — text tokens are assumed to be order-independent (bag-of-tokens for cross-attention).

### Encoder
- Processes concatenated `[video (L_vid, 256), text (L_txt, 256)]` as a single sequence
- Self-attention allows video clips to attend to each other AND to text tokens
- Output: `memory` — contextualized representations for all positions

### Decoder
- 10 learnable query embeddings (`query_embed`) attend to the encoder memory
- Each query learns to specialize in detecting a different type of moment
- Cross-attention: decoder queries attend to the full video+text memory
- Output: `hs` — 10 contextualized query vectors

---

## Output Heads

### 1. Span Prediction (`span_embed`)
```python
span_embed = MLP(256 → 256 → 256 → 2, depth=3)
outputs_coord = span_embed(hs).sigmoid()  # (batch, 10, 2)
# output: (center_normalized, width_normalized) in [0, 1]
```
Center and width are both normalized by total video duration.

To convert back to seconds:
```python
center_sec = pred_center * (max_v_l * clip_len)  # = pred_center * 150
width_sec   = pred_width  * (max_v_l * clip_len)
start_sec   = center_sec - width_sec / 2
end_sec     = center_sec + width_sec / 2
```

### 2. Foreground Classification (`class_embed`)
```python
class_embed = Linear(256, 2)
outputs_class  # (batch, 10, 2) — foreground(0) vs background(1)
```
After inference, queries with high foreground probability are kept.

### 3. Saliency Scores (`saliency_proj`)
```python
saliency_proj = Linear(256, 1)
saliency_scores = saliency_proj(vid_mem).squeeze(-1)  # (batch, L_vid)
```
The video memory (encoder output for video positions) is projected to per-clip saliency scalars.
These are NOT the detection outputs — they are a separate saliency head supervised by human annotation.

---

## Loss Functions

### 1. Hungarian Matching (Required Before Loss)
Matches each ground-truth span to the best-scoring predicted query:
```python
cost = 10 * L1_span_cost + 1 * GIoU_cost + 4 * class_cost
```
Only matched predictions are supervised for span regression. Unmatched → treated as background.

Matching is done **without gradients** (Hungarian algorithm on CPU).

### 2. Span Regression Loss
```python
loss_span = L1(pred_span_cxw, gt_span_cxw)       # normalized center+width
loss_giou = 1 - GIoU(pred_span_xx, gt_span_xx)   # start+end format
```
Weight: `loss_span * 10 + loss_giou * 1`

### 3. Classification Loss
```python
loss_label = CrossEntropy(pred_logits, target_classes, weight=[1.0, 0.1])
# background (class 1) has 10x lower weight via eos_coef=0.1
```

### 4. Saliency Margin Loss
```python
loss_saliency = max(0, margin + neg_score - pos_score).mean()
# margin = 0.2
# pos: clips with highest annotation sum among relevant_clip_ids
# neg: clips with lowest annotation sum + easy negatives outside relevant region
```
This is a **ranking loss**, not regression. It just needs positives to score higher than negatives by a margin.

### 5. Optional Contrastive Alignment Loss
```python
# NCE loss between matched decoder queries and text tokens
loss_contrastive_align = NCE(proj_queries, proj_txt_mem, temperature=0.07)
```
Not used in the official CLIP-only checkpoint (disabled in `opt.json`).

### Total Loss
```python
total = 10*loss_span + 1*loss_giou + 4*loss_label + 1*loss_saliency
```

---

## Hyperparameters (Official Training Config)

| Parameter | Value | Notes |
|---|---|---|
| `lr` | 1e-4 | AdamW |
| `lr_drop` | 400 | StepLR decay to 1/10 at epoch 400 |
| `wd` | 1e-4 | Weight decay |
| `n_epoch` | 200 | Full training |
| `bsz` | 32 | Batch size |
| `grad_clip` | 0.1 | Gradient clipping (aggressive) |
| `num_queries` | 10 | Detection slots |
| `hidden_dim` | 256 | Internal dimension |
| `enc_layers` | 2 | Shallow encoder |
| `dec_layers` | 2 | Shallow decoder |
| `input_dropout` | 0.5 | Input projection dropout |
| `dropout` | 0.1 | Transformer dropout |
| `max_v_l` | 75 | Max clips per video |
| `max_q_l` | 32 | Max query tokens |

---

## Official Benchmark Results (CLIP-only model, val set)

| Metric | Score |
|---|---|
| MR R1@0.5 | **53.23%** |
| MR R1@0.7 | **34.00%** |
| MR mAP | **30.58%** |
| MR mAP@0.5 | **54.80%** |
| HL Hit1 (Good) | **66.26%** |
| HL mAP (Good) | **57.93%** |
| MR mAP (Long moments) | **41.27%** |
| MR mAP (Short moments) | **3.11%** ← critical weakness |

**Critical observation:** Performance on short moments (< ~10 seconds) is near-random. The model works by attending over 2-second clips — very short events are poorly localized because the span regression head lacks the resolution to distinguish adjacent clips.

---

## Critical Weaknesses (Design-Level)

### 1. Short Moment Localization Failure
mAP of 3.11% for short moments. The 2-second clip granularity creates a fundamental resolution floor. If a moment is 4 seconds long, the model must predict `width ≈ 0.027` (normalized). The decoder query specialization doesn't help here.

### 2. Saliency Supervision Fragility
The saliency loss uses positive/negative clip pairs sampled from the annotation. With only 1 positive and 1 negative (hard+easy = 2 each), the gradient signal is weak and stochastic per batch.

### 3. Text Position Encoding Disabled
`use_txt_pos=False` — text tokens have no positional information. Effectively treats the query as a bag of token embeddings. This works because CLIP text features already encode word order via attention, but wastes the transformer's sequence modeling capacity for text.

### 4. No Cross-Modal Pretraining
The official model trains from scratch on QVHighlights only (no pretraining for the transformer component itself). The contrastive alignment loss variant helps but is not in the official released model.

### 5. Fixed Number of Query Slots
`num_queries=10` is a hard ceiling. A video with 11 relevant moments will have one missed. In practice QVHighlights has 1-2 relevant windows per query, so this rarely matters.

---

## What the TEF Actually Does

The temporal endpoint feature `[t_start/L, t_end/L]` appended to each clip's visual features gives the transformer explicit knowledge of WHERE in the video each clip sits. Without TEF, the model must infer position entirely from positional encoding — which works but slows learning. TEF provides a strong inductive bias: clips near the center of the video have TEF ≈ [0.5, 0.5], clips at the start have TEF ≈ [0.0, 0.013].

This is important for your project: when you use the trained model for inference on YOUR downloaded clips, you must reconstruct TEF from the clip position within the 150s segment.
