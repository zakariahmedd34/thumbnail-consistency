# Moment-DETR Paper Notes — Engineering-Level Analysis

**Paper:** "QVHighlights: Detecting Moments and Highlights in Videos with Natural Language Queries"  
**Authors:** Jie Lei, Tamara L. Berg, Mohit Bansal (UNC Chapel Hill)  
**Venue:** NeurIPS 2021  
**Source read:** `2107.09609v2.pdf`, `model.py`, `start_end_dataset.py`, `matcher.py`, `opt.json`

---

## What the Paper Actually Claims

The paper makes THREE simultaneous contributions:
1. **QVHighlights dataset** — a new benchmark combining moment retrieval + highlight detection
2. **Moment-DETR model** — a transformer-based temporal detection model
3. **Evaluation protocol** — combined MR + HL metrics

Most prior work (2D-TAN, VSLNet, etc.) handled either moment retrieval OR highlight detection. QVHighlights defines a joint task where both are evaluated simultaneously on the same model.

---

## Key Design Decisions and Their Motivations

### Why DETR?
DETR eliminated anchor boxes and NMS from object detection by reformulating detection as set prediction with Hungarian matching. The same logic applies here: temporal moments have no canonical size/position, so anchor-free set prediction is natural. The Hungarian matcher learns which query slot "owns" which ground-truth window.

**Practical implication:** With `num_queries=10`, the model can predict up to 10 moments per query. QVHighlights has 1–5 relevant windows per query in practice, so the model has surplus capacity.

### Why Concatenate Video and Text in the Encoder?
```python
src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
```
This is "early fusion" — the encoder's self-attention allows video clips to directly attend to text tokens and vice versa. This is more powerful than processing them separately but requires the model to learn which text tokens are discriminative for which video positions.

Compare to late fusion (process separately, merge at output) — early fusion has higher expressive capacity but is harder to train on limited data.

### Why L1 + GIoU for Span Regression?
L1 loss alone doesn't penalize non-overlapping predictions the same way as overlapping ones — two predictions 10 seconds apart or 50 seconds apart both have L1 loss proportional to their absolute difference. GIoU (generalized IoU) for temporal spans adds an overlap term that changes the gradient landscape: predictions that completely miss the target get a stronger gradient to move toward it.

### Why Not Use the Full SlowFast for the CLIP-only Model?
The official released checkpoint uses only CLIP visual features (512-dim), not SlowFast. Looking at `opt.json`:
```json
"v_feat_dirs": ["features/clip_features"],
"v_feat_dim": 512
```
This suggests that for the benchmark evaluation they found CLIP-only competitive (while being 5x smaller). SlowFast captures motion information but the temporal reasoning in the transformer partially compensates.

---

## The Saliency Task — Critical Details Often Missed

The saliency task in QVHighlights is **not** the same as general video saliency. It is:
- **Query-conditional** — what is salient GIVEN this specific query
- **Relative** — clips are ranked within a video, not scored absolutely
- **Ordinal** — annotators give 0/1/2/3/4 scores, but the supervision uses pairwise ranking

The saliency margin loss is:
```
max(0, margin + S(neg_clip) - S(pos_clip))
```
This is a hinge loss. It does NOT regress to specific score values. It only ensures positives score higher than negatives by at least `margin=0.2`.

**Implication for your pipeline:** The saliency scores M1 outputs are NOT calibrated probabilities. They are relative rankings within a video. You cannot compare saliency scores across videos. This matters for your consistency score — you need to normalize per video.

---

## The Weakly-Supervised ASR Pretraining

From `start_end_dataset.py`:
```python
if "subs_train" not in self.data_path:
    # use annotated saliency labels
else:
    # use subtitle windows as pseudo-ground-truth
    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)
```

The `subs_train.jsonl` file contains (query=ASR_subtitle, video, time_window) triplets. This is used as weakly-supervised pretraining before fine-tuning on the human-annotated data. The paper shows this improves performance significantly.

**For your project:** You have `subs_train.jsonl`. Using it for pretraining before fine-tuning on your 2000-sample subset should help, especially for saliency. The pretraining step runs with `data/subs_train.jsonl` as the train_path (it already handles the `subs_train` branch in the code).

---

## What the Paper Does NOT Tell You

### 1. Feature Extraction Details
The paper says "SlowFast + CLIP features" but doesn't specify:
- Exactly which SlowFast variant (R50 or R101)
- Frame sampling rate for SlowFast
- Whether CLIP features are frame-level or clip-level averaged
- Whether features are computed before or after temporal windowing

You need to inspect the actual feature arrays to understand their structure.

### 2. NMS and Post-Processing
The paper reports results with NMS disabled (`nms_thd=-1`), but the code supports NMS. The number of predictions after NMS affects recall — with NMS disabled and `num_queries=10`, you always get 10 predictions.

### 3. Training Instability
The code has `grad_clip=0.1` which is aggressive. This suggests the training can be unstable, particularly early on. The paper doesn't discuss training stability.

### 4. The `eos_coef=0.1` Asymmetry
Background class gets 10x lower weight. This means the model is biased toward predicting MORE foreground (lower false negative rate at the cost of more false positives). For your use case (you want to find the most salient moment), this is the right tradeoff.

---

## Important Equations

### Temporal Span Normalization
All spans are normalized by `max_v_l * clip_len = 75 * 2 = 150` seconds:
```
center_norm = (start + end) / 2 / 150
width_norm  = (end - start) / 150
```

### GIoU for 1D Temporal Spans
```
IoU(A, B) = |A ∩ B| / |A ∪ B|
GIoU(A, B) = IoU(A, B) - |C \ (A ∪ B)| / |C|
```
where C is the smallest span containing both A and B.

### Saliency Ranking Loss
```
loss = Σ max(0, margin + score(neg[i]) - score(pos[i])) / (N * num_pairs)
```
The `* 2` in the code is to maintain the same scale when `num_pairs` doubles (hard+easy).

---

## Connection to Your Project

The paper's joint task (MR + HL) maps directly to your pipeline:
- **MR output** = specific moment window retrieved by your M2-generated query
- **HL output** = per-clip saliency scores for the whole video

Your consistency score combines these: the thumbnail should represent a moment that M1 BOTH localizes (MR) AND scores as salient (HL). A thumbnail that scores low on both metrics is likely clickbait.
