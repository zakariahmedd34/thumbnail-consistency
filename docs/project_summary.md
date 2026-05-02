# Video Thumbnail Selection — Project Design Summary

> One month of thinking, distilled into one document.

---

## What This Project Is

A deep learning research project for **automatic video thumbnail selection**, evolving from a naive pipeline into a principled two-stage self-supervised framework. The core contribution is solving the problem that most papers ignore: **creator thumbnails are not reliable ground truth**, so training directly on them teaches the wrong thing.

---

## The Journey — How the Problem Evolved

### Stage 1 — Naive starting point
The original idea:
1. Extract frames from a video at 1 fps
2. Compute `CLIP(frame, thumbnail)` similarity
3. Use similarity scores as weak labels
4. Train a model to rank frames

**Why this failed:** The model would simply learn to approximate CLIP. This is circular — the training signal comes from the same model being used to evaluate. Also, thumbnails are often edited, zoomed, composited, or clickbait, so `CLIP(frame, thumbnail)` is a noisy and misleading label.

---

### Stage 2 — Multimodal transformer ranker
Upgraded to a proper trainable architecture:

| Component | Choice |
|---|---|
| Vision encoder | ViT-B/16 |
| Text encoder | BERT-base |
| Fusion | Cross-attention transformer (frame queries, title keys/values) |
| Ranking head | MLP → score ∈ [0,1] per frame |
| Loss | ListMLE + BCE + CLIP alignment |
| Labels | CLIP(frame, thumbnail) — still weak |

Added:
- **Temporal transformer** over frame sequence with sinusoidal positional encoding
- **Quality features** per frame: face count, face size, face centeredness, aesthetic score (CLIP-IQA), sharpness — concatenated with visual embeddings before projection
- **Tensor shapes** fully specified through the pipeline

**Why ranking was the wrong framing:** Most papers do not solve ranking — they solve retrieval. The ranking formulation adds complexity without payoff. Most papers quietly treat the creator thumbnail as ground truth and find the nearest frame. The real contribution should be elsewhere.

---

### Stage 3 — Recognizing the ground truth problem
Two fundamental problems identified:

**Problem 1 — Thumbnail type ambiguity:**
- Type A thumbnails are representative (cooking channel shows the finished dish)
- Type B thumbnails are clickbait (shocked face engineered for clicks)
- YouTube is dominated by Type B
- Training on both teaches the model to predict clickbait, not video content

**Problem 2 — The evaluation was unclear:**
- Ranking metrics are ambiguous
- CLIP similarity as label creates circularity
- The paper by Yu et al. (2101.00073) solved this cleanly: predict a target embedding → retrieve nearest frame by MSE → evaluate with Precision@θ

---

### Stage 4 — Title-transcript pretraining idea
New idea: use Whisper to get the transcript, then train a contrastive model on `(title, transcript)` pairs as self-supervised pretraining.

**Why this was partially right but incomplete:**
- It learns TEXT ↔ TEXT alignment, not IMAGE ↔ TEXT
- The thumbnail problem requires frame-level visual scoring
- The trained encoder has never seen an image
- Still does not resolve the noisy label problem for stage 2

---

### Stage 5 — The circular assumption caught
The contrastive pretraining on `(thumbnail_i, transcript_i)` pairs was proposed — then correctly identified as circular: using same-video pairs as positives assumes the thumbnail is consistent with the transcript, which is exactly the assumption being questioned.

**This was the key intellectual breakthrough of the project.**

---

### Stage 6 — The clean solution

**The insight:** Frames and transcripts are guaranteed to be consistent with each other because they are temporally co-produced during filming. Neither is chosen after the fact. The thumbnail is the only thing chosen after filming — and therefore the only thing that can be misleading.

**Training:** Never use the thumbnail. Train a contrastive model on `(frame_t, transcript_segment_t)` pairs aligned by timestamp. Positive pairs are guaranteed by physical co-occurrence, not human assumption.

**Inference:** Score the thumbnail as an outsider against the learned video context:
```
consistency_score = cosine_sim(
    mean(embed(frames) + embed(transcript)),
    embed(thumbnail)
)
```

**Why this is valid:**
- The pretext task (frame-transcript alignment) is causally related to the downstream task (thumbnail-video consistency)
- Noise (transcription errors, blurry frames) is random and uncorrelated with the task — the model averages over it
- This is the same theoretical foundation as CLIP, wav2vec, and SimCLR

---

## The Final Two-Stage Pipeline

### Stage 1 — Thumbnail-Video Consistency Scoring
**Goal:** Learn what the video is about without ever touching the thumbnail.

| | Detail |
|---|---|
| Input | Video frames + Whisper transcript |
| Positive pairs | (frame_t, transcript_segment_t) — temporally aligned |
| Negative pairs | (frame_t, transcript_j) — from different video |
| Loss | InfoNCE contrastive loss |
| Output | Trained consistency scorer |
| At inference | cosine_sim(video_context, thumbnail) → P ∈ [0,1] |

**Use:** Filter the dataset. Keep videos where consistency score ≥ τ (threshold set at the valley of the bimodal score distribution). Discard clickbait/misleading videos.

---

### Stage 2 — Frame Retrieval
**Goal:** Given a video with a verified-consistent thumbnail, find which frame is most similar to it.

| | Detail |
|---|---|
| Input | Clean subset from Stage 1 only |
| Architecture | ViT frames + BERT title → temporal transformer → fusion → FC layers → embedding ô |
| Loss | MSE(ô, embed(thumbnail)) |
| Retrieval | argmin MSE(ô, frame_i) over all candidate frames |
| Evaluation | Precision@θ — same metric as Yu et al. paper |

---

## Key Architecture Details

### Per-frame feature vector (Stage 2)
```
ViT-B/16 output:     v ∈ ℝ⁷⁶⁸
Quality features:    q ∈ ℝ⁵  [face_count, face_size, centerness, aesthetic, sharpness]
Concatenation:       [v ‖ q] ∈ ℝ⁷⁷³
Projection:          Linear(773, 512) + LayerNorm → frame token f_i ∈ ℝ⁵¹²
```

### Temporal transformer
```
Input:  [f₁, f₂, … fₙ] + sinusoidal positional encoding (by timestamp, not index)
Layers: 2–4 self-attention layers
Output: [h₁, h₂, … hₙ] — temporally contextualised frame tokens
```

### Tensor shapes through Stage 2
```
frames:      [B, N, 3, 224, 224]
title:       [B, T]                   T = token length
quality:     [B, N, 5]                precomputed, loaded from disk
after ViT:   [B, N, 768]
after proj:  [B, N, 512]              d_model = 512
after temp:  [B, N, 512]
after fuse:  [B, N, 512]
scores:      [B, N]                   one per frame
```

### Loss function (Stage 2)
```
L_total = λ₁·L_rank + λ₂·L_bce + λ₃·L_align
λ₁ = 1.0  (ListMLE over CLIP-derived frame rankings)
λ₂ = 0.5  (BCE: top-K frames = 1, rest = 0)
λ₃ = 0.3  (contrastive: fused repr → CLIP thumbnail embedding)
```

---

## Labeling Strategy

| Strategy | Method | Risk |
|---|---|---|
| Fixed threshold τ = 0.7 | frames above threshold = positive | class imbalance across video types |
| Per-video percentile (top 10%) | always label top 10% per video as positive | consistent ratio, handles edge cases |
| Soft labels | min-max normalize CLIP scores per video → BCE target | eliminates τ entirely, preserves ordinal info |

**Recommended:** Start with percentile, switch to soft labels once training is stable.

---

## Papers Referenced

| Paper | Key contribution to your thinking |
|---|---|
| Fantini (EPFL thesis, 2410.19825) | Multistage pipeline: Hecate downsampling, CLIP-IQA aesthetics, multi-criteria scoring, variety enhancement, face identity, logo score |
| Yu et al. (2101.00073) | Clean evaluation metric (Precision@θ), embedding prediction + MSE retrieval, context gating, multimodal fusion with audio |

### What Fantini's paper added to your design
- Replace single importance score with **weighted multi-criteria scoring**
- **LLM keyword extraction → CLIP matching** for richer semantic signal
- **Eyes-closed filter** using Eye Aspect Ratio (EAR) from face landmarks
- **Variety enhancement**: best frame per character, per emotion, per keyword
- **Image harmonization** after segmentation + background generation
- Drop the BiLSTM — temporal modelling is unnecessary for this task

### What Yu et al.'s paper added
- Predict a target embedding → retrieve nearest frame (not ranking)
- Precision@θ as clean, interpretable evaluation
- Context gating layer for across-modal interaction
- Showed audio is a useful modality

---

## The Contribution of This Project

1. **Identifies the core flaw** in all existing thumbnail selection papers: they assume creator thumbnails are ground truth without questioning it
2. **Proposes a self-supervised consistency scorer** trained on frame-transcript alignment — no human annotation, no circular assumptions
3. **Uses the scorer as a data filter** to produce a clean training set for frame retrieval
4. **Shows that training Stage 2 on filtered data** outperforms training on noisy unfiltered data (the testable main claim)
5. **Evaluation is clean**: Precision@θ on a held-out set of consistent videos, directly comparable to Yu et al.

---

## Recommended Project Title

> **Annotation-Free Video Thumbnail Consistency Scoring via Frame-Transcript Contrastive Learning**

Alternative titles:
- *A Two-Stage Framework for Thumbnail Relevance Filtering and Representative Frame Retrieval in Online Videos*
- *Noise-Aware Multimodal Video Thumbnail Selection with Self-Supervised Consistency Filtering*
- *Self-Supervised Thumbnail-Video Consistency Detection for Noise-Robust Frame Selection*

---

## What to Do When You Return

1. Implement Stage 1 first — it is self-contained and its output (the consistency scores) is what makes Stage 2 meaningful
2. Run Stage 1 on your full dataset and plot the score distribution — verify it is bimodal before committing to the filter threshold
3. Manually inspect ~50 videos at the boundary (scores near τ) to sanity-check the scorer
4. Train Stage 2 on the clean subset only
5. Compare Precision@θ against Yu et al. as your baseline — you have their exact metric and can use their model as comparison

---

*Summary generated after one month of iterative design discussion. The problem is well-formed. The methodology is sound. Come back to it.*
