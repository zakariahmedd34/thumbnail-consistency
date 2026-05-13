# Project Overview — Thumbnail Consistency / Clickbait Detection

**Last updated:** 2026-05-07  
**Status:** Data pipeline complete. M1 feature inspection pending. M2 v1–v3 trained on subset.

---

## Core Research Question

> Given a YouTube thumbnail, does the video actually contain the visual event the thumbnail implies?

This is a **thumbnail-to-moment consistency** problem. The pipeline detects clickbait by measuring the semantic distance between what a thumbnail promises and what a video delivers.

---

## Pipeline Architecture

```
Thumbnail (image)
      │
      ▼
   [M2]  CLIP encoder + Transformer decoder
      │
      ▼
 Generated Query  (natural language description)
      │
      ▼
   [M1]  Moment-DETR (frozen or fine-tuned)
      │
      ▼
 Retrieved Moments + Saliency Scores
      │
      ▼
 Consistency Score  ←── compare with thumbnail embedding
      │
      ▼
 Clickbait / Consistent label
```

---

## What Is Novel Here

Reproducing Moment-DETR is **not** the research contribution. The field already has moment retrieval. What is novel:

1. **Using generated queries as moment retrieval probes** — using a vision→language model to produce the query that M1 receives. Nobody checks whether the thumbnail implies the same moment M1 thinks is salient.

2. **Thumbnail consistency scoring** — a soft, embedding-level measure of whether the thumbnail semantically agrees with the video's most salient region.

3. **Domain shift analysis** — studying the gap between stylized thumbnails (M2 inference input) and natural video frames (M2 training data), and quantifying how much it hurts.

The consistency pipeline is the research claim. M1 and M2 individually are baselines that enable it.

---

## Current State of Each Component

| Component | Status | Notes |
|---|---|---|
| Data preprocessing | ✅ Complete | 2000 train, 1608 M2 pairs, CLIP-ranked frames |
| M1 baseline | ⏳ Pending | Features available, not yet trained on subset |
| M2 v1 (LSTM/GRU) | ✅ Trained | Frozen CLIP + custom vocab decoder |
| M2 v2 (Improved) | ✅ Trained | Improved version of v1 |
| M2 v3 (Transformer) | ✅ Trained | CLIP visual tokens → Transformer decoder |
| Consistency scorer | ❌ Not built | Most important unresolved component |
| Domain shift study | ❌ Not done | Critical for research narrative |

---

## Hardest Unresolved Design Question

**How is the consistency score defined?**

This is the most important question in the project. Candidate formulations:

### Option A — CLIP Embedding Cosine Similarity
```
score = cos_sim(CLIP(thumbnail), mean(CLIP(retrieved_frames)))
```
Simple, zero extra training. Measures visual similarity only, misses semantic intent.

### Option B — Query-Moment Text-Video Similarity
```
score = cos_sim(CLIP_text(generated_query), CLIP_video(retrieved_moment_features))
```
Uses the M1 pipeline's CLIP features directly. Measures semantic grounding.

### Option C — Saliency-Weighted IoU
```
score = Σ saliency_score[t] * (thumbnail_temporal_window ∩ M1_window[t])
```
Requires mapping the thumbnail to a temporal window — circular / ill-defined without ground truth.

### Option D — Ranking Consistency
```
score = rank of the moment most similar to thumbnail, out of all M1 predictions
```
How high does M1 rank the moment that visually resembles the thumbnail? Lower rank = more consistent.

**Recommendation:** Start with Option B. It is grounded in existing features, requires no new training, and is interpretable: if the generated query retrieves moments that are semantically close to the thumbnail, the video is consistent.

---

## Research Narrative (Draft)

> "Current approaches to video moment retrieval assume queries are human-authored descriptions of genuine events. We ask a different question: what moment does the thumbnail claim exists? We train a thumbnail→query generator (M2) on human-annotated saliency data, use it to produce semantic queries from thumbnails, and feed these queries into a Moment-DETR-based retrieval model (M1). The consistency between the thumbnail embedding and the retrieved moment embedding defines a clickbait score. We evaluate this pipeline on QVHighlights and demonstrate that low-consistency scores correlate with misleading thumbnails."

---

## Compute Budget Guidance (Colab T4)

| Task | Estimated Time |
|---|---|
| Inspect/unpack official features | 15 min |
| Run M1 baseline (official features, full train) | 15–20 hrs |
| Run M1 baseline (2000 subset) | 4–6 hrs |
| Re-extract M2 frames (1500 videos, CLIP) | 30 min |
| Train M2 v3 (Transformer, 1608 pairs) | 2–4 hrs |
| Build consistency scorer + evaluate | 2–3 hrs |

---

## Immediate Next Steps (Priority Order)

1. Apply the 3 `clip_score` bug fixes to `data_collection_preparation.ipynb`
2. Inspect `moment_detr_features.tar.gz` — understand exact feature format before training
3. Run M1 on official features at full scale → get baseline numbers
4. Run M1 on 2000-sample subset → understand performance degradation
5. Define consistency score formula (recommend Option B)
6. Build consistency evaluation on a small manually-labelled holdout
