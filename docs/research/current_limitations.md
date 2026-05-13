# Current Limitations — Known Issues and Research Boundaries

**Last updated:** 2026-05-07

---

## L1 — Thumbnail Domain Gap [HIGH SEVERITY]

**Problem:** M2 is trained on natural video frames extracted from QVHighlights clips. At inference, it receives stylized YouTube thumbnails.

**Root cause:** No thumbnail ground truth exists in QVHighlights. We constructed training data by extracting frames from annotated high-saliency clips. Real YouTube thumbnails are edited images, not raw frames.

**Gap characteristics:**
- Thumbnails have higher saturation, contrast, and sharpness than raw frames
- Thumbnails often contain text overlays (channel name, title words)
- Thumbnails may be composites of multiple frames or entirely generated
- Thumbnails tend to show close-up faces for engagement — raw frames have full-scene composition

**Impact on M1:** None directly — M1 works on video features, not thumbnails.  
**Impact on M2:** The CLIP encoder sees an out-of-distribution input at inference. The distribution of visual embeddings shifts. The decoder was trained on one embedding distribution and receives another.

**Impact on pipeline:** The generated query from a thumbnail may describe the CLIP-latent-space nearest neighbor in training data, not the actual content of the thumbnail.

**Mitigation (current):** Using CLIP ViT-B/32 which is highly robust to visual style variation due to pre-training on diverse web images.

**Mitigation (future):** Fine-tune CLIP on (thumbnail, frame) pairs from the same video. Requires collecting thumbnail images from YouTube — possible with yt-dlp's thumbnail download option.

**Scientific validity:** The domain gap does NOT invalidate the consistency pipeline. It is a known limitation that reduces M2 quality, and quantifying it is a research contribution. The pipeline works despite the gap — and the gap itself is discussable.

---

## L2 — Single Frame Cannot Represent Temporal Events [MEDIUM SEVERITY]

**Problem:** Many QVHighlights queries describe temporal actions ("woman starts laughing", "man falls from bike"). A single frame cannot capture these.

**Root cause:** Annotation is at clip level; we select a single frame to represent the clip.

**Impact on M2:** The model learns visual→text mappings that associate visual snapshots with temporal descriptions. It learns associations like "blurry motion frame + person" → "person running" which are statistically plausible but not always correct.

**Impact on consistency pipeline:** If the thumbnail shows a dynamic action (which thumbnails often do — they pick dramatic moments), M2 may fail to describe the action correctly.

**Not solvable within current project scope.** Acknowledgeable as a limitation. The CLIP-ranked frame selection mitigates this partially by selecting frames that have the highest semantic match to the query text.

---

## L3 — M2 Vocabulary Bottleneck [HIGH SEVERITY]

**Problem:** Custom vocabulary built from 2000 training queries. ~6000 tokens. All out-of-vocabulary words → `<UNK>`.

**Impact:** Degrades generation quality for any thumbnail content not represented in training vocabulary. Most proper nouns, specialized activities, and domain-specific terms will be unknown.

**Status:** Fix identified (see m2_thumbnail_query_generation.md). Requires switching to pre-trained tokenizer or fine-tuning GPT-2.

**Priority:** High. This is the single most tractable improvement to M2 quality.

---

## L4 — Short Moment Localization Failure in M1 [MEDIUM SEVERITY]

**Problem:** Official M1 achieves mAP of only 3.11% for short moments (< ~6 seconds). The 2-second clip granularity creates a localization floor.

**Impact on consistency pipeline:** If the thumbnail depicts a very short event (a reaction, a jump cut), M1 may fail to retrieve the correct moment even when given the correct query.

**Root cause:** The span regression head must predict very small widths for short moments. The Hungarian matcher penalizes span cost heavily (weight=10) but the regression is ill-conditioned for widths near zero.

**Mitigation:** For the consistency pipeline, use the SALIENCY OUTPUT of M1 rather than the span predictions for short-moment videos. The saliency per clip is better calibrated than the span regression for short moments.

---

## L5 — Consistency Score Not Yet Defined [CRITICAL]

**Problem:** The consistency score — the central claim of the pipeline — has not been formally defined, implemented, or evaluated.

**Impact:** The research contribution is unvalidated. All other work is preprocessing toward an undefined goal.

**Priority:** This must be resolved before any paper writing or evaluation.

**Candidate formulations are documented in `project_overview.md` (Options A–D).**

---

## L6 — No Clickbait Ground Truth [MEDIUM SEVERITY]

**Problem:** QVHighlights does not contain annotations for "is this thumbnail misleading?" The dataset is about moment retrieval, not consistency.

**Impact:** Evaluation of the full pipeline requires either manual labelling or a proxy metric.

**Mitigation:**
1. Use `best_timestamp ∈ relevant_windows` as a proxy for consistency (see `ablation_ideas.md`)
2. Manually annotate 50–100 examples for a small gold evaluation set
3. Construct synthetic clickbait pairs

**Scientific note:** The lack of ground truth is a limitation but not a fatal one for a course project. "Proof of concept" with a small labelled set is publishable.

---

## L7 — CLIP Score Race Condition (Known Bug, Partially Fixed)

**Problem:** `_get_clip()` is not thread-safe. Multiple ThreadPoolExecutor workers can attempt simultaneous model initialization, causing exceptions that result in `clip_score=0.0` fallbacks.

**Impact:** 39 pairs in the current dataset have `clip_score=0.0` despite being valid frames from high-saliency clips.

**Fix ready:** Pre-load CLIP before executor, use `None` instead of `0.0` for fallbacks.

**Status:** Fix documented in previous session, not yet applied to notebook.

---

## Limitations Worth Discussing in a Paper

In order of research interest:

1. **Domain gap (L1)** — well-motivated, quantifiable, connects to broader generalization literature
2. **Single-frame temporal limitation (L2)** — fundamental constraint of static thumbnail representation
3. **Vocabulary coverage (L3)** — engineering limitation but worth reporting with numbers
4. **Short moment failure (L4)** — important for real-world application where thumbnails often show brief dramatic moments
5. **Annotation noise in saliency labels** — annotators disagreed significantly on some clips; the saliency supervision is noisy
