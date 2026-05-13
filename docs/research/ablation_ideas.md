# Ablation Study Plan

**Last updated:** 2026-05-07  
**Priority:** Execute in order — earlier ablations inform later ones.

---

## Ablation 1 — M1 Feature Configuration (Highest Priority)

**Hypothesis:** CLIP-only features are sufficient for this project; adding SlowFast does not justify compute cost.

| Config | v_feat_dim | Expected mAP | Compute cost |
|---|---|---|---|
| CLIP-only + TEF | 514 | ~53% (official) | Low |
| SlowFast + CLIP + TEF | 2818 | ~55–57% | 2x |
| TEF only | 2 | ~30% | Very low |

**How to run:** Change `--v_feat_dirs` and `--v_feat_dim` in the train command. Training cost is the same — only the input projection changes.

**Expected outcome:** CLIP-only is sufficient for the consistency pipeline. SlowFast adds motion representation but at 5x the feature cost.

---

## Ablation 2 — M2 Frame Selection Method

**Hypothesis:** CLIP-ranked frame selection produces better M2 training data than center-frame selection.

| Frame selection | Method | Expected CLIP text sim. | Expected retrieval quality |
|---|---|---|---|
| Center frame | `best_timestamp = best_clip_id * 2 + 1` | Baseline | Baseline |
| CLIP-ranked (current) | `argmax(CLIP_sim(frames, query))` | +X | +Y |
| Random from clip | `random.choice(clip_frames)` | Baseline - noise | Below baseline |

**How to measure:** 
1. Re-run M2 training with center-frame pairs (save original `best_timestamp` before CLIP reranking)
2. Compare: CLIP text similarity of generated queries vs ground truth queries
3. Compare: R1@0.5 when using generated query in M1

**Why this matters:** Validates the main engineering contribution of your preprocessing pipeline.

---

## Ablation 3 — M2 Architecture (LSTM vs GRU vs Transformer)

Already partially implemented via v1 (LSTM/GRU) vs v3 (Transformer).

| Architecture | Parameters | BLEU-1 | CLIP text sim | Training time |
|---|---|---|---|---|
| GRU (v1) | ~2M | Measured | Measure | ~30 min |
| LSTM (v1) | ~2M | Measured | Measure | ~30 min |
| Transformer (v3) | ~4M | Measured | Measure | ~1 hr |
| GPT-2 fine-tuned | ~117M | Expected best | Expected best | ~3 hrs |

**Key finding to report:** Whether the architectural improvement (Transformer > LSTM/GRU) is meaningful given the tiny dataset size. With 1608 training samples, a more powerful architecture may overfit more.

---

## Ablation 4 — CLIP Score Threshold Effect on M2 Quality

**Hypothesis:** Filtering training pairs by clip_score > 0.22 improves M2 generation quality despite reducing dataset size.

| Threshold | Training pairs | Expected BLEU | Expected CLIP text sim |
|---|---|---|---|
| No filter | 1608 | Baseline | Baseline |
| > 0.22 | ~1569 | Slightly higher | Slightly higher |
| > 0.25 | ~1200 | ?? | ?? |
| > 0.28 | ~800 | ?? | ?? |

**Note:** The fallback pairs (clip_score=None) should be included with threshold=0.22 since they are center frames from high-saliency clips (good quality, just unscored).

---

## Ablation 5 — Query Quality Impact on Consistency Scoring

**This is the most important ablation for the research narrative.**

**Hypothesis:** The consistency score is more reliable when computed with higher-quality queries.

| Query source | Expected consistency score distribution | AUC on clickbait test set |
|---|---|---|
| Ground truth query | Tightest distribution, highest AUC | Ceiling performance |
| M2 Transformer query | Moderate | Target for project |
| M2 LSTM/GRU query | Wider distribution | Baseline |
| Random query (control) | Flat distribution | ~50% (random) |

**How to run:** For each query source, compute the consistency score on the same set of videos. Compare how well each separates consistent vs inconsistent thumbnails.

This ablation directly shows: "how much does M2 query quality matter for the downstream clickbait detection task?"

---

## Ablation 6 — Consistency Score Formula

**Hypothesis:** CLIP-based semantic similarity outperforms IoU-based consistency.

| Score formula | Description | Expected quality |
|---|---|---|
| `cos_sim(CLIP_text(M2_query), CLIP_vid(M1_moment))` | Text-video CLIP | Best semantic alignment |
| `cos_sim(CLIP_img(thumbnail), CLIP_vid(M1_moment))` | Visual CLIP | Good but misses text |
| `max_IoU(M1_windows, thumbnail_time_window)` | Temporal overlap | Requires thumbnail time mapping |
| `saliency_score_at_thumbnail_time` | Saliency only | Ignores thumbnail semantics |

**Recommendation:** Implement Options A and B, report both. The difference reveals how much the query generation step adds over direct visual comparison.

---

## Evaluation Infrastructure Needed

For ablations 5 and 6, you need a **small labelled evaluation set** of (thumbnail, video, label) triples where label ∈ {consistent, clickbait}. Options:

1. **Manual labelling:** Annotate 50–100 thumbnails from your downloaded QVHighlights videos as consistent or not. Takes ~2 hours.

2. **Proxy labelling:** Use the original `relevant_windows` annotation. If the thumbnail timestamp (best_timestamp) falls within a relevant window, label it consistent. Otherwise, label it inconsistent. This is imperfect but immediate.

3. **Synthetic clickbait:** Take a video's frame and pair it with a DIFFERENT video's query. This gives a hard negative — the frame is real but the query is mismatched.

Option 2 is the fastest path to a working evaluation.
