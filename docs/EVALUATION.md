# Evaluation Process — End-to-End

> How the data prepared in `data_collection_preparation.ipynb` flows into the
> evaluation notebooks, what metrics we compute, and how to interpret them.

---

## 0. TL;DR

```
data_collection_preparation.ipynb   ─►   processed/*.jsonl + features
                                                │
                                                ▼
                  run_pipeline_allinone.ipynb (or sanity_consistency.ipynb)
                                                │
                                                ▼
                       AUC, Cohen's d, score histogram on 600 pairs
```

The whole evaluation is **600 pairs** in one file:
`processed/consistency_eval.jsonl` (300 positives + 300 hard negatives).

---

## 1. The pipeline at a glance

```
   Thumbnail T (frame from QVHighlights video)
      │
      ├──► CLIP-img (frozen) ──────────────────────────► e_T
      │
      └──► GT query (from QVHighlights annotation)
                                │
                                ▼
                       CLIP-text (frozen) ──► e_q
                                                │
                                                ▼
   Video V's pre-extracted CLIP features (75 clips × 512-d)
                                │
                                ▼
              Moment-DETR (official ckpt) ──► saliency s[t]
                                │
                                ▼
              saliency-softmax pool ──► e_clip
                                │
                                ▼
   C(T, V) = 0.5 · cos(e_q, e_clip) + 0.5 · cos(e_T, e_clip)
                                │
                                ▼
                  AUC on 600 (label, C) pairs
```

**For V1 (smoke test)**: queries are **ground-truth** QVH annotations.
**For V2 (full eval)**: queries are **M2-generated** (ClipCap output).

---

## 2. Data prepared by `data_collection_preparation.ipynb`

Each cell range in the notebook produces one downstream artifact. The table maps
**notebook section → output file → who consumes it**.

| Notebook section | Output file | Consumer (eval notebook) |
|---|---|---|
| §5 Build Processed Dataset | `train_full_processed.jsonl` | (internal — used by §6) |
| §6 Subset Selection | `train_processed.jsonl` (2000) | M1 audit |
| §8 Download Video Clips | `qvh_videos/*.mp4` | §9 frame extraction |
| §9 Extract Best Frame | `qvh_frames/{vid}.jpg` | M2 + evaluation thumbnails |
| §10 Build M2 Pairs | **`m2_train_pairs.jsonl`** | `train_clipcap.ipynb` |
| §13 Download Thumbnails | `thumbnail_index.jsonl` + `qvh_thumbnails/{yt_id}.jpg` | Domain-gap experiment |
| §14 M1 Subset (official format) | `m1_train_subset.jsonl`, `m1_val.jsonl` | M1 audit (not retraining) |
| §15 M2 Val Pairs | **`m2_val_pairs.jsonl`** (300) | `train_clipcap.ipynb` early stop |
| §16 **Consistency Eval Set** | **`consistency_eval.jsonl`** (600) | **All eval notebooks** ★ |
| §17 Cache CLIP Embeddings | `m2_image_emb.npz`, `thumbnail_image_emb.npz` | Optional speedup |
| §2.5 Unpack Features (v8) | `Moment_DETR/moment_detr/features/{clip,clip_text}_features/` | M1 inference ★ |

★ = required for the R1 smoke test.

### What each row in `consistency_eval.jsonl` looks like

```json
{
  "pair_id": "pos_707",        // or "neg_707_8520"
  "label": 1,                  // 1 = consistent, 0 = mismatched
  "thumb_qid": 707,
  "thumb_vid": "Exr8FNpynKo_210.0_360.0",
  "frame_path": "/.../qvh_frames/Exr8FNpynKo_210.0_360.0.jpg",
  "thumb_query": "Man in red hoodie stretches on the street.",
  "video_qid": 707,            // same as thumb_qid for positives, different for negatives
  "video_vid": "Exr8FNpynKo_210.0_360.0",
  "video_query_for_ref": "Man in red hoodie stretches on the street."
}
```

### How positives and negatives are constructed

```
Positive pair: thumbnail's frame + that thumbnail's OWN video
               (frame_i,  vid_i)   ← they came from the same QVH record

Negative pair: thumbnail's frame + a DIFFERENT video whose query is dissimilar
               (frame_i,  vid_j)   ← i ≠ j
               with cos(CLIP_text(q_i), CLIP_text(q_j)) < 0.6

The < 0.6 filter prevents "accidental positives" — random pairings that happen
to be semantically related ("two different vlogs of cooking" — both labelled
neg but actually consistent).
```

---

## 3. What the evaluation notebook does (run by run)

Open `notebooks/run_pipeline_allinone.ipynb` on Colab. The cells, top to bottom:

| Cell | Purpose |
|---|---|
| 1 | Mount Drive |
| 2 | Install transformers / torch / tqdm |
| 3 | Clone upstream Moment-DETR (for the model code only — `build_model`, span utils) |
| 4 | Load M1 official checkpoint into a frozen `m1` |
| 5 | Load CLIP-img + CLIP-text (frozen). Defensive wrappers for HF version quirks |
| 6 | Define `m1_predict_saliency(v_feat, query)` |
| 7 | Define inline scoring math — `cosine_sim`, `saliency_weighted_pool`, `consistency_score` |
| 8 | Define inline metrics — `roc_auc`, `cohens_d` |
| 9 | Load `consistency_eval.jsonl` |
| 10 | **Main loop**: for each of 600 pairs → embed, get saliency, pool, score |
| 11 | Print AUC / Cohen's d / separation, verdict (PASS/FAIL) |
| 12 | Save histogram to Drive |

### The per-pair scoring step (pseudocode)

```python
for pair in 600_pairs:
    e_T      = CLIP_img(pair.frame_path)              # (512,)
    e_q      = CLIP_text(pair.thumb_query)            # (512,)
    v_feat   = load_npz(features/{pair.video_vid}.npz) # (n_clips, 512)
    saliency = M1(v_feat, pair.thumb_query)           # (n_clips,)
    e_clip   = saliency_weighted_pool(v_feat, saliency,
                                       tau=0.1, top_p=0.25)  # (512,)
    C        = 0.5 * cos(e_q, e_clip) + 0.5 * cos(e_T, e_clip)
    results.append({label: pair.label, score: C})
```

---

## 4. The score formula — what each term does

| Term | What it captures | Why it's there |
|---|---|---|
| `cos(e_q, e_clip)` | Semantic match between the query string and the salient video region | Tests whether M1's saliency aligns with the linguistic content of the query |
| `cos(e_T, e_clip)` | Visual match between the thumbnail image and the salient video region | Bypasses the query path — catches cases where M2 generates a bad query but the visuals still match |
| `α = 0.5` (V1 default) | Balanced weight | Lets the two terms complement each other |
| `saliency_weighted_pool(..., top_p=0.25)` | Pool only the top-25% most salient clips | Avoids dilution from background clips |
| `softmax(s/τ), τ=0.1` | Sharp pooling | Concentrates weight on the very most salient clip(s) |

When you set `α = 0` you turn off the query path (the **thumb_only** ablation). When `α = 1` you turn off the direct visual path. The full plan ablates these.

---

## 5. Metrics

### Primary

- **AUC (ROC)** — area under the curve on the 600 (label, score) pairs. Implemented via Mann-Whitney U (no sklearn). **Target ≥ 0.75** with GT queries (R1 gate).

### Secondary

- **Cohen's d** — effect size of the pos vs neg separation. > 1.0 is large.
- **mean(pos) − mean(neg)** — raw separation in absolute score units.
- **Welch's t-test** — significance of the difference (uses `scipy` if available).
- **Confusion matrix at threshold = median** — accuracy / precision / recall / F1.

### Visual

- Score-distribution histogram (positive vs negative). Should show two clearly
  separated bumps if the pipeline works. If they overlap heavily → AUC ≈ 0.5.

---

## 6. Reading the result

```
======================================================
  R1 SMOKE TEST RESULT
======================================================
  AUC          : 0.834
  Cohen's d    : 1.42
  mean(pos)    : 0.387
  mean(neg)    : 0.211
  separation   : +0.176

  PASS — AUC 0.834 >= 0.75.
```

### Decision matrix

| AUC | What it means | Next action |
|---|---|---|
| ≥ 0.85 | Excellent — formula works very well with GT queries | Confidently train M2; expect clipcap_k3 in 0.75-0.85 range |
| 0.75–0.85 | Good — pipeline works | Train M2; expect clipcap_k3 ~5-10 pts below GT |
| 0.65–0.75 | Marginal — formula is borderline | Tune τ, top_p, α before training M2 |
| 0.55–0.65 | Weak — something is wrong | Inspect: low-saliency cases, M1 ckpt sanity, vid mismatches |
| < 0.55 | Broken | Check feature loading; check that positives/negatives aren't swapped |

---

## 7. After the smoke test passes — V2 ablation

The full evaluation (not in `run_pipeline_allinone.ipynb` yet) re-runs the same
loop **5 times** with different query sources:

| Source | Where the query comes from | Expected AUC |
|---|---|---|
| `gt_query` | Ground-truth QVH annotation (ceiling) | 0.83 ± 0.05 |
| `clipcap_k3` | 3 nucleus-sampled queries from trained M2, mean-aggregated | 0.75 ± 0.05 |
| `clipcap_k1` | 1 beam-3 query from trained M2 | 0.70 ± 0.05 |
| `thumb_only` | α = 0, no query path | 0.65 ± 0.05 |
| `random_query` | Random query from the eval set | ≈ 0.50 |

The publishable headline is the gap between **clipcap_k3** and **random_query**
(measures the value of M2) and between **gt_query** and **clipcap_k3** (measures
the quality cost of using generated queries).

---

## 8. Outputs that land on Drive

After running `run_pipeline_allinone.ipynb`:

```
thumbnail_project/outputs/
├── predictions/
│   └── allinone_gt_smoke.jsonl       # 600 rows: {pair_id, label, score}
└── figures/
    └── allinone_gt_smoke.png         # score-distribution histogram
```

These persist across Colab session restarts (they live on Drive).

---

## 9. Troubleshooting

| Symptom | Likely cause | Where to look |
|---|---|---|
| AUC ≈ 0.5 with GT queries | All-zero saliency → check M1 ckpt loaded correctly | Print `saliency.std()` in cell 9 — should not be 0 |
| `FileNotFoundError: features/...npz` for many vids | M1 features not extracted, or extracted to wrong path | See "Feature extraction issue" below |
| Cohen's d > 0 but AUC = 0 | Positives scoring LOWER than negatives | Check that `label = 1` means positive in your construction |
| `skipped` is high (>200) | Many vids missing features | Re-extract features; verify `feat_path` exists for sample vids |
| Random results between runs | RNG seeds not pinned | Verify `random.seed(2026)` ran in the eval-pair builder |

---

## 10. Connection to the writeup

- **R1 smoke test (this doc)** = the methodology validation in your paper.
- **clipcap_k3 vs random_query** AUC gap = the M2 contribution headline.
- **gt_query vs clipcap_k3** AUC gap = the "query-quality cost" finding.
- **thumb_only AUC** = the visual-only baseline you compare against.

For the paper, you'll report all five rows of the ablation table plus the
histogram for clipcap_k3 vs random_query.
