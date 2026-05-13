# M2 — Thumbnail → Query Generation: Architecture Analysis & Limitations

**Last updated:** 2026-05-07

---

## What M2 Is Trying to Do

M2 inverts the normal CLIP retrieval direction. Normally: text → find matching image. M2 does: image → generate descriptive text. It is a **visual captioning** model, specialized to produce the kind of short factual descriptions that annotators wrote for QVHighlights moments.

This is not a novel architecture problem — visual captioning is solved. The challenge is:
1. The training data (video frames) differs from inference data (stylized thumbnails).
2. The vocabulary of QVHighlights queries is specific and small.
3. The generated query needs to be semantically useful as a retrieval probe for M1, not just accurate.

---

## Current M2 Implementations (What You Built)

### Version 1 & 2 — CLIP + LSTM/GRU (Baseline)
```
Image → CLIP ViT-B/32 → 512-dim visual feature
                                │
                                ▼  (projected to embed_dim=256)
                        Decoder initial state
                                │
                        ┌───────┴──────────┐
                        ▼                  ▼
               LSTM (hidden=256)    GRU (hidden=256)
                        │
                        ▼
               Linear(256, vocab_size)
                        │
                        ▼
               Token prediction (custom vocab)
```

**Key design choices and their implications:**
- Frozen CLIP encoder — correct choice, prevents catastrophic forgetting, reduces compute
- Custom vocabulary from training queries — problematic (see Limitation 1 below)
- Teacher forcing during training — correct
- Greedy decoding at inference — acceptable for evaluation speed

### Version 3 — CLIP + Transformer Decoder
```
Image → CLIP ViT-B/32 → visual token features (patch embeddings) → (L_vis, 512)
                                │
                                ▼  projected to d_model=256
                       Visual token sequence
                                │
                       ┌────────┴────────┐
                       │  Transformer    │
                       │  Decoder        │ (2 layers, 4 heads)
                       │  cross-attn to  │
                       │  visual tokens  │
                       └────────┬────────┘
                                │
                       Linear(256, vocab_size)
```

v3 uses **visual token cross-attention** (all 196+1 patch tokens from ViT) instead of a single pooled feature. This gives the decoder spatial attention over image regions — architecturally stronger.

---

## Critical Limitations of Current M2

### Limitation 1 — Custom Vocabulary (Most Important Fix)

**Problem:** The vocabulary is built from the ~2000 training query strings. At 12–15 words per query with ~6000 unique tokens total, the model can only output words it saw during training. Unknown words (proper nouns, rare verbs, domain-specific terms) are all `<UNK>`.

**Impact:** At inference on real thumbnails, the generated query will be constrained to a small, specific vocabulary. If the thumbnail shows a chess game but "chess" isn't in the vocabulary, the model cannot say it.

**Fix:** Use a pre-trained tokenizer (BPE or WordPiece). The minimal change is switching from `word2idx` to `transformers` tokenizer. A better fix is fine-tuning GPT-2 instead of training a decoder from scratch.

**Research framing:** This vocabulary bottleneck is worth measuring explicitly. Run inference on a held-out set and count the `<UNK>` rate and vocabulary coverage.

### Limitation 2 — BLEU-1 Is a Weak Evaluation Metric

**Problem:** BLEU-1 is unigram overlap. It rewards models that produce high-frequency words ("man", "woman", "walks") regardless of semantic accuracy. A model that always outputs "a man talks to the camera" would score reasonably on BLEU-1 against a diverse query set.

**Impact:** You cannot trust BLEU-1 to validate that M2 improves. A model with lower BLEU-1 might produce more semantically useful retrieval queries.

**Better metrics:**
- **CIDEr** — consensus-based, weighted by TF-IDF (rewards distinctive words)
- **METEOR** — includes stemming and synonym matching
- **SPICE** — scene-graph based, better for factual description
- **CLIP similarity** — embed generated query and ground truth with CLIP text encoder, compute cosine similarity. This directly measures retrieval-relevant alignment.

For your project, **CLIP text similarity** is the most meaningful metric because it measures whether the generated query would retrieve the same moments as the ground-truth query.

### Limitation 3 — Training/Inference Domain Gap

**Problem:** M2 trains on natural video frames (extracted from QVHighlights clips) but receives stylized YouTube thumbnails at inference. Thumbnails have:
- Text overlays (channel name, title fragments)
- Color grading / saturation enhancement
- Zoomed-in faces
- Composite/edited visuals
- Artificial brightness/contrast

**Impact:** CLIP embeddings of thumbnails will differ from CLIP embeddings of raw frames even for the same scene. The decoder, trained to produce queries from frame-style embeddings, may generate degraded queries from thumbnail-style embeddings.

**This is the primary research finding to document experimentally.** Quantify:
```
CLIP_frame_embedding vs CLIP_thumbnail_embedding for same event
→ cosine distance = domain gap magnitude
```

### Limitation 4 — Short Training Set (1608 pairs)

**Problem:** 1608 (image, query) pairs is very small for visual captioning. Standard captioning models use COCO (120K pairs) or Conceptual Captions (3M pairs).

**Impact:** The decoder will overfit to common patterns in QVHighlights queries ("man walks", "woman talks", "people sit"). Rare or complex events will be poorly described.

**Mitigation without more data:**
- Pre-initialize decoder weights from a pre-trained language model (GPT-2 small)
- Aggressive dropout, label smoothing (v3 already uses `label_smoothing=0.05`)
- Data augmentation: perturb CLIP embeddings slightly during training

---

## Recommended M2 Architecture (Strong Baseline)

For a course project that still looks research-oriented, the strongest architecture with reasonable effort is:

```
Image → CLIP ViT-B/32 (frozen) → pooled 512-dim feature
              │
              ▼
    Linear projection (512 → 768)
              │
              ▼
    GPT-2 Small (fine-tuned, 117M params)
    Condition: prepend projected visual token to token sequence
              │
              ▼
    Generated query text (BPE vocabulary, real words)
```

This is essentially ClipCap (Mokady et al. 2021) — a published approach that directly addresses this task. Citing and adapting it is academically legitimate.

The key difference from your current v3: GPT-2 has a real vocabulary, real pre-trained language priors, and generates coherent text even on rare inputs. Fine-tuning it on 1608 samples is feasible in 2–4 hours on a T4.

---

## M2 Evaluation Strategy

### Intrinsic Evaluation (on M2 itself)
1. **CLIP text similarity:** `cos_sim(CLIP_text(generated), CLIP_text(ground_truth))` — primary metric
2. **CIDEr score** — for comparison with captioning literature
3. **Vocabulary coverage:** % of generated tokens not `<UNK>`
4. **Length distribution:** do generated queries match GT length distribution?

### Extrinsic Evaluation (M2 as component of full pipeline)
1. **Retrieval quality:** Feed generated query into M1. Compare retrieved moments with those from ground-truth query.
   - Metric: `R1@0.5(M2_query) / R1@0.5(GT_query)` — the ratio shows how much M2 degrades retrieval
2. **Consistency correlation:** On a labelled set of consistent/clickbait thumbnails, does the consistency score separate the two classes?

---

## Open Question: What Makes a Good M2 Query for This Pipeline?

This is not obvious. A query like "woman smiles" might be:
- Good BLEU (matches GT if GT says "woman smiles at camera")
- Bad retrieval (too generic, matches too many moments)
- Good consistency score (specific enough to locate one region)

A good M2 query for the **pipeline** needs to be specific enough to produce discriminative moment retrieval, not just accurate against GT text. This suggests tuning M2 with a **retrieval-in-the-loop** loss — but that's complex to implement. For the course project, CLIP text similarity to GT is the practical proxy.
