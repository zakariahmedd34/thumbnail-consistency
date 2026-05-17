"""
Build notebooks/pipeline_walkthrough.ipynb — an explanatory tour of the project.

Each section is (markdown explanation, code cell) pairs. The notebook is fully
runnable on CPU with synthetic data — no GPU, no M1 checkpoint, no real videos
needed. It is meant for discussion and onboarding, not production runs.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "pipeline_walkthrough.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = []

# ---------- Title --------------------------------------------------------------
CELLS.append(md("""\
# Pipeline Walkthrough — Thumbnail-Video Semantic Consistency

> **Audience:** anyone new to this project.
> **Purpose:** a guided tour of every `src/` module, with markdown explanations
> before each code block. Runnable on CPU using **synthetic data** so you can
> walk through it without needing GPU access or real videos.

This notebook is for *discussion and onboarding*. For real training, see
`notebooks/train_clipcap.ipynb`. For real evaluation, see
`notebooks/run_evaluation.ipynb`.

**Reading order before this notebook:**
1. [README.md](../README.md) — high-level intro
2. [docs/FINAL_DESIGN.md](../docs/FINAL_DESIGN.md) — architecture decisions
3. [docs/GUIDE.md](../docs/GUIDE.md) — navigation guide

---

## What we are building

```
Thumbnail T
  ├── CLIP-img (frozen) ─────────────────────────► e_T ───┐
  └── ClipCap-M2 ★ trained ─► query q ─► CLIP-text ─► e_q │
                                                          ▼
              Moment-DETR (official ckpt, frozen) ──► saliency s[t]
                                                          │
                              saliency-softmax pool ──► e_clip_w
                                                          │
              C = mean_k max_w [0.5*cos(e_q, e_clip) + 0.5*cos(e_T, e_clip)]
```

**The only thing that learns** is the M2 prefix MLP. Everything else is
pretrained-and-frozen or pure deterministic math.

This notebook will demonstrate each box in that diagram with a small runnable
example."""))


# ---------- Section 1: Setup ---------------------------------------------------
CELLS.append(md("""\
---

## 1. Setup — make `src/` importable

We add the project root to `sys.path` so that `from src.xxx import yyy` works
from inside a notebook. This is the only path-manipulation we ever do — every
real piece of logic lives under `src/`."""))

CELLS.append(code("""\
import sys
from pathlib import Path

PROJECT = Path('.').resolve()
# In Jupyter the cwd is usually the notebook's folder; walk up if needed.
while PROJECT.name == 'notebooks':
    PROJECT = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

print(f'Project root: {PROJECT}')
print(f'sys.path[0] : {sys.path[0]}')
"""))


# ---------- Section 2: Pure-math foundation -----------------------------------
CELLS.append(md("""\
---

## 2. The heart of the project: `src/scoring/aggregate.py`

This module is **pure NumPy**. No GPU, no torch, no models. It implements the
mathematical core of the consistency score:

| Function | What it computes |
|---|---|
| `cosine(a, b)` | Cosine similarity between two vectors |
| `softmax(x)` | Softmax along an axis |
| `saliency_weighted_pool(v, s, tau, top_p)` | Saliency-weighted pooled video feature |
| `per_chunk_score(e_q, e_T, e_clip, alpha)` | Per-(query, chunk) score |
| `aggregate_per_query([...])` | `max` over chunks for one query |
| `aggregate_consistency([...])` | `mean` over queries -> final C(T, V) |
| `consistency_score(...)` | The whole thing, end-to-end |

Because everything here is pure functions, **all of it is unit-tested**
([tests/test_aggregate.py](../tests/test_aggregate.py) -> 18/18 pass).

Let's tour it."""))


CELLS.append(md("""\
### 2.1 Cosine similarity — the building block

The whole score reduces to cosine similarities between three things: the
thumbnail embedding, the query embedding, and the pooled-video embedding."""))

CELLS.append(code("""\
import numpy as np
from src.scoring.aggregate import cosine

a = np.array([1.0, 0.0, 0.0])
b = np.array([0.7, 0.7, 0.0])
c = np.array([0.0, 1.0, 0.0])
d = -a

print(f'cosine(a, a) = {cosine(a, a):+.3f}   # identical -> 1')
print(f'cosine(a, b) = {cosine(a, b):+.3f}   # 45-degree apart -> ~0.707')
print(f'cosine(a, c) = {cosine(a, c):+.3f}   # orthogonal -> 0')
print(f'cosine(a, d) = {cosine(a, d):+.3f}   # opposite -> -1')
"""))


CELLS.append(md("""\
### 2.2 Saliency-weighted pooling — `e_clip_w`

For a video chunk of `T` clips, M1 gives us a saliency vector `s ∈ R^T`.
We *don't* just average the video features. Instead:

1. Keep the **top-p fraction** of clips by saliency (default 25%).
2. Convert those scores to weights via `softmax(s / τ)` with `τ=0.1`.
3. Take a **weighted average** of the corresponding video features.

This makes the pooled embedding focus on the temporally most relevant region
rather than diluting across the whole video.

**Toy demo:** four clips, only one is salient. The pooled embedding should
recover the salient clip's direction exactly."""))

CELLS.append(code("""\
from src.scoring.aggregate import saliency_weighted_pool

rng = np.random.default_rng(0)
T, D = 4, 8

target = rng.normal(size=D)
other  = rng.normal(size=D)
v_feat = np.stack([other, other, target, other])  # only clip 2 = target
sal    = np.array([0.1, 0.1, 5.0, 0.1])           # clip 2 dominates

pooled = saliency_weighted_pool(v_feat, sal, tau=0.1, top_p=0.25)

print(f'cosine(pooled, target) = {cosine(pooled, target):+.4f}   # ~= 1.0  OK')
print(f'cosine(pooled, other)  = {cosine(pooled, other):+.4f}   # ~= random')
"""))


CELLS.append(md("""\
### 2.3 Per-chunk score — what `alpha` controls

For one query `q` and one chunk `w`:

```
c(k, w) = alpha * cos(e_q, e_clip_w) + (1-alpha) * cos(e_T, e_clip_w)
```

- `alpha = 1` -> score uses **only the query path** (M2 must be good for high scores).
- `alpha = 0` -> score uses **only the thumbnail directly** (no M2, pure visual).
- `alpha = 0.5` -> balanced (V1 default).

The ablation that tells you whether M2 helps is the comparison between
`alpha=0.5` (with ClipCap queries) and `alpha=0` (`thumb_only`)."""))

CELLS.append(code("""\
from src.scoring.aggregate import per_chunk_score

D = 4
e_q     = np.array([1.0, 0.0, 0.0, 0.0])
e_T     = np.array([0.0, 1.0, 0.0, 0.0])
e_clipA = np.array([1.0, 0.0, 0.0, 0.0])  # video aligned with query
e_clipB = np.array([0.0, 1.0, 0.0, 0.0])  # video aligned with thumbnail
e_clipC = np.array([0.0, 0.0, 1.0, 0.0])  # video aligned with neither

print('alpha=0.5   (default)')
print(f'  query-aligned  video -> {per_chunk_score(e_q, e_T, e_clipA, alpha=0.5):+.3f}')
print(f'  thumb-aligned  video -> {per_chunk_score(e_q, e_T, e_clipB, alpha=0.5):+.3f}')
print(f'  unrelated      video -> {per_chunk_score(e_q, e_T, e_clipC, alpha=0.5):+.3f}')

print()
print('alpha=1.0   (query path only)')
print(f'  query-aligned  -> {per_chunk_score(e_q, e_T, e_clipA, alpha=1.0):+.3f}')
print(f'  thumb-aligned  -> {per_chunk_score(e_q, e_T, e_clipB, alpha=1.0):+.3f}')

print()
print('alpha=0.0   (thumb_only ablation)')
print(f'  query-aligned  -> {per_chunk_score(e_q, e_T, e_clipA, alpha=0.0):+.3f}')
print(f'  thumb-aligned  -> {per_chunk_score(e_q, e_T, e_clipB, alpha=0.0):+.3f}')
"""))


CELLS.append(md("""\
### 2.4 Multi-query, multi-chunk aggregation

The final score combines `K` queries over `W` chunks:

```
per_query[k]  = max_w c(k, w)               # best matching chunk for query k
C(T, V)       = mean_k per_query[k]         # average across the K queries
```

**Why `max` over chunks, `mean` over queries?**

- A consistent thumbnail only needs *some* part of the video to match -> take the
  best chunk (`max_w`).
- The thumbnail's truthful meaning should be supported by *most* generated
  queries -> average across queries (`mean_k`), not max.

Use `max_k` instead of `mean_k` and you'd inflate scores: any single lucky
query wins. The mean is the conservative choice."""))

CELLS.append(code("""\
from src.scoring.aggregate import consistency_score

D, T = 8, 6

rng = np.random.default_rng(42)
target = rng.normal(size=D); target /= np.linalg.norm(target)
e_T = target
e_q1 = target            # query 1 well-aligned with thumbnail
e_q2 = rng.normal(size=D); e_q2 /= np.linalg.norm(e_q2)  # query 2 random

# Two chunks: chunk 0 contains a clip pointing to `target`; chunk 1 random.
v0 = rng.normal(size=(T, D)); v0[3] = target * 4
v1 = rng.normal(size=(T, D))
sal_q1_c0 = np.array([0.1, 0.1, 0.2, 5.0, 0.1, 0.1])  # M1 found target in chunk 0
sal_q1_c1 = rng.normal(size=T)
sal_q2_c0 = rng.normal(size=T)
sal_q2_c1 = rng.normal(size=T)

chunks = [
    {'v_feat': v0, 'saliency': [sal_q1_c0, sal_q2_c0]},
    {'v_feat': v1, 'saliency': [sal_q1_c1, sal_q2_c1]},
]

out = consistency_score(e_T=e_T, e_q_list=[e_q1, e_q2], chunks=chunks)
print(f'C(T, V)            : {out[\"C\"]:+.4f}')
print(f'per_query (K=2)    : {[round(x,3) for x in out[\"per_query\"]]}')
print(f'per_query_chunk    :')
for k, row in enumerate(out['per_query_chunk']):
    print(f'   query {k}: chunks -> {[round(x,3) for x in row]}')
"""))


# ---------- Section 3: Chunking -----------------------------------------------
CELLS.append(md("""\
---

## 3. Chunking — handling videos longer than 150 seconds

`src/scoring/chunking.py` splits a video's clip-feature sequence into windows
of size 75 (M1's max input) with stride 60 (overlap 15 = 30 seconds).

| Input video length | Behavior |
|---|---|
| ≤ 75 clips (≤ 150s) | Single chunk, no-op |
| > 75 clips | Sliding windows with 30-second overlap |"""))

CELLS.append(code("""\
from src.scoring.chunking import chunk_indices

print('Short video (50 clips):    ', chunk_indices(50))
print('Exactly 75 clips (150s):   ', chunk_indices(75))
print('Long video (180 clips, 6m):', chunk_indices(180))
print('Very long (300 clips, 10m):', chunk_indices(300))
"""))


# ---------- Section 4: Evaluation metrics ------------------------------------
CELLS.append(md("""\
---

## 4. Evaluation metrics — `src/eval/metrics.py`

This module produces the headline numbers reported in the writeup:

- **ROC-AUC** via Mann-Whitney U (no sklearn dependency).
- **Cohen's d** — effect size of pos vs neg separation.
- **Welch's t** — significance test (uses `scipy` if installed, fallback to t-stat only).

Below: a perfect separation should give AUC=1.0, an inverted scoring AUC=0.0,
random scoring ~= 0.5."""))

CELLS.append(code("""\
from src.eval.metrics import roc_auc, summarize_separation, confusion_at_threshold

scores_perfect  = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
scores_inverted = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
labels          = [1, 1, 1, 0, 0, 0]

print(f'AUC (perfect):  {roc_auc(scores_perfect,  labels):.3f}')
print(f'AUC (inverted): {roc_auc(scores_inverted, labels):.3f}')

# Random control
rng = np.random.default_rng(0)
n = 300
s = rng.uniform(0, 1, size=2*n).tolist()
l = [1]*n + [0]*n
print(f'AUC (random):   {roc_auc(s, l):.3f}')

# Full summary
print()
print('Full summary (perfect case):')
import json
print(json.dumps(summarize_separation(scores_perfect, labels), indent=2))
"""))


# ---------- Section 5: ClipCap M2 ---------------------------------------------
CELLS.append(md("""\
---

## 5. M2 = ClipCap (★ the only trained component)

`src/models/clipcap.py` defines `ClipCapM2`:

```
[CLIP image emb (512)] ─► [Prefix MLP, TRAINED, ~12M params]
                                        │
                                        ▼
                          [10 prefix tokens x 768d]
                                        │
                                        ▼
                          [GPT-2 small, FROZEN, 117M params]
                                        │
                                        ▼
                          generated query (BPE tokens)
```

The whole trick of ClipCap is: GPT-2 already speaks English fluently. You don't
have to re-learn language modeling — you just have to learn how to "translate"
a CLIP image embedding into prefix tokens that GPT-2 can continue.

The Prefix MLP is the ONLY thing that updates during M2 training.
Loss: cross-entropy against the GT query, with label smoothing 0.05.

### Why frozen GPT-2 works on only 1608 training pairs

Image captioning models that train from scratch (LSTM/Transformer decoders,
v1/v2/v3 in `notebooks/model 2/`) need huge amounts of data because they must
learn English from those pairs. ClipCap doesn't — it inherits all of GPT-2's
language priors and only needs to learn the cross-modal alignment, which is a
much smaller problem."""))

CELLS.append(md("""\
### 5.1 Inspect the model — what is trainable?

Loading GPT-2 is slow (~10s) on first call because it downloads the weights.
After the first run it's cached on disk."""))

CELLS.append(code("""\
import torch
from src.models.clipcap import ClipCapM2

model = ClipCapM2(clip_dim=512, prefix_len=10, gpt2_name='gpt2', freeze_gpt2=True)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f'Trainable params : {trainable:>12,}')
print(f'Frozen params    : {frozen:>12,}')
print(f'Trainable / total: {trainable / (trainable+frozen):.2%}')
print()
print('Trainable modules:')
for n, p in model.named_parameters():
    if p.requires_grad:
        print(f'  {n:40s} {tuple(p.shape)}')
"""))


CELLS.append(md("""\
### 5.2 Forward pass — synthetic CLIP embedding + a target caption

This is exactly what happens during training. The logits returned correspond
to the next-token prediction for each position in the caption."""))

CELLS.append(code("""\
B = 2
# Fake CLIP image embeddings (L2-normalized).
fake_clip = torch.randn(B, 512)
fake_clip = fake_clip / fake_clip.norm(dim=-1, keepdim=True)

# Fake captions tokenized with GPT-2 BPE.
tok = model.tokenizer
text = ['<|endoftext|> a man walks in a park <|endoftext|>',
        '<|endoftext|> a dog plays with a ball <|endoftext|>']
inp = tok(text, return_tensors='pt', padding='max_length', max_length=12, truncation=True)
input_ids       = inp['input_ids']
attention_mask  = inp['attention_mask']

logits = model(fake_clip, input_ids, attention_mask)
print(f'input_ids shape : {tuple(input_ids.shape)}')
print(f'logits shape    : {tuple(logits.shape)}     # (B, T, vocab=50257)')
print(f'vocab size      : {logits.size(-1)}')
"""))


CELLS.append(md("""\
### 5.3 Generation — nucleus sampling for K=3 queries

The trained model produces semantically diverse queries via nucleus sampling
(`top_p=0.9, temperature=1.1`) — not beam search (which gives boring
paraphrases).

⚠️ **Note:** this notebook uses an *untrained* model. The generated text will
be incoherent — that's expected, because the prefix MLP is random. After real
training, this same call produces sensible captions."""))

CELLS.append(code("""\
gens = model.generate(
    fake_clip,
    max_new_tokens=15,
    num_return_sequences=3,
    do_sample=True,
    top_p=0.9,
    temperature=1.1,
    no_repeat_ngram_size=2,
)
for i, qs in enumerate(gens):
    print(f'Sample {i} — 3 nucleus-sampled queries:')
    for q in qs:
        print(f'  • {q[:80]}')
    print()
print('(Untrained model -> text is incoherent. Real training fixes this.)')
"""))


# ---------- Section 6: Moment-DETR wrapper ------------------------------------
CELLS.append(md("""\
---

## 6. M1 = Moment-DETR wrapper

`src/models/moment_detr.py` provides two classes:

| Class | Use when |
|---|---|
| `MomentDETRFeaturePredictor` | You have pre-extracted CLIP features (.npz). **This is the fast path for our evaluation loop.** |
| `MomentDETRVideoPredictor` | You have raw video files and want M1 to re-encode them. Use for ad-hoc demos. |

Both return the same dict structure:

```python
{
    'query':                  str,
    'pred_relevant_windows':  [[start_s, end_s, fg_score], ... 10 rows],
    'pred_saliency_scores':   [s_0, s_1, ..., s_T-1],   # per-clip
}
```

This notebook will not actually load the M1 checkpoint (that requires
downloading the file). Instead we'll inspect the API and the helper functions
that the consistency scorer uses to load features."""))

CELLS.append(code("""\
from src.models.moment_detr import (
    MomentDETRFeaturePredictor, MomentDETRVideoPredictor,
    load_video_features, load_text_features,
)

# Show class signatures without loading anything.
print('MomentDETRFeaturePredictor:')
print('  __init__(ckpt_path, repo_root, device=None)')
print('  predict_from_features(v_feat: (T, 512), t_feats: List[(L_q, 512)], queries) -> List[dict]')
print()
print('MomentDETRVideoPredictor:')
print('  __init__(ckpt_path, repo_root, clip_model=\"ViT-B/32\", device=None)')
print('  predict(video_path: str, queries: List[str]) -> List[dict]')
print()
print('Helpers:')
print('  load_video_features(feature_dir, vid)  -> np.ndarray (T, 512)')
print('  load_text_features(feature_dir, qid)   -> np.ndarray (L_q, 512)')
"""))


# ---------- Section 7: ConsistencyScorer end-to-end ---------------------------
CELLS.append(md("""\
---

## 7. Putting it all together — `ConsistencyScorer`

`src/scoring/consistency.py` ties the pieces together:

1. Embed the thumbnail with CLIP-img -> `e_T`.
2. Embed the queries with CLIP-text -> `{e_q_k}`.
3. Load the video's CLIP features -> `v_feat`.
4. Chunk if `>75` clips.
5. For each chunk, ask M1 for saliency given each query.
6. Pool saliency-weighted, compute per-chunk scores, aggregate.

In production this needs the M1 checkpoint. In this notebook we'll **mock the
M1 predictor** so we can demonstrate the end-to-end path on a CPU."""))

CELLS.append(code("""\
import numpy as np
from src.scoring.consistency import ConsistencyScorer, ScoringConfig


class FakeM1Predictor:
    \"\"\"Mock that returns deterministic synthetic saliency.\"\"\"
    def predict_from_features(self, v_feat, t_feats, queries=None):
        T = v_feat.shape[0]
        out = []
        for q_idx, _ in enumerate(t_feats):
            # Pretend the relevant moment is clip T//2 (the middle).
            sal = np.zeros(T, dtype=np.float32)
            sal[T // 2] = 5.0
            sal[T // 2 - 1] = 3.0
            sal[T // 2 + 1] = 3.0
            spans = [[0.0, float(T)*2, 0.9]] + [[0.0, 0.0, 0.0]] * 9
            out.append({
                'query': queries[q_idx] if queries else None,
                'pred_relevant_windows': spans,
                'pred_saliency_scores': sal.tolist(),
            })
        return out


class FakeFeatureDir:
    \"\"\"Mock load_video_features by intercepting the Path lookup.\"\"\"
    pass


# Build a scorer with the mock; we won't actually call .score() in this cell
# because that path uses real CLIP image encoding (slow first time).
fake_m1 = FakeM1Predictor()
scoring_cfg = ScoringConfig(alpha=0.5, tau=0.1, top_p=0.25)
print('ScoringConfig:', scoring_cfg)
print()
print('FakeM1Predictor returns synthetic saliency centered at the middle clip.')
print('Below we compose the consistency score MANUALLY using aggregate.py,')
print('without going through encode_image (which needs the CLIP model).')
"""))


CELLS.append(md("""\
### 7.1 Hand-wiring the full score with synthetic embeddings

Skipping CLIP-image encoding (which would load the full CLIP model). We
hand-construct embeddings and call the aggregation directly. This is exactly
what `ConsistencyScorer.score(...)` does internally, minus the encoding step."""))

CELLS.append(code("""\
from src.scoring.aggregate import consistency_score

D = 512
T = 75   # one 150-second chunk
rng = np.random.default_rng(7)

# === Build a 'consistent' pair ============================================
target = rng.normal(size=D); target /= np.linalg.norm(target)
e_T  = target                            # thumbnail embeds to `target`
e_q  = target + 0.05 * rng.normal(size=D); e_q /= np.linalg.norm(e_q)
# video: random clips except clip T//2 (and neighbors) point to `target`
v_pos = rng.normal(size=(T, D))
v_pos[T//2]   = target * 5
v_pos[T//2-1] = target * 3
v_pos[T//2+1] = target * 3
v_pos = v_pos / np.linalg.norm(v_pos, axis=-1, keepdims=True)

# M1 saliency: concentrated at the relevant region.
sal_pos = fake_m1.predict_from_features(v_pos, [np.zeros((1, D))], ['q'])[0]
sal_pos = np.asarray(sal_pos['pred_saliency_scores'])

out_pos = consistency_score(
    e_T=e_T, e_q_list=[e_q],
    chunks=[{'v_feat': v_pos, 'saliency': [sal_pos]}],
    alpha=0.5, tau=0.1, top_p=0.25,
)

# === Build a 'mismatched' pair (random video) =============================
v_neg = rng.normal(size=(T, D))
v_neg = v_neg / np.linalg.norm(v_neg, axis=-1, keepdims=True)
sal_neg = fake_m1.predict_from_features(v_neg, [np.zeros((1, D))], ['q'])[0]
sal_neg = np.asarray(sal_neg['pred_saliency_scores'])

out_neg = consistency_score(
    e_T=e_T, e_q_list=[e_q],
    chunks=[{'v_feat': v_neg, 'saliency': [sal_neg]}],
    alpha=0.5, tau=0.1, top_p=0.25,
)

print(f'Consistent pair  -> C = {out_pos[\"C\"]:+.4f}')
print(f'Mismatched pair  -> C = {out_neg[\"C\"]:+.4f}')
print(f'Separation       -> {out_pos[\"C\"] - out_neg[\"C\"]:+.4f}')
print()
print('A working pipeline should produce a large positive separation here.')
"""))


# ---------- Section 8: scaling to the real eval set ---------------------------
CELLS.append(md("""\
---

## 8. Scaling up — what happens on the real eval set

In `notebooks/run_evaluation.ipynb`, the same `ConsistencyScorer` is called
600 times: once per `(thumbnail, video, queries)` triple from
`processed/consistency_eval.jsonl`. The 600 resulting `C` values plus their
ground-truth `label ∈ {0, 1}` are fed into `summarize_separation`.

The expected headline numbers (target ranges):

| Source | Expected AUC | What it tells us |
|---|---|---|
| `gt_query` (ground-truth QVH query) | 0.85–0.95 | Ceiling — formula upper bound |
| `clipcap_k3` (3 nucleus M2 queries) | 0.75–0.85 | What M2 actually delivers |
| `clipcap_k1` (single beam-3 M2 query) | 0.70–0.80 | Cost of using a single query |
| `thumb_only` (`alpha=0`, no query path) | 0.65–0.75 | Visual-only baseline |
| `random_query` (control) | 0.45–0.55 | Sanity — should be ~= random |

If `clipcap_k3` does NOT beat `random_query`, M2 isn't contributing signal.
If `thumb_only` already wins, M2 isn't contributing *over* direct visual match.

Below: a synthetic version of this experiment so you can see what the output
table looks like before running the real one."""))

CELLS.append(code("""\
from src.eval.metrics import summarize_separation
import random

rng = np.random.default_rng(0)
N = 300

# Simulate four query-source experiments at different separation strengths.
def fake_scores(separation_mean, n_pos=N, n_neg=N):
    pos = rng.normal(loc=0.40 + separation_mean, scale=0.10, size=n_pos)
    neg = rng.normal(loc=0.40 - separation_mean, scale=0.10, size=n_neg)
    scores = np.concatenate([pos, neg])
    labels = np.array([1]*n_pos + [0]*n_neg)
    return scores, labels

experiments = {
    'gt_query':     fake_scores(0.18),
    'clipcap_k3':   fake_scores(0.12),
    'clipcap_k1':   fake_scores(0.09),
    'thumb_only':   fake_scores(0.07),
    'random_query': fake_scores(0.00),
}

print(f'{'source':15s}  {'AUC':>6s}  {'Cohen d':>9s}  {'sep':>6s}')
print('-' * 45)
for src, (s, l) in experiments.items():
    out = summarize_separation(s, l)
    print(f'{src:15s}  {out['auc']:6.3f}  {out['cohens_d']:9.3f}  '
          f'{out['mean_pos'] - out['mean_neg']:+.3f}')
print()
print('In the real eval, the ROW ORDER above is the key result.')
print('clipcap_k3 should beat thumb_only AND random_query.')
"""))


# ---------- Section 9: How everything connects --------------------------------
CELLS.append(md("""\
---

## 9. How everything connects — diagram + file map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              At RUNTIME                                  │
└──────────────────────────────────────────────────────────────────────────┘

   thumbnail.jpg ──► src.models.encoders.encode_image  ──► e_T (512,)
                                                                │
                                                                ▼
   thumbnail.jpg ──► src.models.clipcap.ClipCapM2.generate ──► [q_1,q_2,q_3]
                                       │
                                       ▼
                          src.models.encoders.encode_text  ──► {e_q_k} (512,)
                                                                │
                                                                ▼
   processed/feature_dir/vid.npz ──► src.models.moment_detr.load_video_features
                                                                │
                                                                ▼
                                      src.scoring.chunking.chunk_indices
                                                                │
                                                                ▼
                       MomentDETRFeaturePredictor.predict_from_features
                                       │
                                       ▼
                              saliency s[t] per clip per query
                                       │
                                       ▼
                    src.scoring.aggregate.saliency_weighted_pool ──► e_clip_w
                                       │
                                       ▼
                    src.scoring.aggregate.per_chunk_score(alpha=0.5)
                                       │
                                       ▼
                    src.scoring.aggregate.aggregate_consistency
                                       │
                                       ▼
                                   C(T, V)
                                       │
                                       ▼
                       src.eval.metrics.summarize_separation
                                       │
                                       ▼
                       AUC, Cohen's d, t-test on 600 pairs
```

### File-to-step map

| Step | File |
|---|---|
| Encode thumbnail | [src/models/encoders.py](../src/models/encoders.py) -> `encode_image` |
| Generate queries | [src/models/clipcap.py](../src/models/clipcap.py) -> `ClipCapM2.generate` |
| Encode queries | [src/models/encoders.py](../src/models/encoders.py) -> `encode_text` |
| Load video features | [src/models/moment_detr.py](../src/models/moment_detr.py) -> `load_video_features` |
| Chunk | [src/scoring/chunking.py](../src/scoring/chunking.py) -> `chunk_indices` |
| Run M1 | [src/models/moment_detr.py](../src/models/moment_detr.py) -> `MomentDETRFeaturePredictor` |
| Pool saliency | [src/scoring/aggregate.py](../src/scoring/aggregate.py) -> `saliency_weighted_pool` |
| Per-chunk score | [src/scoring/aggregate.py](../src/scoring/aggregate.py) -> `per_chunk_score` |
| Final aggregation | [src/scoring/aggregate.py](../src/scoring/aggregate.py) -> `aggregate_consistency` |
| Headline metrics | [src/eval/metrics.py](../src/eval/metrics.py) -> `summarize_separation` |"""))


# ---------- Section 10: Where to go next --------------------------------------
CELLS.append(md("""\
---

## 10. Where to go next

You have now seen every component of the V1 pipeline. To run the real thing:

| Day-1 evening | [sanity_consistency.ipynb](sanity_consistency.ipynb) | GT-query AUC smoke test (R1 gate) |
| Day 1 -> 2 | [train_clipcap.ipynb](train_clipcap.ipynb) | Train M2, generate K=3 queries |
| Day 2 | [run_evaluation.ipynb](run_evaluation.ipynb) | Full ablation, produce headline numbers |

### Checklist for discussion

When you discuss this notebook with the team, the key questions to align on:

1. Are we comfortable that **M2 is the only thing that trains?**
2. Are we comfortable with `alpha=0.5` (equal weight on query vs thumbnail paths)?
3. Are we comfortable with **mean over queries, max over chunks** as aggregation?
4. Is the 600-pair AUC the right *primary* metric (vs the 50-pair manual)?
5. Do we plan to run the `top_p` sensitivity sweep on Day 3?

If everyone says yes to (1)–(4), the V1 plan is locked. (5) is a stretch goal.

### When to revisit this notebook

- A new teammate joins -> send them here first.
- Someone proposes a change to the score formula -> demonstrate it here first.
- A reviewer asks "why this aggregation?" -> cells §2.4 and §8 are the answer.
"""))


# ---------- write -------------------------------------------------------------
def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} with {len(CELLS)} cells "
          f"({sum(1 for c in CELLS if c['cell_type']=='code')} code, "
          f"{sum(1 for c in CELLS if c['cell_type']=='markdown')} md).")


if __name__ == "__main__":
    main()
