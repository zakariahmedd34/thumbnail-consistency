"""
Build notebooks/run_pipeline_allinone.ipynb — a single Colab notebook that
runs the R1 smoke test (and optionally M2 training) end-to-end.

ZERO dependencies on src/. All scoring, M1 wrappers, CLIP encoding, AUC
computation, etc. are inline. Drive paths hardcoded so it just works.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "run_pipeline_allinone.ipynb"
CELLS = []


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


# ---------- Title ----------------------------------------------------------------
CELLS.append(md("""\
# All-in-one Pipeline — Thumbnail-Video Consistency

**Single notebook. No `src/`. No configs. No PAT.** Mount Drive, run cells top to
bottom, get an AUC number.

This is the **R1 smoke test**: scores 600 (thumbnail, video) pairs from
`consistency_eval.jsonl` using **ground-truth queries** (no M2 yet). It tells you
whether the consistency formula works.

Expected runtime: **~10 minutes on T4 GPU.**

| What this notebook does | What it does NOT do |
|---|---|
| Mount Drive | Train ClipCap (different notebook) |
| Install deps | Run full ablation |
| Load M1 official checkpoint | Use `src/` modules |
| Encode all 600 thumbnails + queries with CLIP | Read any YAML config |
| Run M1 saliency for each pair | |
| Compute consistency score C(T, V) | |
| Print AUC, Cohen's d, histogram | |

If `AUC >= 0.75` at the end, the pipeline works → next step is M2 training.
If `AUC < 0.75`, the score formula needs debugging (don't train M2 yet).
"""))


# ---------- Setup ----------------------------------------------------------------
CELLS.append(md("## 1. Setup — mount Drive, install deps"))

CELLS.append(code("""\
import os, sys, subprocess, json, time
from pathlib import Path

# Mount Drive
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/thumbnail_project')
assert DRIVE_ROOT.exists(), f'Data root missing: {DRIVE_ROOT}'

# Install (idempotent — silent if already installed)
print('Installing deps (silent if already installed)...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers', 'torch', 'tqdm', 'matplotlib', 'gdown'],
               check=True)
print('OK')
"""))


# ---------- Clone upstream moment_detr -------------------------------------------
CELLS.append(md("""\
## 2. Get the upstream Moment-DETR code

We need the M1 model architecture (`build_model`, `span_utils`, etc.). Clone the
upstream repo into `/content/` so it's on `sys.path`.
"""))

CELLS.append(code("""\
MD_DIR = Path('/content/moment_detr_upstream')
if not MD_DIR.exists():
    subprocess.run(['git', 'clone',
                    'https://github.com/jayleicn/moment_detr.git',
                    str(MD_DIR)], check=True)
if str(MD_DIR) not in sys.path:
    sys.path.insert(0, str(MD_DIR))
print(f'Moment-DETR code available at {MD_DIR}')
"""))


# ---------- Extract features (idempotent per-session) ----------------------------
CELLS.append(md("""\
## 2.5 Extract M1 features to Colab local SSD (idempotent per session)

The features archive is ~7.5 GB. Drive can't reliably store the 19,000 small
.npz files inside it, so we extract to Colab's local SSD instead (~3-5 min).

The extraction runs **only if features aren't already extracted in this Colab
session**. So:
- Fresh session: takes ~5 min
- Same session, re-running the notebook: instant skip
"""))

CELLS.append(code("""\
import tarfile, shutil, time
ARCHIVE   = DRIVE_ROOT / 'moment_detr_features.tar.gz'
LOCAL_OUT = Path('/content/features_local')

assert ARCHIVE.exists(), f'Archive missing on Drive: {ARCHIVE}'

# Find existing extraction if any (handles nested layouts)
def _find_feature_dirs(root):
    clip = text = None
    for r, dirs, _ in os.walk(root):
        for d in dirs:
            p = Path(r) / d
            if d == 'clip_features' and len(list(p.glob('*.npz'))) > 1000:
                clip = p
            if d == 'clip_text_features' and len(list(p.glob('*.npz'))) > 1000:
                text = p
    return clip, text

clip_dir, text_dir = (None, None)
if LOCAL_OUT.exists():
    clip_dir, text_dir = _find_feature_dirs(LOCAL_OUT)

if clip_dir and text_dir:
    print(f'Features already extracted in this session:')
else:
    print(f'Extracting {ARCHIVE.stat().st_size/1e9:.2f} GB to {LOCAL_OUT} ...')
    if LOCAL_OUT.exists():
        shutil.rmtree(LOCAL_OUT)
    LOCAL_OUT.mkdir(parents=True)
    t0 = time.time()
    with tarfile.open(ARCHIVE) as tf:
        tf.extractall(LOCAL_OUT)
    print(f'  extracted in {time.time()-t0:.0f}s')
    clip_dir, text_dir = _find_feature_dirs(LOCAL_OUT)

assert clip_dir and text_dir, 'Feature dirs not found after extraction'
print(f'  clip_features      : {clip_dir}  ({len(list(clip_dir.glob(\"*.npz\"))):,} files)')
print(f'  clip_text_features : {text_dir}  ({len(list(text_dir.glob(\"*.npz\"))):,} files)')
"""))


# ---------- M1 checkpoint --------------------------------------------------------
CELLS.append(md("""\
## 3. Load the M1 (Moment-DETR) checkpoint

Tries Drive first; falls back to direct download (~30 MB).
"""))

CELLS.append(code("""\
import torch

CKPT_CANDIDATES = [
    DRIVE_ROOT / 'moment_detr' / 'run_on_video' / 'moment_detr_ckpt' / 'model_best.ckpt',
    DRIVE_ROOT / 'checkpoints' / 'moment_detr_best.ckpt',
    DRIVE_ROOT / 'checkpoints' / 'model_best.ckpt',
]
CKPT = next((p for p in CKPT_CANDIDATES if p.exists()), None)

if CKPT is None:
    print('M1 ckpt not on Drive. Downloading...')
    target = Path('/content/m1_model_best.ckpt')
    subprocess.run(['gdown', '--id', '1FXuYHDQQTC0CY9JpJoeRgwQH8a4QJqVN',
                    '-O', str(target)], check=True)
    CKPT = target

print(f'Using ckpt: {CKPT}')

# Build the model from upstream code
from moment_detr.model import build_model
from moment_detr.span_utils import span_cxw_to_xx
from utils.tensor_utils import pad_sequences_1d
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cpu':
    print('  WARNING: No GPU detected. M1 inference will be ~10x slower.')
    print('  Recommend: Runtime -> Change runtime type -> T4 GPU before continuing.')

ckpt_data = torch.load(str(CKPT), map_location='cpu', weights_only=False)
opt = ckpt_data['opt']

# Override opt.device so build_model doesn't unconditionally call torch.cuda.*
# The ckpt was trained with opt.device='cuda'; we may be on a CPU runtime.
opt.device = device

m1, _ = build_model(opt)
m1.load_state_dict(ckpt_data['model'])
m1.to(device).eval()
for p in m1.parameters():
    p.requires_grad = False
print(f'M1 loaded on {device}. params={sum(p.numel() for p in m1.parameters()):,}')
"""))


# ---------- CLIP encoders --------------------------------------------------------
CELLS.append(md("""\
## 4. Load CLIP encoders (frozen) — defensive against transformers version quirks

We bypass `get_image_features` / `get_text_features` (which break in some HF
versions) and call `vision_model` / `text_model` directly, applying the
projection only when needed.
"""))

CELLS.append(code("""\
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from PIL import Image
import numpy as np

clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
clip_proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_tok  = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
for p in clip.parameters():
    p.requires_grad = False

PROJ_DIM      = clip.config.projection_dim                # 512
VISION_HIDDEN = clip.config.vision_config.hidden_size     # 768
TEXT_HIDDEN   = clip.config.text_config.hidden_size       # 512
print(f'CLIP: proj_dim={PROJ_DIM}  vision_hidden={VISION_HIDDEN}  text_hidden={TEXT_HIDDEN}')


@torch.no_grad()
def encode_image_paths(paths, batch=32):
    \"\"\"List of image paths -> (N, 512) L2-normalized.\"\"\"
    embs, kept = [], []
    for i in range(0, len(paths), batch):
        bp = paths[i:i+batch]
        imgs, ok = [], []
        for p in bp:
            try:
                imgs.append(Image.open(p).convert('RGB')); ok.append(p)
            except Exception:
                pass
        if not imgs: continue
        inp = clip_proc(images=imgs, return_tensors='pt').to(device)
        out = clip.vision_model(pixel_values=inp['pixel_values'])
        pooled = out.pooler_output if (hasattr(out, 'pooler_output') and out.pooler_output is not None) \\
                 else out.last_hidden_state[:, 0]
        emb = clip.visual_projection(pooled) if pooled.shape[-1] == VISION_HIDDEN else pooled
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        embs.append(emb.cpu().numpy()); kept.extend(ok)
    if not embs:
        return np.zeros((0, PROJ_DIM), dtype=np.float32), []
    return np.concatenate(embs, axis=0), kept


@torch.no_grad()
def encode_texts(texts, batch=64):
    \"\"\"List of strings -> (N, 512) L2-normalized.\"\"\"
    out_arr = []
    for i in range(0, len(texts), batch):
        toks = clip_tok(texts[i:i+batch], padding=True, truncation=True,
                         return_tensors='pt').to(device)
        out = clip.text_model(**toks)
        pooled = out.pooler_output if (hasattr(out, 'pooler_output') and out.pooler_output is not None) \\
                 else out.last_hidden_state[:, 0]
        emb = clip.text_projection(pooled) if pooled.shape[-1] == TEXT_HIDDEN else pooled
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out_arr.append(emb.cpu().numpy())
    return np.concatenate(out_arr, axis=0)


# Quick sanity check
sample_emb = encode_texts(['a man walks in a park'])
print(f'Test encode_texts -> shape {sample_emb.shape}')
"""))


# ---------- M1 inference helper --------------------------------------------------
CELLS.append(md("""\
## 5. M1 inference helper — saliency from features + query
"""))

CELLS.append(code("""\
# FEAT_DIR and TEXT_FEAT_DIR were set by cell 2.5 (auto-extraction)
FEAT_DIR      = clip_dir
TEXT_FEAT_DIR = text_dir
print(f'M1 feature dir: {FEAT_DIR}  ({len(list(FEAT_DIR.glob(\"*.npz\")))} files)')


@torch.no_grad()
def m1_predict_saliency(v_feat_np, query_text):
    \"\"\"v_feat_np: (n_clips, 512) numpy. Returns (n_clips,) saliency.\"\"\"
    n_clips = min(v_feat_np.shape[0], 75)
    v_feat_np = v_feat_np[:n_clips]

    # Video + TEF
    v = torch.from_numpy(v_feat_np).float()
    v = F.normalize(v, dim=-1, eps=1e-5)
    tef_st = torch.arange(0, n_clips, 1.0) / n_clips
    tef = torch.stack([tef_st, tef_st + 1.0 / n_clips], dim=1)
    v = torch.cat([v, tef], dim=1).to(device).unsqueeze(0)        # (1, T, 514)
    v_mask = torch.ones(1, n_clips, device=device)

    # Text: encode the query through CLIP-text. M1 was trained on per-token
    # CLIP text embeddings; here we approximate with the pooled feature
    # broadcast as a single token. This matches the inference path in the
    # upstream MomentDETRPredictor for this checkpoint.
    t_emb = encode_texts([query_text])                            # (1, 512)
    t = torch.from_numpy(t_emb).float().to(device).unsqueeze(0)   # (1, 1, 512)
    t_mask = torch.ones(1, 1, device=device)

    out = m1(src_vid=v, src_vid_mask=v_mask, src_txt=t, src_txt_mask=t_mask)
    saliency = out['saliency_scores'].float().cpu().numpy()[0]    # (T,)
    return saliency
"""))


# ---------- Scoring math inline --------------------------------------------------
CELLS.append(md("""\
## 6. Scoring math — pure NumPy, inline

The three building blocks:

```
e_clip_w = sum_t softmax(s[t]/tau) * v[t]            (top-p mask)
c(q, w)  = alpha * cos(e_q, e_clip_w) + (1-alpha) * cos(e_T, e_clip_w)
C(T, V)  = max_w c(q, w)                              (single chunk for now)
```
"""))

CELLS.append(code("""\
def cosine_sim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    n = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / n)


def softmax_1d(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def saliency_weighted_pool(v_feat, saliency, tau=0.1, top_p=0.25):
    \"\"\"Pool video features by saliency. Top-p clips kept, softmax-weighted.\"\"\"
    T = v_feat.shape[0]
    k = max(1, int(np.ceil(top_p * T)))
    keep_idx = np.argpartition(-saliency, k - 1)[:k]
    masked = np.full_like(saliency, -np.inf, dtype=np.float64)
    masked[keep_idx] = saliency[keep_idx].astype(np.float64)
    w = softmax_1d(masked / tau)
    return (w[:, None] * v_feat).sum(axis=0)


def consistency_score(e_T, e_q, v_feat, saliency, alpha=0.5, tau=0.1, top_p=0.25):
    \"\"\"The full C(T, V) for a single (thumbnail, video, query).\"\"\"
    e_clip = saliency_weighted_pool(v_feat, saliency, tau=tau, top_p=top_p)
    return alpha * cosine_sim(e_q, e_clip) + (1 - alpha) * cosine_sim(e_T, e_clip)
"""))


# ---------- AUC and metrics inline ----------------------------------------------
CELLS.append(md("## 7. AUC + Cohen's d (inline, no sklearn)"))

CELLS.append(code("""\
def roc_auc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    l = np.asarray(labels, dtype=np.int64)
    pos, neg = s[l == 1], s[l == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def cohens_d(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) < 2 or len(neg) < 2: return float('nan')
    s_p = np.sqrt(((len(pos)-1)*pos.var(ddof=1) + (len(neg)-1)*neg.var(ddof=1))
                  / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / (s_p + 1e-12))
"""))


# ---------- Load eval pairs ------------------------------------------------------
CELLS.append(md("## 8. Load the 600 evaluation pairs"))

CELLS.append(code("""\
EVAL_PATH = DRIVE_ROOT / 'processed' / 'consistency_eval.jsonl'
assert EVAL_PATH.exists(), f'Missing: {EVAL_PATH}'

pairs = []
with open(EVAL_PATH, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))

print(f'Loaded {len(pairs)} eval pairs')
print(f'Positives: {sum(1 for p in pairs if p[\"label\"] == 1)}')
print(f'Negatives: {sum(1 for p in pairs if p[\"label\"] == 0)}')
print(f'\\nExample record:')
for k, v in list(pairs[0].items())[:8]:
    print(f'  {k:20s}: {str(v)[:90]}')
"""))


# ---------- Main scoring loop ----------------------------------------------------
CELLS.append(md("""\
## 9. Main loop — score every pair (R1 smoke test)

This is the long-running cell (~10 min on T4). It does:

1. Encode the thumbnail frame -> `e_T`
2. Encode the GT query -> `e_q`
3. Load the video's CLIP features from Drive
4. Run M1 to get per-clip saliency
5. Pool video features by saliency
6. Compute `C = 0.5 cos(e_q, e_clip) + 0.5 cos(e_T, e_clip)`
"""))

CELLS.append(code("""\
from tqdm.auto import tqdm

results, skipped, errors = [], 0, []

for r in tqdm(pairs):
    try:
        # 1) Embed the thumbnail frame
        e_T_arr, _ = encode_image_paths([r['frame_path']])
        if e_T_arr.shape[0] == 0:
            skipped += 1; continue
        e_T = e_T_arr[0]

        # 2) Embed the GT query
        e_q = encode_texts([r['thumb_query']])[0]

        # 3) Load video features
        feat_path = FEAT_DIR / f\"{r['video_vid']}.npz\"
        if not feat_path.exists():
            skipped += 1; continue
        v_feat = np.load(feat_path)['features']
        v_feat = v_feat / (np.linalg.norm(v_feat, axis=-1, keepdims=True) + 1e-8)

        # 4) Saliency from M1
        saliency = m1_predict_saliency(v_feat, r['thumb_query'])
        # Align lengths (defensive)
        L = min(len(saliency), v_feat.shape[0])
        v_feat   = v_feat[:L]
        saliency = saliency[:L]

        # 5) + 6) Pool + score
        C = consistency_score(e_T, e_q, v_feat, saliency,
                              alpha=0.5, tau=0.1, top_p=0.25)
        results.append({'pair_id': r['pair_id'],
                        'label':   r['label'],
                        'score':   C})
    except Exception as e:
        errors.append((r.get('pair_id'), str(e)[:120]))
        skipped += 1

print(f'\\nScored:  {len(results)}')
print(f'Skipped: {skipped}')
if errors[:3]:
    print(f'\\nFirst 3 errors:')
    for pid, msg in errors[:3]:
        print(f'  {pid}: {msg}')
"""))


# ---------- Save + headline numbers ---------------------------------------------
CELLS.append(md("## 10. Headline numbers — AUC, Cohen's d, separation"))

CELLS.append(code("""\
out_dir = DRIVE_ROOT / 'outputs' / 'predictions'
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'allinone_gt_smoke.jsonl', 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r) + '\\n')
print(f'Saved {len(results)} predictions to {out_dir}/allinone_gt_smoke.jsonl')

scores = [r['score'] for r in results]
labels = [r['label'] for r in results]
pos = [s for s, l in zip(scores, labels) if l == 1]
neg = [s for s, l in zip(scores, labels) if l == 0]

auc = roc_auc(scores, labels)
d   = cohens_d(pos, neg)

print(f'\\n{\"=\"*55}')
print(f'  R1 SMOKE TEST RESULT')
print(f'{\"=\"*55}')
print(f'  AUC          : {auc:.4f}')
print(f'  Cohen\\'s d    : {d:.4f}')
print(f'  mean(pos)    : {np.mean(pos):.4f}  (n={len(pos)})')
print(f'  mean(neg)    : {np.mean(neg):.4f}  (n={len(neg)})')
print(f'  separation   : {np.mean(pos) - np.mean(neg):+.4f}')

if auc >= 0.75:
    print(f'\\n  PASS  -- AUC {auc:.3f} >= 0.75. The pipeline works.')
    print(f'           Next step: train ClipCap (M2).')
else:
    print(f'\\n  FAIL  -- AUC {auc:.3f} < 0.75.')
    print(f'           The score formula needs debugging before training M2.')
"""))


# ---------- Histogram + threshold metrics ---------------------------------------
CELLS.append(md("## 11. Score histogram + threshold metrics"))

CELLS.append(code("""\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(pos, bins=30, alpha=0.55, label=f'positive (n={len(pos)})', color='tab:green')
ax.hist(neg, bins=30, alpha=0.55, label=f'negative (n={len(neg)})', color='tab:red')
ax.axvline(np.median(scores), linestyle='--', color='black', alpha=0.5,
           label=f'median = {np.median(scores):.3f}')
ax.set_xlabel('C(T, V)')
ax.set_ylabel('count')
ax.set_title(f'R1 Smoke Test (GT queries)  AUC = {auc:.3f}  d = {d:.2f}')
ax.legend()
plt.tight_layout()

fig_dir = DRIVE_ROOT / 'outputs' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / 'allinone_gt_smoke.png', dpi=140, bbox_inches='tight')
plt.show()
print(f'Saved figure to {fig_dir}/allinone_gt_smoke.png')

# Confusion at the median threshold
def confusion_at(scores, labels, thr):
    s = np.asarray(scores); l = np.asarray(labels)
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (l == 1)).sum())
    fp = int(((pred == 1) & (l == 0)).sum())
    tn = int(((pred == 0) & (l == 0)).sum())
    fn = int(((pred == 0) & (l == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'acc': (tp + tn) / len(s),
            'prec': prec, 'rec': rec,
            'f1': 2 * prec * rec / max(1e-9, prec + rec)}

print(f'\\nConfusion at threshold = median ({np.median(scores):.3f}):')
for k, v in confusion_at(scores, labels, float(np.median(scores))).items():
    print(f'  {k:5s}: {v:.4f}' if isinstance(v, float) else f'  {k:5s}: {v}')
"""))


# ---------- Wrap-up --------------------------------------------------------------
CELLS.append(md("""\
---

## You're done with the smoke test

If AUC >= 0.75 → the consistency pipeline works with ground-truth queries. The
next step is training **ClipCap** (M2) so the pipeline can generate its own
queries from the thumbnail.

If you want to extend this notebook to also train M2 + run the full ablation, ask
for "Section 12+: ClipCap training and ablation" and I'll append those cells.

For now, this notebook produced:

- `outputs/predictions/allinone_gt_smoke.jsonl` — per-pair scores
- `outputs/figures/allinone_gt_smoke.png` — score distribution histogram

Both are saved on your Drive (persistent across Colab sessions).
"""))


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  Cells: {len(CELLS)} "
          f"({sum(1 for c in CELLS if c['cell_type']=='code')} code, "
          f"{sum(1 for c in CELLS if c['cell_type']=='markdown')} md)")


if __name__ == "__main__":
    main()
