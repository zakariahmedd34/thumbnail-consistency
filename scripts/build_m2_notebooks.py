"""
Build three notebook-style entrypoints that mirror the CLI commands:

  notebooks/m2_score_single.ipynb     — score one (thumbnail, video) pair, pick any M2
  notebooks/m2_ablate.ipynb           — run multiple M2s on the same eval set
  notebooks/compare_m2_generators.ipynb — qualitative side-by-side viewer

Each notebook has the same bootstrap pattern as sanity_consistency.ipynb:
  cell 0 = bootstrap (mount Drive, extract features, sys.path)
  cell 1+ = experiment cells (run-all or step-through)
"""
from __future__ import annotations
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": text.splitlines(keepends=True)}


# ---------- Shared bootstrap cell (mirror of inject_colab_bootstrap.py) -----

BOOTSTRAP = """\
# [COLAB_BOOTSTRAP_v2_modular]
# Each Colab session:
#   1. Mount Drive             (Drive has the repo + the tar.gz of M1 features)
#   2. cd into the repo        (so relative paths to src/, configs/, etc. work)
#   3. Symlink Drive data dirs (processed/, qvh_frames/, etc.) into the repo
#   4. EXTRACT M1 features tar.gz from Drive ->  LOCAL SSD  (/content/features_local)
#      Reason: 19,000 small .npz files corrupt on Drive's FUSE; LOCAL SSD is fast.
#   5. Clone upstream moment_detr code (for build_model imports)
#
# Drive is ONLY used as the SOURCE of the tar.gz. Features ALWAYS live on
# /content/ (LOCAL SSD) after extraction. Same pattern as
# run_pipeline_allinone.ipynb cell 2.5 and sanity_consistency.ipynb cell 0.

import os, sys, subprocess, time, tarfile, shutil
from pathlib import Path

# ----- 1. Mount Drive ----------------------------------------------------
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

# ----- 2. Find repo + cd ------------------------------------------------
DRIVE_ROOT = Path('/content/drive/MyDrive/thumbnail_project')
REPO_DIR   = DRIVE_ROOT / 'thumbnail-consistency'
assert REPO_DIR.exists(), f"Upload your repo folder to {REPO_DIR} first."

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# ----- 3. Symlink data dirs ---------------------------------------------
for d in ['processed', 'outputs', 'qvh_videos', 'qvh_frames', 'qvh_thumbnails',
          'qvh_frames_within_neg', 'qvh_frames_multi', 'checkpoints']:
    src, dst = DRIVE_ROOT / d, REPO_DIR / d
    if src.exists() and not dst.exists() and not dst.is_symlink():
        try: os.symlink(src, dst)
        except OSError: pass

# ----- 4. Extract M1 features from Drive .tar.gz -> LOCAL SSD ----------
# Source on Drive (the 7.5 GB archive):
ARCHIVE_ON_DRIVE = DRIVE_ROOT / 'moment_detr_features.tar.gz'
# Destination on LOCAL SSD (NOT on Drive — Drive can't hold 19k small files):
EXTRACTED_LOCAL  = Path('/content/features_local')

def _find_feature_dirs(root):
    \"\"\"Look inside `root` for clip_features/ and clip_text_features/ dirs
    that contain .npz files. Returns (clip_dir, text_dir) paths or (None, None).\"\"\"
    clip = text = None
    for r, dirs, _ in os.walk(root):
        for d in dirs:
            p = Path(r) / d
            if d == 'clip_features' and len(list(p.glob('*.npz'))) > 1000:
                clip = p
            if d == 'clip_text_features' and len(list(p.glob('*.npz'))) > 1000:
                text = p
    return clip, text

# Check ONLY local SSD (NOT Drive) for an already-extracted copy from earlier
# in THIS Colab session. If you restart the runtime, /content is wiped and
# this check returns (None, None), triggering a fresh extraction.
CLIP_FEAT_DIR, TEXT_FEAT_DIR = (None, None)
if EXTRACTED_LOCAL.exists():
    CLIP_FEAT_DIR, TEXT_FEAT_DIR = _find_feature_dirs(EXTRACTED_LOCAL)

if CLIP_FEAT_DIR and TEXT_FEAT_DIR:
    print(f"[features] Found local extraction from earlier this session (skip).")
    print(f"           {EXTRACTED_LOCAL}")
else:
    assert ARCHIVE_ON_DRIVE.exists(), (
        f"Source archive missing on Drive: {ARCHIVE_ON_DRIVE}\\n"
        f"This is the only file the bootstrap reads from Drive."
    )
    # Wipe any stale partial extraction and start fresh
    if EXTRACTED_LOCAL.exists():
        shutil.rmtree(EXTRACTED_LOCAL)
    EXTRACTED_LOCAL.mkdir(parents=True)
    sz_gb = ARCHIVE_ON_DRIVE.stat().st_size / 1e9
    print(f"[features] Extracting {sz_gb:.2f} GB:")
    print(f"           FROM (Drive)     : {ARCHIVE_ON_DRIVE}")
    print(f"           TO   (local SSD) : {EXTRACTED_LOCAL}")
    t0 = time.time()
    with tarfile.open(ARCHIVE_ON_DRIVE) as tf:
        tf.extractall(EXTRACTED_LOCAL)
    print(f"[features] Done in {time.time()-t0:.0f}s.")
    CLIP_FEAT_DIR, TEXT_FEAT_DIR = _find_feature_dirs(EXTRACTED_LOCAL)

assert CLIP_FEAT_DIR and TEXT_FEAT_DIR, (
    f"Feature dirs not found after extraction. Check {EXTRACTED_LOCAL}"
)

# ----- 5. Clone upstream Moment-DETR (for build_model imports) ---------
MD_DIR = Path('/content/moment_detr_upstream')
if not MD_DIR.exists():
    subprocess.run(['git', 'clone', 'https://github.com/jayleicn/moment_detr.git',
                    str(MD_DIR)], check=True)
if str(MD_DIR) not in sys.path:
    sys.path.insert(0, str(MD_DIR))

# ----- 6. Verification summary -----------------------------------------
import torch
print()
print(f"Repo            : {REPO_DIR}")
print(f"Device          : {'cuda' if torch.cuda.is_available() else 'cpu (WARN: slow)'}")
print(f"Drive tar.gz    : {ARCHIVE_ON_DRIVE}  ({sz_gb:.2f} GB)" if 'sz_gb' in dir() else "")
print(f"clip_features   : {CLIP_FEAT_DIR}  ({len(list(CLIP_FEAT_DIR.glob('*.npz'))):,} files)")
print(f"clip_text_feats : {TEXT_FEAT_DIR}  ({len(list(TEXT_FEAT_DIR.glob('*.npz'))):,} files)")
print(f"src/ importable : {(REPO_DIR / 'src' / '__init__.py').exists()}")
"""


# ---------- Notebook 1: single-pair scoring ---------------------------------

SINGLE_CELLS = [
    md("""\
# M2 Single-Pair Scoring

Notebook equivalent of `python -m src.cli.score`. Choose any M2 generator
from the registry, give it a thumbnail + video vid, get a consistency score.

Use this for: quick spot checks, demos, debugging.
"""),
    code(BOOTSTRAP),
    md("""\
## 1. Install M2 dependencies (only if you'll use a heavy generator)

- Lightweight M2s (`gt`, `random`, `clipcap`) need **nothing** extra.
- VLMs (`qwen2vl_*`, `llava16_aspect`, `aspect_rewriter`) need this cell.

Skip the next cell if you only plan to run lightweight M2s.
"""),
    code("""\
# Run this if you're using qwen2vl_*, llava16_aspect, or aspect_rewriter.
# Otherwise comment it out — it costs ~1 min on first run.
import sys, subprocess

USE_VLM = True       # ★ set False to skip VLM deps and save 1 min

if USE_VLM:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'transformers>=4.45', 'accelerate>=0.30', 'bitsandbytes>=0.43',
                    'qwen-vl-utils', 'sentencepiece'], check=True)
    print('VLM deps installed.')
else:
    print('Skipped VLM deps (USE_VLM=False).')
"""),
    md("## 2. Pick an M2 generator"),
    code("""\
from src.m2_generators import build_m2, available
print("Available M2 generators:")
for name in available():
    print(f"  - {name}")

# === EDIT THIS LINE to try a different M2 ===
M2_NAME = 'gt'   # one of: gt, random, clipcap, qwen2vl_simple, qwen2vl_aspect, llava16_aspect

# Per-generator kwargs (some need a lookup or ckpt)
M2_KWARGS = {
    'gt'             : {'lookup_jsonl': 'processed/consistency_eval.jsonl'},
    'random'         : {'pool_jsonl'  : 'processed/consistency_eval.jsonl', 'seed': 2026},
    'clipcap'        : {'ckpt_path'   : 'checkpoints/clipcap_best.pt'},
    'qwen2vl_simple' : {'model_id': 'Qwen/Qwen2-VL-7B-Instruct',          'load_in_4bit': True},
    'qwen2vl_aspect' : {'model_id': 'Qwen/Qwen2-VL-7B-Instruct',          'load_in_4bit': True},
    'llava16_aspect' : {'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',  'load_in_4bit': True},
}

print(f'\\nBuilding M2: {M2_NAME}')
m2 = build_m2(M2_NAME, **M2_KWARGS[M2_NAME])
print(f'Ready: {m2}')
"""),
    md("## 2. Build M1 + the consistency scorer (frozen across experiments)"),
    code("""\
import yaml
from src.models.moment_detr import MomentDETRFeaturePredictor
from src.scoring.consistency import ConsistencyScorer, ScoringConfig

cfg_yaml = yaml.safe_load(open('configs/consistency.yaml'))

m1 = MomentDETRFeaturePredictor(
    ckpt_path=cfg_yaml['paths']['m1_ckpt'],
    repo_root=str(REPO_DIR),
)
scoring_cfg = ScoringConfig(
    alpha=cfg_yaml['alpha'], tau=cfg_yaml['tau'], top_p=cfg_yaml['top_p'],
    chunk_window=cfg_yaml['chunk_window'], chunk_stride=cfg_yaml['chunk_stride'],
    damp_with_fg=cfg_yaml['damp_with_fg'],
)
# Override feature_dir to the LOCAL extracted path (not the YAML default)
scorer = ConsistencyScorer(m1, str(CLIP_FEAT_DIR), scoring_cfg)
print('Scorer ready.')
"""),
    md("""\
## 3. Pick one (thumbnail, video) pair and score it

We pull a random pair from `consistency_eval.jsonl` so the setup is reproducible.
Edit `pair_idx` or set `THUMB_PATH` / `VIDEO_VID` / `QID` manually below.
"""),
    code("""\
import json, random
random.seed(2026)

pairs = [json.loads(l) for l in open('processed/consistency_eval.jsonl', encoding='utf-8') if l.strip()]
pair = random.choice([p for p in pairs if p['label'] == 1])    # take a positive

THUMB_PATH = pair['frame_path']
VIDEO_VID  = pair['video_vid']
QID        = pair['thumb_qid']    # needed only for the 'gt' generator
GT_QUERY   = pair['thumb_query']

print(f'Thumbnail : {THUMB_PATH}')
print(f'Video     : {VIDEO_VID}')
print(f'GT query  : {GT_QUERY}')
"""),
    md("## 4. Generate queries from M2 and compute the score"),
    code("""\
# Step A: M2 generates K queries from the thumbnail.
#         (For 'gt' we pass qid to look up the human-authored query.)
gen_kwargs = {'qid': QID} if M2_NAME == 'gt' else {}
m2_out = m2.generate_with_metadata(THUMB_PATH, k=3, **gen_kwargs)

print('--- M2 output ---')
print(f'm2_name      : {m2_out[\"m2_name\"]}')
print(f'elapsed_ms   : {m2_out[\"elapsed_ms\"]:.1f}')
for i, q in enumerate(m2_out['queries']):
    print(f'  query[{i}]: {q}')
"""),
    code("""\
# Step B: Plug the queries into the scorer.
result = scorer.score(
    thumbnail_path=THUMB_PATH,
    vid=VIDEO_VID,
    queries=m2_out['queries'],
)

print('--- Consistency score ---')
print(f'C(T, V)      : {result[\"C\"]:+.4f}')
print(f'n_chunks     : {result[\"n_chunks\"]}')
print(f'max_fg_prob  : {result[\"max_fg_prob\"]:.3f}')
print(f'per_query    : {[f\"{c:+.3f}\" for c in result[\"per_query\"]]}')
"""),
    md("## 5. Bundle a result JSON (matches the schema in INTEGRATION_PLAN.md)"),
    code("""\
import json

payload = {
    'schema_version': '1.0',
    'consistency_score': float(result['C']),
    'overall_score'    : float(result['C']),
    'm2_generator'     : m2_out['m2_name'],
    'generated_queries': m2_out['queries'],
    'per_query_scores' : [
        {'query': q, 'score': float(c)}
        for q, c in zip(m2_out['queries'], result['per_query'])
    ],
    'temporal'         : {'max_fg_prob': float(result['max_fg_prob']),
                          'n_chunks'   : int(result['n_chunks'])},
    'config'           : {'alpha': scoring_cfg.alpha, 'tau': scoring_cfg.tau,
                          'top_p': scoring_cfg.top_p, 'k_queries': 3},
    'thumbnail'        : {'path': THUMB_PATH},
    'video'            : {'vid': VIDEO_VID},
    'ground_truth'     : {'query': GT_QUERY, 'qid': int(QID)},
    'explanation'      : {
        'm2_metadata' : m2_out.get('metadata', {}),
        'elapsed_ms'  : {'m2_generation': m2_out['elapsed_ms']},
    },
}

print(json.dumps(payload, indent=2, default=str))
"""),
]


# ---------- Notebook 2: multi-M2 ablation -----------------------------------

ABLATE_CELLS = [
    md("""\
# M2 Ablation — Run Multiple Generators on the Same Eval Set

Notebook equivalent of `python -m src.cli.ablate`. Runs the consistency
scorer with each chosen M2 against the same `consistency_eval*.jsonl`,
saves per-M2 score JSONLs, prints AUC + Cohen's d, and writes the ablation
table to `outputs/figures/ablation_table.csv`.

This is the **headline result table** for the writeup.
"""),
    code(BOOTSTRAP),
    md("""\
## 1. Install M2 dependencies (only if your M2_LIST contains a VLM)

Skip this cell if `M2_LIST` only contains `gt`, `random`, and/or `clipcap`.
"""),
    code("""\
import sys, subprocess
USE_VLM = True   # ★ set False to skip VLM deps
if USE_VLM:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'transformers>=4.45', 'accelerate>=0.30', 'bitsandbytes>=0.43',
                    'qwen-vl-utils', 'sentencepiece'], check=True)
    print('VLM deps installed.')
else:
    print('Skipped VLM deps.')
"""),
    md("## 2. Configure the ablation"),
    code("""\
import yaml

# ★ EDIT THESE TWO LINES to change the experiment
EVAL_PATH    = 'processed/consistency_eval.jsonl'         # also: consistency_eval_hard / _query_mismatch
M2_LIST      = ['gt', 'random']                            # add 'clipcap', 'qwen2vl_simple', etc.
K_QUERIES    = 3

# Per-generator kwargs
M2_KWARGS = {
    'gt'             : {'lookup_jsonl': EVAL_PATH},
    'random'         : {'pool_jsonl'  : EVAL_PATH, 'seed': 2026},
    'clipcap'        : {'ckpt_path'   : 'checkpoints/clipcap_best.pt'},
    'qwen2vl_simple' : {'model_id': 'Qwen/Qwen2-VL-7B-Instruct',          'load_in_4bit': True},
    'qwen2vl_aspect' : {'model_id': 'Qwen/Qwen2-VL-7B-Instruct',          'load_in_4bit': True},
    'llava16_aspect' : {'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',  'load_in_4bit': True},
}

cfg_yaml = yaml.safe_load(open('configs/consistency.yaml'))
print(f'Eval set      : {EVAL_PATH}')
print(f'M2 generators : {M2_LIST}')
print(f'K queries     : {K_QUERIES}')
print(f'alpha / tau / top_p : {cfg_yaml[\"alpha\"]} / {cfg_yaml[\"tau\"]} / {cfg_yaml[\"top_p\"]}')
"""),
    md("## 2. Build M1 + scorer (once, shared across all M2 runs)"),
    code("""\
from src.models.moment_detr import MomentDETRFeaturePredictor
from src.scoring.consistency import ConsistencyScorer, ScoringConfig

m1 = MomentDETRFeaturePredictor(
    ckpt_path=cfg_yaml['paths']['m1_ckpt'],
    repo_root=str(REPO_DIR),
)
scoring_cfg = ScoringConfig(
    alpha=cfg_yaml['alpha'], tau=cfg_yaml['tau'], top_p=cfg_yaml['top_p'],
    chunk_window=cfg_yaml['chunk_window'], chunk_stride=cfg_yaml['chunk_stride'],
)
scorer = ConsistencyScorer(m1, str(CLIP_FEAT_DIR), scoring_cfg)
print('M1 + scorer ready.')
"""),
    md("## 3. Load the eval pairs"),
    code("""\
import json
pairs = [json.loads(l) for l in open(EVAL_PATH, encoding='utf-8') if l.strip()]
print(f'Loaded {len(pairs)} pairs from {EVAL_PATH}')
print(f'  positives: {sum(1 for p in pairs if p[\"label\"] == 1)}')
print(f'  negatives: {sum(1 for p in pairs if p[\"label\"] == 0)}')
"""),
    md("""\
## 4. Run the ablation — one M2 at a time

Each M2 takes one of these paths:
- **lightweight** (`gt`, `random`, `clipcap`): seconds to load, < 1s/pair
- **heavy** (Qwen, LLaVA): 1-2 min to load, 3-7 sec/pair

Heavy M2s should be run one at a time in this notebook (or restart the runtime
between them) to avoid OOM. Use the GPU memory cleanup cell below if needed.
"""),
    code("""\
from tqdm.auto import tqdm
from src.m2_generators import build_m2
from src.eval.metrics import summarize_separation
from pathlib import Path
import gc, torch

OUT_DIR = REPO_DIR / 'outputs' / 'predictions'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ablation_rows = []

for m2_name in M2_LIST:
    print(f'\\n========== {m2_name} ==========')
    m2 = build_m2(m2_name, **M2_KWARGS.get(m2_name, {}))

    results = []
    for r in tqdm(pairs, desc=m2_name):
        # gt needs qid; others ignore extra kwargs
        gen_kwargs = {'qid': r['thumb_qid']} if m2_name == 'gt' else {}
        try:
            qs = m2.generate(r['frame_path'], k=K_QUERIES, **gen_kwargs)
            out = scorer.score(
                thumbnail_path=r['frame_path'],
                vid=r['video_vid'],
                queries=qs,
            )
            results.append({'pair_id': r['pair_id'], 'label': r['label'],
                            'score': float(out['C']), 'queries': qs})
        except Exception as e:
            print(f'  FAIL pair {r[\"pair_id\"]}: {e}')

    # Save per-M2 predictions
    out_path = OUT_DIR / f'scores_{m2_name}.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + '\\n')
    print(f'  saved -> {out_path} ({len(results)} pairs)')

    # Summary
    scores = [x['score'] for x in results]
    labels = [x['label'] for x in results]
    stats  = summarize_separation(scores, labels)
    stats['m2']        = m2_name
    stats['n_scored']  = len(results)
    ablation_rows.append(stats)
    print(f"  AUC = {stats['auc']:.4f}   Cohen's d = {stats['cohens_d']:.3f}")

    # Free GPU before loading the next M2
    del m2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""),
    md("## 5. Build the ablation table"),
    code("""\
import pandas as pd

df = pd.DataFrame(ablation_rows)
keys = ['m2', 'n_scored', 'n_pos', 'n_neg',
        'mean_pos', 'mean_neg', 'auc', 'cohens_d', 'welch_t', 'welch_p']
df = df[keys].set_index('m2').round(4)

fig_dir = REPO_DIR / 'outputs' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
csv_path = fig_dir / f'ablation_table_{Path(EVAL_PATH).stem}.csv'
df.to_csv(csv_path)
print(f'Saved {csv_path}')
df
"""),
    md("## 6. Histogram per M2 (saved + displayed)"),
    code("""\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(M2_LIST), figsize=(4.5 * len(M2_LIST), 4), sharey=True)
if len(M2_LIST) == 1:
    axes = [axes]

for ax, m2_name in zip(axes, M2_LIST):
    rows = [json.loads(l) for l in open(OUT_DIR / f'scores_{m2_name}.jsonl', encoding='utf-8') if l.strip()]
    pos = [r['score'] for r in rows if r['label'] == 1]
    neg = [r['score'] for r in rows if r['label'] == 0]
    auc = df.loc[m2_name, 'auc']
    ax.hist(pos, bins=25, alpha=0.55, label=f'pos (n={len(pos)})', color='tab:green')
    ax.hist(neg, bins=25, alpha=0.55, label=f'neg (n={len(neg)})', color='tab:red')
    ax.set_title(f'{m2_name}\\nAUC = {auc:.3f}', fontsize=10)
    ax.set_xlabel('C(T, V)')
    ax.legend(fontsize=8)
plt.tight_layout()
out_png = fig_dir / f'ablation_{Path(EVAL_PATH).stem}.png'
plt.savefig(out_png, dpi=140, bbox_inches='tight')
plt.show()
print(f'Saved {out_png}')
"""),
]


# ---------- Notebook 3: qualitative side-by-side -----------------------------

COMPARE_CELLS = [
    md("""\
# M2 Generators — Qualitative Side-by-Side Comparison

Pick N pairs from the eval set. For each, show:
- the thumbnail image
- 3-4 frames sampled from the video
- the ground-truth QVH query
- queries produced by each M2 you're comparing
- the resulting consistency score from each M2

Use this for: paper figures, eyeballing failure modes, M2 generation
inspection.
"""),
    code(BOOTSTRAP),
    md("""\
## 1. Install M2 dependencies (only if you'll use a VLM)
"""),
    code("""\
import sys, subprocess
USE_VLM = True   # ★ set False to skip if comparing only lightweight M2s
if USE_VLM:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'transformers>=4.45', 'accelerate>=0.30', 'bitsandbytes>=0.43',
                    'qwen-vl-utils', 'sentencepiece'], check=True)
    print('VLM deps installed.')
else:
    print('Skipped VLM deps.')
"""),
    md("## 2. Choose M2s to compare and an eval set"),
    code("""\
M2_LIST   = ['gt', 'random']     # add 'clipcap' / 'qwen2vl_simple' etc.
EVAL_PATH = 'processed/consistency_eval.jsonl'
N_SAMPLES = 5

M2_KWARGS = {
    'gt'             : {'lookup_jsonl': EVAL_PATH},
    'random'         : {'pool_jsonl'  : EVAL_PATH, 'seed': 2026},
    'clipcap'        : {'ckpt_path'   : 'checkpoints/clipcap_best.pt'},
    'qwen2vl_simple' : {'model_id': 'Qwen/Qwen2-VL-7B-Instruct',          'load_in_4bit': True},
    'qwen2vl_aspect' : {'model_id': 'Qwen/Qwen2-VL-7B-Instruct',          'load_in_4bit': True},
    'llava16_aspect' : {'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',  'load_in_4bit': True},
}
print(f'Comparing {M2_LIST} on {N_SAMPLES} pairs from {EVAL_PATH}')
"""),
    md("## 2. Build scorer once + all M2s up front"),
    code("""\
import yaml, gc, torch
from src.m2_generators import build_m2
from src.models.moment_detr import MomentDETRFeaturePredictor
from src.scoring.consistency import ConsistencyScorer, ScoringConfig

cfg_yaml = yaml.safe_load(open('configs/consistency.yaml'))
m1 = MomentDETRFeaturePredictor(
    ckpt_path=cfg_yaml['paths']['m1_ckpt'], repo_root=str(REPO_DIR),
)
scoring_cfg = ScoringConfig(alpha=cfg_yaml['alpha'], tau=cfg_yaml['tau'], top_p=cfg_yaml['top_p'])
scorer = ConsistencyScorer(m1, str(CLIP_FEAT_DIR), scoring_cfg)

# NOTE: heavy M2s (Qwen, LLaVA) loaded together can OOM. If so:
#   1. comment out the heavy ones in M2_LIST
#   2. restart runtime
#   3. load one at a time
m2_instances = {}
for name in M2_LIST:
    print(f'  building {name}...')
    m2_instances[name] = build_m2(name, **M2_KWARGS.get(name, {}))
print('Done.')
"""),
    md("## 3. Sample pairs and run all M2s on them"),
    code("""\
import json, random, time
random.seed(7)

pairs = [json.loads(l) for l in open(EVAL_PATH, encoding='utf-8') if l.strip()]
positives = [p for p in pairs if p['label'] == 1]
sampled = random.sample(positives, min(N_SAMPLES, len(positives)))
print(f'Sampled {len(sampled)} positive pairs')

table = []   # one row per pair, with all M2 outputs
for p in sampled:
    row = {
        'pair_id'   : p['pair_id'],
        'thumb_vid' : p['thumb_vid'],
        'video_vid' : p['video_vid'],
        'frame_path': p['frame_path'],
        'gt_query'  : p['thumb_query'],
        'm2_runs'   : {},
    }
    for name, m2 in m2_instances.items():
        gen_kwargs = {'qid': p['thumb_qid']} if name == 'gt' else {}
        t0 = time.time()
        qs = m2.generate(p['frame_path'], k=3, **gen_kwargs)
        gen_ms = (time.time() - t0) * 1000
        out = scorer.score(thumbnail_path=p['frame_path'],
                            vid=p['video_vid'], queries=qs)
        row['m2_runs'][name] = {
            'queries': qs, 'score': float(out['C']), 'gen_ms': gen_ms,
        }
    table.append(row)
print('Scored all pairs.')
"""),
    md("## 4. Render side-by-side panels"),
    code("""\
import matplotlib.pyplot as plt
import cv2, textwrap
from PIL import Image

VIDEOS_DIR = REPO_DIR / 'qvh_videos'

def sample_frames(video_path, n=4):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = []
    for i in range(n):
        target = int(total * (i + 0.5) / n) if total else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, fr = cap.read()
        if ret:
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


for row in table:
    n_vid_frames = 4
    n_cols = 1 + n_vid_frames
    fig, axes = plt.subplots(1, n_cols, figsize=(3.3 * n_cols, 3.3))

    # Thumb
    try:
        axes[0].imshow(Image.open(row['frame_path']).convert('RGB'))
        axes[0].set_title(f\"THUMB\\n{row['thumb_vid'][:22]}\", fontsize=9)
    except Exception:
        axes[0].text(0.5, 0.5, 'thumb missing', ha='center')
    axes[0].axis('off')

    # Video frames
    vp = VIDEOS_DIR / f\"{row['video_vid']}.mp4\"
    vf = sample_frames(vp, n_vid_frames) if vp.exists() else []
    for j in range(n_vid_frames):
        ax = axes[1 + j]
        if j < len(vf):
            ax.imshow(vf[j])
            ax.set_title(f'VIDEO @ ~{int(100*(j+0.5)/n_vid_frames)}%', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center')
        ax.axis('off')

    # Title with all M2 outputs
    lines = [f\"pair_id={row['pair_id']}    GT: {row['gt_query']}\"]
    for name, info in row['m2_runs'].items():
        lines.append(f\"\\n[{name}]  score={info['score']:+.3f}  ({info['gen_ms']:.0f} ms)\")
        for i, q in enumerate(info['queries']):
            lines.append(f\"    q{i}: {textwrap.shorten(q, 90)}\")
    fig.suptitle(\"\\n\".join(lines), fontsize=8.5, y=1.0 + 0.08*len(row['m2_runs']))
    plt.tight_layout()
    plt.show()
"""),
    md("## 5. Optional — save the comparison as JSON for the report"),
    code("""\
out_path = REPO_DIR / 'outputs' / 'predictions' / 'qual_comparison.jsonl'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    for row in table:
        f.write(json.dumps(row, ensure_ascii=False) + '\\n')
print(f'Saved {out_path}')
"""),
]


# ---------- write ----------------------------------------------------------

def write_nb(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md   = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"Wrote {path}  ({len(cells)} cells, {n_code} code, {n_md} md)")


def main():
    write_nb("m2_score_single.ipynb",       SINGLE_CELLS)
    write_nb("m2_ablate.ipynb",             ABLATE_CELLS)
    write_nb("compare_m2_generators.ipynb", COMPARE_CELLS)


if __name__ == "__main__":
    main()
