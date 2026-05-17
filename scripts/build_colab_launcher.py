"""
Build notebooks/colab_launcher.ipynb -- a single Colab notebook that handles:

  - Drive mount
  - Repo clone (public OR private via a Personal Access Token)
  - Symlinks for Drive data into the cloned repo
  - Manifest check (which data files are present)
  - Direct-open links to all downstream notebooks (training, eval, smoke test)

This is the only notebook a teammate needs to open in Colab to get going.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "colab_launcher.ipynb"

CELLS = []


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


# ---------- Title ----------------------------------------------------------------
CELLS.append(md("""\
# Colab Launcher — Thumbnail Consistency Pipeline

**This is the one notebook to open first.** It sets up Colab, clones the repo,
links to your data on Drive, and tells you what to run next.

Run all cells top-to-bottom **once per Colab session**. After that, open any of the
downstream notebooks linked at the bottom.

| What this notebook does | What it does NOT do |
|---|---|
| Mount Google Drive | Train any model |
| Clone the GitHub repo (handles private repos via PAT) | Run the data pipeline |
| Symlink Drive data folders into the repo | Run evaluation |
| Check that all required files exist | Make coffee |

If you only have one Colab tab open, make it this one.
"""))


# ---------- Step 1: Drive mount --------------------------------------------------
CELLS.append(md("""\
---
## Step 1 — Mount Google Drive

We need access to `/content/drive/MyDrive/thumbnail_project/` where the data lives
(videos, frames, processed JSONLs, Moment-DETR features).
"""))

CELLS.append(code("""\
import os, sys
from pathlib import Path

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print('Drive already mounted.')

DRIVE_ROOT = '/content/drive/MyDrive/thumbnail_project'
assert os.path.exists(DRIVE_ROOT), (
    f'DRIVE_ROOT not found: {DRIVE_ROOT}\\n'
    f'Make sure the thumbnail_project folder exists at the top of your MyDrive.'
)
print(f'OK: data root = {DRIVE_ROOT}')
"""))


# ---------- Step 2: Clone repo (handles private) ---------------------------------
CELLS.append(md("""\
---
## Step 2 — Clone the GitHub repo

**If the repo is public:** this cell runs without any input.

**If the repo is private:** the cell will prompt you for a Personal Access Token (PAT).
- Generate one at: https://github.com/settings/tokens
- Required scope: `repo` (read access)
- The token is read via `getpass` (not echoed) and used only for this clone.

The repo is cloned to `/content/repo` (fast local SSD), not Drive. Code runs from
there; data stays on Drive via symlinks (next step).
"""))

CELLS.append(code("""\
import subprocess
from getpass import getpass

REPO_USER = 'zakariahmedd34'
REPO_NAME = 'thumbnail-consistency'
BRANCH    = 'feature/v1-pipeline'     # <-- change if your branch is named differently
REPO_DIR  = '/content/repo'

PUBLIC_URL = f'https://github.com/{REPO_USER}/{REPO_NAME}.git'


def _is_public(url):
    \"\"\"True if the remote responds to anonymous ls-remote.\"\"\"
    p = subprocess.run(['git', 'ls-remote', url],
                       capture_output=True, text=True, timeout=15)
    return p.returncode == 0


def _clone_or_update(url, branch, dest):
    if os.path.exists(dest):
        print(f'Repo already at {dest} -- fetching latest...')
        subprocess.run(['git', '-C', dest, 'remote', 'set-url', 'origin', url],
                       check=True)
        subprocess.run(['git', '-C', dest, 'fetch', 'origin'], check=True)
        # Try to switch to requested branch (create local tracking branch if needed)
        r = subprocess.run(['git', '-C', dest, 'checkout', branch],
                           capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run(['git', '-C', dest, 'checkout',
                           '-b', branch, f'origin/{branch}'], check=True)
        subprocess.run(['git', '-C', dest, 'pull', '--ff-only'], check=True)
    else:
        subprocess.run(['git', 'clone', '--branch', branch, url, dest], check=True)


if _is_public(PUBLIC_URL):
    print('Repo is PUBLIC -- cloning anonymously.')
    _clone_or_update(PUBLIC_URL, BRANCH, REPO_DIR)
else:
    print('Repo is PRIVATE -- a GitHub Personal Access Token (PAT) is needed.')
    print('Generate one at: https://github.com/settings/tokens   (scope: repo)')
    token = getpass('Paste your PAT (hidden): ').strip()
    auth_url = f'https://{token}@github.com/{REPO_USER}/{REPO_NAME}.git'
    _clone_or_update(auth_url, BRANCH, REPO_DIR)
    # IMPORTANT: rewrite origin to anonymous URL so the token isn't stored on disk
    subprocess.run(['git', '-C', REPO_DIR, 'remote', 'set-url', 'origin', PUBLIC_URL],
                   check=True)
    del token

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

current_branch = subprocess.run(
    ['git', '-C', REPO_DIR, 'rev-parse', '--abbrev-ref', 'HEAD'],
    capture_output=True, text=True).stdout.strip()
last_commit = subprocess.run(
    ['git', '-C', REPO_DIR, 'log', '-1', '--oneline'],
    capture_output=True, text=True).stdout.strip()
print(f'\\nOK: repo at {REPO_DIR}')
print(f'    branch     : {current_branch}')
print(f'    last commit: {last_commit}')
"""))


# ---------- Step 3: Symlinks -----------------------------------------------------
CELLS.append(md("""\
---
## Step 3 — Symlink Drive data into the repo

Code runs from `/content/repo` (fast). Data stays on Drive (persistent).
Symlinks make Drive folders *appear* inside the repo so relative paths in configs
work without any edits.

Symlinks created:

| Inside repo | Points to |
|---|---|
| `processed/` | `/content/drive/MyDrive/thumbnail_project/processed/` |
| `qvh_videos/` | `.../qvh_videos/` |
| `qvh_frames/` | `.../qvh_frames/` |
| `qvh_thumbnails/` | `.../qvh_thumbnails/` |
| `outputs/` | `.../outputs/` (so checkpoints survive runtime restarts) |
| `Moment_DETR/moment_detr/features/` | `.../moment_detr/features/` |
"""))

CELLS.append(code("""\
# Symlink data folders into the repo so relative paths in configs work.
LINKS = {
    'processed'        : f'{DRIVE_ROOT}/processed',
    'qvh_videos'       : f'{DRIVE_ROOT}/qvh_videos',
    'qvh_frames'       : f'{DRIVE_ROOT}/qvh_frames',
    'qvh_thumbnails'   : f'{DRIVE_ROOT}/qvh_thumbnails',
}

for name, target in LINKS.items():
    if not os.path.exists(target):
        print(f'  SKIP {name}  (target missing on Drive: {target})')
        continue
    if os.path.islink(name) or os.path.exists(name):
        existing = os.readlink(name) if os.path.islink(name) else '(not a symlink)'
        print(f'  OK   {name}  -> already linked ({existing})')
        continue
    os.symlink(target, name)
    print(f'  NEW  {name}  -> {target}')

# Moment-DETR features (nested path)
os.makedirs('Moment_DETR/moment_detr', exist_ok=True)
md_target = f'{DRIVE_ROOT}/moment_detr/features'
md_link   = 'Moment_DETR/moment_detr/features'
if os.path.exists(md_target):
    if not os.path.exists(md_link):
        os.symlink(md_target, md_link)
        print(f'  NEW  {md_link}  -> {md_target}')
    else:
        print(f'  OK   {md_link}  (already linked)')
else:
    print(f'  SKIP {md_link}  (need to unpack moment_detr_features.tar.gz first)')

# outputs/ -> Drive so checkpoints survive
out_target = f'{DRIVE_ROOT}/outputs'
os.makedirs(out_target, exist_ok=True)
if not os.path.exists('outputs'):
    os.symlink(out_target, 'outputs')
    print(f'  NEW  outputs/  -> {out_target}')
else:
    print(f'  OK   outputs/  (already linked)')
"""))


# ---------- Step 4: Manifest check -----------------------------------------------
CELLS.append(md("""\
---
## Step 4 — Verify what's on disk

Check which data artifacts exist. Anything marked MISSING needs to be produced by
running the data-collection notebook (see "What to do next" below).
"""))

CELLS.append(code("""\
PROCESSED = Path('processed')

REQUIRED = [
    ('m1_train_subset.jsonl',     'M1 official-format subset (for audits)'),
    ('m1_val.jsonl',              'M1 val (for audits)'),
    ('m2_train_pairs.jsonl',      'M2 training pairs (1621 expected)'),
    ('m2_val_pairs.jsonl',        'M2 val pairs for early stopping (300 expected)'),
    ('consistency_eval.jsonl',    'Primary eval set (600 pos+neg pairs)'),
    ('thumbnail_index.jsonl',     'Real YouTube thumbnails index'),
    ('m2_image_emb.npz',          'Cached CLIP frame embeddings'),
    ('thumbnail_image_emb.npz',   'Cached CLIP thumbnail embeddings'),
    ('category_map.json',         'qid -> category for stratification'),
]

print(f\"{'File':40s}  {'Status':10s}  {'Size':>12s}  Purpose\")
print('-' * 110)
ok_count = miss_count = 0
for name, purpose in REQUIRED:
    p = PROCESSED / name
    if p.exists():
        sz = p.stat().st_size
        sz_s = f'{sz/1024:.1f} KB' if sz < 1e6 else f'{sz/1e6:.1f} MB'
        print(f'{name:40s}  {\"OK\":10s}  {sz_s:>12s}  {purpose}')
        ok_count += 1
    else:
        print(f'{name:40s}  {\"MISSING\":10s}  {\"-\":>12s}  {purpose}')
        miss_count += 1

print('-' * 110)
print(f'Present : {ok_count}/{len(REQUIRED)}')
print(f'Missing : {miss_count}/{len(REQUIRED)}')

# Moment-DETR features
clip_feat_dir = Path('Moment_DETR/moment_detr/features/clip_features')
text_feat_dir = Path('Moment_DETR/moment_detr/features/clip_text_features')
if clip_feat_dir.exists():
    n_v = len(list(clip_feat_dir.glob('*.npz')))
    n_t = len(list(text_feat_dir.glob('*.npz'))) if text_feat_dir.exists() else 0
    print(f'\\nM1 features:  clip_features={n_v}, clip_text_features={n_t}')
else:
    print(f'\\nM1 features:  MISSING (unpack moment_detr_features.tar.gz first)')
"""))


# ---------- Step 5: What to do next ---------------------------------------------
CELLS.append(md("""\
---
## Step 5 — What to do next

You're now ready to open any of the downstream notebooks. Click any link below to
open it in the same Colab session (it inherits the bootstrap state).

> Tip: Colab opens notebooks via `File -> Open notebook -> GitHub` or directly via
> the `notebooks/` folder in the left sidebar (already cloned for you).

### Recommended order

| # | Notebook | Purpose | Compute | Time |
|---|---|---|---|---|
| 1 | `notebooks/pipeline_walkthrough.ipynb` | Understand every module via synthetic data | CPU | ~3 min |
| 2 | `notebooks/data_collection_preparation.ipynb` | If any artifact is MISSING above, run the relevant section | Mixed | varies |
| 3 | `notebooks/sanity_consistency.ipynb` | **R1 GATE** — GT-query AUC on 600 pairs (must be >= 0.75) | T4 GPU | ~10 min |
| 4 | `notebooks/train_clipcap.ipynb` | Train ClipCap M2 + generate K=3 queries | T4 GPU | ~2 h |
| 5 | `notebooks/run_evaluation.ipynb` | 5-source ablation + headline AUC table | T4 GPU | ~30 min |

### Gate logic

```
   Walkthrough OK?
        |
        v
   All artifacts present?   ----> NO ----> Run data_collection sections 15/17/18
        |
        YES
        v
   Sanity AUC >= 0.75 ?     ----> NO ----> STOP, score formula is broken
        |
        YES
        v
   Train ClipCap
        |
        v
   Run full evaluation
```

### To run unit tests right now (sanity check)
"""))

CELLS.append(code("""\
# Run the pure-math unit tests. Should pass 18/18.
import subprocess
r = subprocess.run(['python', '-m', 'pytest', 'tests/', '-q'],
                   capture_output=True, text=True)
print(r.stdout)
print(r.stderr)
"""))


# ---------- Bottom: helpful commands --------------------------------------------
CELLS.append(md("""\
---
## Quick commands (just in case)

| Goal | Command in a Colab cell |
|---|---|
| Pull latest code from GitHub | `!cd /content/repo && git pull --ff-only` |
| Check which branch you're on | `!cd /content/repo && git status` |
| Re-mount Drive after disconnect | `from google.colab import drive; drive.mount('/content/drive', force_remount=True)` |
| Inspect a saved JSONL | `!head -2 processed/consistency_eval.jsonl` |
| Free up GPU memory mid-session | `import gc, torch; gc.collect(); torch.cuda.empty_cache()` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `fatal: could not read Username for 'https://github.com'` | Repo is private. Re-run Step 2 — it will prompt for a PAT this time. |
| `DRIVE_ROOT not found` in Step 1 | Drive isn't mounted, or the folder isn't named `thumbnail_project` at the top of MyDrive. |
| Symlinks created but `MISSING` in manifest | Drive folder is empty. Run `data_collection_preparation.ipynb` to populate. |
| `pytest` reports `No module named src` | Make sure Step 2 finished and `/content/repo` is on `sys.path`. Re-run Step 2 if needed. |

---

*Last reviewed: this launcher assumes branch `feature/v1-pipeline`. If your branch
name differs, edit `BRANCH` in Step 2.*
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
