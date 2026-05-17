"""
Inject a Colab bootstrap cell into the three modular notebooks so they:

  1. Mount Drive
  2. Find the repo at /content/drive/MyDrive/thumbnail_project/thumbnail-consistency
  3. Register src/ on sys.path
  4. Extract Moment-DETR features to /content/features_local (NOT Drive)
  5. Clone upstream moment_detr code for the M1 wrapper
  6. Verify everything is in place

Idempotent: cells already carrying the v1-bootstrap sentinel are skipped.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
TARGETS = [
    "sanity_consistency.ipynb",
    "train_clipcap.ipynb",
    "run_evaluation.ipynb",
]

MARK = "# [COLAB_BOOTSTRAP_v2_modular]"

BOOTSTRAP_CODE = f'''{MARK}
# Mount Drive + clone upstream moment_detr + extract features to LOCAL SSD.
# Run once at the top of any modular notebook. Idempotent within a session.

import os, sys, subprocess, time, tarfile, shutil
from pathlib import Path

# --- 1. Drive --------------------------------------------------------------
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/thumbnail_project')
REPO_DIR   = DRIVE_ROOT / 'thumbnail-consistency'
assert REPO_DIR.exists(), (
    f"Upload your repo folder to {{REPO_DIR}} first. "
    f"Found nothing at that path."
)

# --- 2. Put repo on sys.path (so `from src.*` works) -----------------------
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# --- 2b. Symlink data dirs INTO the repo so relative paths work ------------
# Notebooks reference 'processed/X.jsonl', 'outputs/Y.png' etc., assuming
# those dirs live INSIDE the repo. In your Drive layout they sit ALONGSIDE
# the repo. Symlinks bridge that gap.
DATA_DIRS_TO_LINK = [
    'processed', 'outputs',
    'qvh_videos', 'qvh_frames', 'qvh_thumbnails',
    'qvh_frames_within_neg', 'qvh_frames_multi',
    'checkpoints',
]
for d in DATA_DIRS_TO_LINK:
    src = DRIVE_ROOT / d
    dst = REPO_DIR / d
    if not src.exists():
        continue
    if dst.exists() or dst.is_symlink():
        continue
    try:
        os.symlink(src, dst)
        print(f"  symlinked  {{d}}/  ->  {{src}}")
    except OSError as e:
        print(f"  WARN: couldn't symlink {{d}}: {{e}}")

# --- 3. Extract M1 features to LOCAL SSD (NOT to Drive) --------------------
# Drive cannot reliably store 19k small .npz files; we extract every session.
ARCHIVE   = DRIVE_ROOT / 'moment_detr_features.tar.gz'
LOCAL_OUT = Path('/content/features_local')


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


CLIP_FEAT_DIR, TEXT_FEAT_DIR = (None, None)
if LOCAL_OUT.exists():
    CLIP_FEAT_DIR, TEXT_FEAT_DIR = _find_feature_dirs(LOCAL_OUT)

if CLIP_FEAT_DIR and TEXT_FEAT_DIR:
    print(f"[features] already extracted this session.")
else:
    assert ARCHIVE.exists(), f"Archive missing: {{ARCHIVE}}"
    if LOCAL_OUT.exists():
        shutil.rmtree(LOCAL_OUT)
    LOCAL_OUT.mkdir(parents=True)
    print(f"[features] extracting {{ARCHIVE.stat().st_size/1e9:.2f}} GB -> {{LOCAL_OUT}}")
    t0 = time.time()
    with tarfile.open(ARCHIVE) as tf:
        tf.extractall(LOCAL_OUT)
    print(f"[features] extracted in {{time.time()-t0:.0f}}s")
    CLIP_FEAT_DIR, TEXT_FEAT_DIR = _find_feature_dirs(LOCAL_OUT)

assert CLIP_FEAT_DIR and TEXT_FEAT_DIR, "Feature dirs not found after extraction."

# --- 4. Clone upstream Moment-DETR (used by src.models.moment_detr) --------
MD_DIR = Path('/content/moment_detr_upstream')
if not MD_DIR.exists():
    subprocess.run(['git', 'clone',
                    'https://github.com/jayleicn/moment_detr.git',
                    str(MD_DIR)], check=True)
if str(MD_DIR) not in sys.path:
    sys.path.insert(0, str(MD_DIR))

# --- 5. Expose path variables for downstream cells -------------------------
# Any cell that reads configs/*.yaml can override:
#   cfg['paths']['feature_dir']   = str(CLIP_FEAT_DIR)
#   cfg['paths']['text_feat_dir'] = str(TEXT_FEAT_DIR)
# (configs already point at /content/features_local/... so usually no override needed.)

# --- 6. Verification report -------------------------------------------------
import torch
print()
print(f"Repo            : {{REPO_DIR}}")
print(f"Device          : {{'cuda' if torch.cuda.is_available() else 'cpu (WARN: slow)'}}")
print(f"clip_features   : {{CLIP_FEAT_DIR}}  ({{len(list(CLIP_FEAT_DIR.glob('*.npz'))):,}} files)")
print(f"clip_text_feats : {{TEXT_FEAT_DIR}}  ({{len(list(TEXT_FEAT_DIR.glob('*.npz'))):,}} files)")
print(f"src/ importable : {{(REPO_DIR / 'src' / '__init__.py').exists()}}")
print()
print("Bootstrap done. You can now run the rest of the notebook.")
'''


def main():
    n_patched = 0
    for nb_name in TARGETS:
        path = NOTEBOOKS_DIR / nb_name
        if not path.exists():
            print(f"SKIP (not found): {nb_name}")
            continue

        nb = json.loads(path.read_text(encoding='utf-8'))

        already = any(
            MARK in (''.join(c['source']) if isinstance(c['source'], list) else c['source'])
            for c in nb['cells']
        )
        if already:
            print(f"OK  (already bootstrapped with {MARK}): {nb_name}")
            continue

        # Remove ANY previous bootstrap cell (older version marks)
        OLD_MARKS = ['# [COLAB_BOOTSTRAP_v1_modular]']
        nb['cells'] = [
            c for c in nb['cells']
            if not any(om in (''.join(c['source']) if isinstance(c['source'], list)
                              else c['source']) for om in OLD_MARKS)
        ]

        bootstrap_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": BOOTSTRAP_CODE.splitlines(keepends=True),
        }
        # Prepend as the FIRST cell
        nb['cells'].insert(0, bootstrap_cell)

        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
        print(f"PATCHED: {nb_name}  (bootstrap v2 inserted, old v1 removed)")
        n_patched += 1

    print(f"\n{n_patched} notebook(s) patched.")


if __name__ == "__main__":
    main()
