# Thumbnail–Video Consistency

> **Question this project answers:** *Does this thumbnail really show what happens in the video?*
>
> We give the system a **thumbnail image** and a **video**. It returns a **score**.
> High score = the thumbnail matches the video. Low score = the thumbnail is misleading (clickbait).
---

## Contents

1. [The idea in simple words](#1-the-idea-in-simple-words)
2. [How the pipeline works](#2-how-the-pipeline-works)
3. [Project folders and files](#3-project-folders-and-files)
4. [Setup (first time only)](#4-setup-first-time-only)
5. [How to run it](#5-how-to-run-it)
6. [Ready example to try](#6-ready-example-to-try)
7. [Evaluation and results so far](#7-evaluation-and-results-so-far)
8. [Current status: what works, what is pending](#8-current-status-what-works-what-is-pending)
9. [Common problems](#9-common-problems)
10. [Changelog](#changelog)

---

## 1. The idea in simple words

A thumbnail is a promise: "this is what you will see". To check that promise we:

1. **Describe the thumbnail in words.** A model looks at the image and writes short sentences
   ("queries"), for example *"a man shows places to visit in Istanbul"*. This is **M2**.
2. **Search the video for that description.** A video model (**Moment-DETR**, called **M1**)
   finds *which seconds of the video* match the sentence.
3. **Compare.** We check how similar the thumbnail and the sentence are to those found moments.
   If the video really contains what the thumbnail shows, the similarity is high.

All the comparing is done with **CLIP**, a model that turns images and text into vectors
(lists of numbers). Similar things get similar vectors.

---

## 2. How the pipeline works

```
 Thumbnail ──► M2 generator ──► K queries (sentences)
     │                               │
     │ CLIP image                    │ CLIP text
     ▼                               ▼
    e_T                            e_q (one per query)
     │                               │
     │      Video features ──► Moment-DETR (M1) ──► "saliency": how relevant each 2-second clip is
     │            │                                   │
     │            └──── weighted average of the most relevant clips ◄──┘
     │                               │
     └──────────────► compare ◄──────┘
                         ▼
                 score C (one number)
```

**The formula** (in `src/scoring.py`):

```
C = average over queries of  max over video windows of
        [ alpha · cos(query, window) + (1 − alpha) · cos(thumbnail, window) ]
```

| Setting (in `config.yaml`) | Meaning | Default |
|---|---|---|
| `alpha` | How much the **query** matters vs. the **thumbnail image** itself | 0.5 |
| `tau` | How "sharp" the focus on the most relevant clips is (smaller = sharper) | 0.1 |
| `top_p` | Fraction of clips kept when averaging (most relevant first) | 0.25 |
| `chunk_window` / `chunk_stride` | Long videos are cut into 150 s windows (Moment-DETR limit is 75 clips) | 75 / 60 |

### The M2 generators (who writes the queries)

| Name | What it does | Needs GPU? | Status |
|---|---|---|---|
| `gt` | Uses the real human-written QVHighlights query. **Best possible case (upper bound).** | No | ✅ works |
| `random` | Picks a random query. **Control: should score badly (AUC ≈ 0.5).** | No | ✅ works |
| `vlm` | A vision-language model (Qwen2-VL-2B, 4-bit) describes the image. | Yes | ⏳ model download not finished |
| `clipcap` | Our own small trained model: CLIP + a small MLP + GPT-2. **The only part we train.** | Yes (training) | ⏳ needs training |

You can also skip M2 completely and type your own queries with `--queries`.

---

## 3. Project folders and files

```
thumbnail-consistency/
├── README.md            ← this document
├── config.yaml          ← THE single settings file (paths, scoring, models, app)
├── requirements.txt     ← Python packages
├── src/                 ← all the real logic
│   ├── cli.py           ← command-line tool: check / generate / score / evaluate / train-clipcap
│   ├── config.py        ← reads config.yaml, resolves file paths (also fixes old Colab paths)
│   ├── scoring.py       ← ★ the score formula + ConsistencyScorer (the heart of the project)
│   ├── moment_detr.py   ← wrapper around Moment-DETR (M1)
│   ├── encoders.py      ← CLIP image/text encoders (loaded once, cached)
│   ├── metrics.py       ← AUC, Cohen's d, Welch t-test
│   └── generators/      ← M2 generators
│       ├── __init__.py  ← registry: build("gt"), available()
│       ├── baselines.py ← gt and random
│       ├── vlm.py       ← Qwen2-VL (any Hugging Face vision-language model)
│       └── clipcap.py   ← ClipCap model + its training loop
├── app/                 ← small website (Flask) around the same pipeline
│   ├── server.py        ← routes: /, /health, /v1/generators, /v1/score
│   ├── templates/       ← HTML page
│   └── static/          ← CSS
├── tests/               ← pytest tests
├── notebooks/
│   └── data_prep_colab.ipynb  ← how the data in data/ was built (run on Colab)
├── data/                ← data (big files are NOT in git, see below)
│   └── processed/       ← JSONL files: train/val/test + the 3 evaluation sets (in git)
├── Moment_DETR/         ← upstream Moment-DETR code + checkpoint (NOT in git, you clone it)
├── outputs/             ← results written by the tools (NOT in git)
└── checkpoints/         ← trained ClipCap weights (NOT in git)
```

**Rule:** logic lives in `src/`. The CLI and the app only call `src/`.

**What is NOT pushed to GitHub** (too big, see `.gitignore`): `.venv/`, `Moment_DETR/`,
`data/features/`, `data/qvh_frames*/`, `data/qvh_thumbnails/`, `*.tar.gz`, `*.pt`, `*.ckpt`,
`outputs/`, `checkpoints/`.

---

## 4. Setup (first time only)

Commands are for **Windows PowerShell**, run from the project folder.

**Step 1 — Python environment**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
# Install PyTorch first (pick the CUDA version of your GPU from pytorch.org), for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Step 2 — Moment-DETR (M1) code + checkpoint**

```powershell
git clone https://github.com/jayleicn/moment_detr Moment_DETR/moment_detr
```

The checkpoint is already inside it at
`Moment_DETR/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt`.

**Step 3 — Data**

- **Video features:** the official QVHighlights CLIP features (`moment_detr_features.tar.gz`)
  extracted so the `.npz` files are in `data/features/clip_features/`.
- **Frames / thumbnails:** `data/qvh_frames/`, `data/qvh_frames_within_neg/`, `data/qvh_thumbnails/`,
  made by `notebooks/data_prep_colab.ipynb` (keep them on Google Drive, copy them here).
- Want the data in another folder? Set `$env:THUMB_DATA_DIR = "D:\path\to\data"`.

**Step 4 — Check everything**

```powershell
python -m src.cli check        # lists every package / file as [ok] or [MISSING]
python -m pytest tests -q      # run the tests
```

---

## 5. How to run it

Put these two lines first in every new PowerShell window (offline mode = don't try to download models;
utf-8 = no text-printing errors):

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"
```

### A. Command line (`python -m src.cli ...`)

| Command | What it does |
|---|---|
| `python -m src.cli check` | Show what is installed and what is missing |
| `python -m src.cli generate --image IMG --gen vlm -k 3` | Only M2: thumbnail → queries |
| `python -m src.cli score --image IMG --vid VID --gen gt --qid QID` | Full score using the real query |
| `python -m src.cli score --image IMG --vid VID --queries "your sentence"` | Full score with your own sentence |
| `python -m src.cli evaluate --set easy --gen gt random --limit 100` | AUC on an evaluation set |
| `python -m src.cli train-clipcap --epochs 20` | Train ClipCap (saves to `checkpoints/clipcap_best.pt`) |

- `VID` is a QVHighlights video id like `QriapOfdU64_60.0_210.0` (its `.npz` must exist in the features folder).
- `--set` can be `easy`, `hard`, `query_mismatch`, or a path to your own JSONL.
- `--output result.json` saves the score result to a file.

**What the score output means:**

| Field | Meaning |
|---|---|
| `C` | **The final consistency score.** Higher = more consistent. |
| `per_query` | Score for each query (C is their average) |
| `max_fg_prob` | How confident Moment-DETR is that it found a matching moment (0–1) |
| `top_moments` | `[start_sec, end_sec, confidence]` of the best moment per query |
| `thumb_video_max_cos` | Best direct similarity between the thumbnail and any video clip |

### B. Website (Flask app)

```powershell
python -m app --port 5050
# open http://127.0.0.1:5050 in the browser
```

Upload a thumbnail, write the video id, choose a generator (or type queries) → get the score
and a verdict (`consistent` if score ≥ `app.threshold` in `config.yaml`).

API for other programs:

- `GET /health` → which models/files were found
- `GET /v1/generators` → list of generators
- `POST /v1/score` → JSON `{"thumbnail_path", "vid", "gen", "k", "qid", "queries"}` or a form upload with a `thumbnail` file

---

## 6. Ready example to try

From `consistency_eval_hard.jsonl`: video `QriapOfdU64_60.0_210.0`, query id `5399`:
*"The man in the black, white and yellow t-shirt is showing places to be visited in Istanbul, Turkey"*.

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"

# 1) Correct thumbnail + correct query  -> expect a HIGHER C
python -m src.cli score --image data\qvh_frames\QriapOfdU64_60.0_210.0.jpg --vid QriapOfdU64_60.0_210.0 --gen gt --qid 5399 -k 1

# 2) Same thumbnail, unrelated query  -> expect a LOWER C
python -m src.cli score --image data\qvh_frames\QriapOfdU64_60.0_210.0.jpg --vid QriapOfdU64_60.0_210.0 --queries "A woman cooks pasta in a kitchen."

# 3) Thumbnail from outside the matching moment (hard negative)  -> expect a slightly LOWER C
python -m src.cli score --image data\qvh_frames_within_neg\QriapOfdU64_60.0_210.0_clip0.jpg --vid QriapOfdU64_60.0_210.0 --gen gt --qid 5399 -k 1

# 4) Small evaluation (fast, no GPU model needed)
python -m src.cli evaluate --set easy --gen gt random --limit 100
```

> The numbers from these commands have not been written here yet. Add them after the first run.

---

## 7. Evaluation and results so far

There are **3 test sets** in `data/processed/` (more details in `data/processed/README.md`).
Each has 300 good pairs (label 1) and about 300 bad pairs (label 0):

| Set | How the bad pairs are made | What it tests |
|---|---|---|
| `easy` (`consistency_eval.jsonl`) | Thumbnail from video A, scored against a different video B | Can it tell different videos apart? |
| `hard` (`consistency_eval_hard.jsonl`) | Same video, but a frame from **outside** the matching moment | Can it find the **right moment**? |
| `query_mismatch` | Same thumbnail and video, but a query about another topic | Does it really **use the query**? |

**Metrics:** **AUC** (0.5 = random guessing, 1.0 = perfect) and **Cohen's d** (how far apart the
good and bad scores are; above 0.8 is a big difference).

**Results (all 600 pairs, k = 1, alpha = 0.5):**

| Generator | Set | AUC | Cohen's d | Mean good | Mean bad |
|---|---|---|---|---|---|
| `gt` | easy | **0.961** | 2.39 | 0.298 | 0.206 |
| `random` | easy | 0.568 | 0.23 | 0.211 | 0.201 |
| `gt` | hard | 0.637 | 0.45 | 0.511 | 0.481 |
| `gt` | query_mismatch | **0.991** | 3.24 | 0.298 | 0.172 |

**What this tells us:**
- With good queries the score separates matching and non-matching videos very well (0.96).
- Random queries are close to 0.5, as expected, so the score is not cheating.
- `query_mismatch` at 0.99: the score really depends on the query.
- `hard` is the weak point (0.64): finding the exact moment inside the same video is difficult.
  This is the main thing to improve.

---

## 8. Current status: what works, what is pending

**✅ Works now (no big model download needed):**
- `check`, `score` with `gt` or your own `--queries`, `evaluate` with `gt` / `random`
- The website with `gt` or typed queries
- Tests in `tests/`

**⏳ Pending:**
- **`vlm` generator:** the Qwen2-VL-2B download was stopped (the PC ran low on memory). Run it again when RAM is free.
- **`clipcap` generator:** training was stopped for the same reason. Run
  `python -m src.cli train-clipcap` (or `--max-train 2000` for a small quick test).
- Comparing `vlm` and `clipcap` against `gt` on all 3 sets.
- Improving the `hard` set score.

**Tips for a 4 GB GPU / low-RAM PC:** close other programs, run one heavy job at a time,
use `--limit` in `evaluate`, and keep `load_in_4bit: true` for the VLM.

---

## 9. Common problems

| Problem | Reason | Fix |
|---|---|---|
| `check` shows `Moment-DETR code MISSING` | `Moment_DETR/` not cloned | Do Setup step 2 |
| `video features MISSING` / `FileNotFoundError ... .npz` | Features not extracted, or wrong `--vid` | Extract into `data/features/clip_features/`; check the vid name |
| `generator 'gt' requires qid` | `gt` needs the query id | Add `--qid` |
| `unknown generator` | Name is wrong | Use `gt`, `random`, `vlm`, or `clipcap` |
| Model tries to download / hangs | Not offline | `$env:HF_HUB_OFFLINE = "1"` (only after the model is downloaded once) |
| `UnicodeEncodeError` when printing | Windows console encoding | `$env:PYTHONIOENCODING = "utf-8"` |
| CUDA out of memory | GPU too small | Use the 2B VLM in 4-bit, smaller `-k`, close other programs |
| Process killed / PC freezes | Not enough RAM | Run one heavy job at a time |

---
