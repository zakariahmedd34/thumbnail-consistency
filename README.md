# Thumbnail–Video Consistency

> **Question this project answers:** *Does this thumbnail really show what happens in the video?*
>
> We give the system a **thumbnail image** and a **video**. It returns a **score**.
> High score = the thumbnail matches the video. Low score = the thumbnail is misleading (clickbait).
> 
![Thumbnail–video consistency pipeline](images/thumbnail_consistency_pipeline_vf.png)

1. **Describe the thumbnail.** The M2 generator (VLM or ClipCap) writes K descriptions. CLIP turns them
   into a pooled text embedding `e_qk` (one per description) and into text tokens for Moment-DETR.
   The thumbnail itself goes through the CLIP image encoder to get `e_T`.
   (The `gt` and `random` generators are baselines and plug into the same M2 slot.)
2. **Find and pool the relevant video content.** Moment-DETR (M1) reads the video features
   (CLIP, one per 2-second clip) together with the text tokens. Its saliency head gives a score
   `s_k,t` for every clip. We keep the top clips in each window and average them, weighted by saliency,
   to get `e_w,k`. The moment spans that Moment-DETR predicts (gray branch) are shown in the output
   but are **not used** in the score.
3. **Calculate the score.** Cosine similarity text–video and thumbnail–video, weighted sum with `alpha`,
   max over windows, mean over the K descriptions → `C(T, V)`.

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

| Name | What it does |
|---|---|
| `gt` | Uses the real human-written QVHighlights query. **Best possible case (upper bound).** |
| `random` | Picks a random query. **Control: should score badly (AUC ≈ 0.5).** |
| `vlm` | A vision-language model (Qwen2-VL-2B, 4-bit) describes the image. |
| `clipcap` | Our own small trained model: CLIP + a small MLP + GPT-2.|

You can also skip M2 completely and type your own queries with `--queries`.

---

## Project Structure

```
thumbnail-consistency/
├── README.md
├── requirements.txt
├── config.yaml                  # Project settings
│
├── src/                         # Core pipeline
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration loading
│   ├── scoring.py               # Consistency scoring
│   ├── metrics.py               # Evaluation metrics
│   ├── encoders.py              # Shared image and text encoders
│   ├── moment_detr.py           # Video moment retrieval
│   └── generators/              # Caption generation methods
│       ├── __init__.py
│       ├── baselines.py         # Reference baselines
│       ├── vlm.py               # Vision-language captioning
│       └── clipcap.py           # ClipCap training and inference
│
├── app/                         # Web interface
│   ├── server.py                # Application routes
│   ├── templates/
│   └── static/
│
├── data/
│   └── processed/               # Dataset splits and evaluation sets
├── notebooks/
│   └── data_prep_colab.ipynb     # Dataset preparation
│
├── tests/
├── images/
├── outputs/                     # results written by the tools
├── checkpoints/                 # Trained model weights
└── Moment_DETR/                 # upstream Moment-DETR code
```
---

## Getting Started

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

## How to run it

### 1. Command line

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

### 2. Flask app

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

## Ready example to try

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

---

## How the system is evaluated

There are **3 test sets** in `data/processed/` (more details in `data/processed/README.md`).
Each has 300 good pairs (label 1) and about 300 bad pairs (label 0):

| Set | How the bad pairs are made | What it tests |
|---|---|---|
| `easy` (`consistency_eval.jsonl`) | Thumbnail from video A, scored against a different video B | Can it tell different videos apart? |
| `hard` (`consistency_eval_hard.jsonl`) | Same video, but a frame from **outside** the matching moment | Can it find the **right moment**? |
| `query_mismatch` | Same thumbnail and video, but a query about another topic | Does it really **use the query**? |

**Metrics:** **AUC** (0.5 = random guessing, 1.0 = perfect) and **Cohen's d** (how far apart the
good and bad scores are; above 0.8 is a big difference).

---
