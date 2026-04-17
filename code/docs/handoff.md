# Project Handoff — Automatic Video Thumbnail Consistency Detection

> This document is written to be read by a team member or AI agent continuing this project.
> Read it fully before touching any code.

---

## What This Project Is

A deep learning research project for automatic YouTube thumbnail selection.
The core research question is:

> **Given a video and its thumbnail, is the thumbnail genuinely representative of the video content, or is it clickbait?**

This is framed as a self-supervised binary classification / scoring problem.
No human-labeled ground truth is assumed during model training.

---

## Why This Problem Is Hard

Most existing papers treat creator-chosen thumbnails as ground truth and train directly on them.
This is wrong because:
- YouTube thumbnails are frequently clickbait (shocked faces, misleading text overlays)
- Training on noisy labels teaches the model to predict clickbait style, not video content
- The model cannot distinguish a good frame from a clickbait frame if it learned from clickbait

This project solves that by separating the problem into two stages.

---

## The Two-Stage Pipeline

### Stage 1 — Thumbnail-Video Consistency Scoring (MAIN CONTRIBUTION)

**Goal:** Learn what the video is actually about, then score the thumbnail against that understanding.

**Key insight:** Frames and transcript segments are guaranteed to be about the same thing because they are temporally co-produced during filming. Neither is chosen after the fact. The thumbnail is the only artifact chosen by the creator after filming — and therefore the only thing that can be misleading.

**Training approach (self-supervised, no labels needed):**
- Build `(frame_t, transcript_segment_t)` pairs aligned by timestamp
- Train a contrastive model (InfoNCE loss) where same-video pairs are positive and cross-video pairs are negative
- The model learns a shared embedding space where frames and spoken words that describe the same content are close together

**Inference:**
```
video_context = mean(embed(frames) + embed(transcript))
consistency_score = cosine_sim(video_context, embed(thumbnail))
```
High score = thumbnail looks like real video content.
Low score = thumbnail may be clickbait or unrelated.

**Optional manual labeling (active learning):**
- Label 300–500 videos manually as: `representative`, `clickbait`, `partial`, `unclear`
- Use as test set ground truth and/or active learning seed
- The `annotation.json` file in each video folder is the placeholder for this

---

### Stage 2 — Frame Retrieval (secondary, builds on Stage 1)

**Goal:** Given a video with a verified-consistent thumbnail, find which frame in the video is most similar to it.

**Approach:**
- Train only on videos where Stage 1 consistency score ≥ threshold τ (clean subset)
- Model predicts a target embedding vector ô
- Retrieve frame with argmin MSE(ô, embed(frame_i))
- Evaluate with Precision@θ (same metric as Yu et al. 2101.00073)

**Why train only on the clean subset:**
Training on all videos (including clickbait) teaches the model to predict clickbait-style thumbnails.
Training only on consistent videos teaches it to predict representative thumbnails.
The improvement in Precision@θ between these two conditions is the main empirical result.

---

## Current Status — What Has Been Done

### ✅ Completed
- Full research design and problem formulation (documented in `project_summary.md`)
- Data collection guide (`data_collection_guide.md`)
- Data collection pipeline notebook (`Thumbnail_Test_Data_Collection_Pipeline.ipynb`)
- Pipeline tested on 1 video (YouTube ID: `oeqPrUmVz-o`, ~5 min, English)
- All pipeline stages produce correct output files
- Frame-transcript alignment pairs verified to work

### ⚠️ Known Bugs in Notebook (fix before scaling)
1. **Timestamp offset:** `frames_index.json` uses `float(i + 1)` but should use `float(i)` — causes 1-second misalignment between frames and transcript
2. **Alignment not saved:** The alignment cell computes `(frame, transcript_text)` pairs but only prints them — must save to `aligned_pairs.json`
3. **Frame filtering not implemented:** Blurry/dark frame removal is marked `TODO`
4. **No skip logic:** Re-running the notebook re-processes already-processed videos — add existence checks
5. **yt-dlp JS warning:** Add `--extractor-args "youtube:player_client=android"` to download command

### ❌ Not Started
- Frame filtering implementation (blur, brightness, near-duplicate detection)
- Stage 1 model (contrastive frame-transcript encoder)
- Manual annotation of 300–500 videos
- Stage 2 model (frame retrieval)
- Evaluation scripts

---

## Data Format — Every Video Folder

```
raw/
  {youtube_video_id}/
    video.mp4              ← original video (can delete after processing)
    thumbnail.jpg          ← highest resolution thumbnail from YouTube
    audio.wav              ← 16kHz mono WAV (can delete after transcription)
    title.txt              ← raw video title
    description.txt        ← full creator description
    meta.json              ← structured metadata (see schema below)
    transcript.json        ← Whisper output with word-level timestamps
    frames_index.json      ← maps each frame filename to its timestamp in seconds
    aligned_pairs.json     ← (frame_file, transcript_text) pairs — NEEDED FOR STAGE 1
    annotation.json        ← manual label placeholder (fill for labeled subset)
    frames/
      frame_00001.jpg
      frame_00002.jpg
      ...
```

### meta.json schema
```json
{
  "video_id": "oeqPrUmVz-o",
  "title": "...",
  "duration_seconds": 314,
  "upload_date": "20231015",
  "channel_id": "UC...",
  "channel_name": "...",
  "category": ["..."],
  "tags": ["...", "..."],
  "view_count": 142000,
  "like_count": 8300,
  "language": "en",
  "thumbnail_url": "https://i.ytimg.com/vi/.../maxresdefault.jpg"
}
```

### transcript.json schema (Whisper output)
```json
[
  {
    "id": 0,
    "start": 0.0,
    "end": 3.2,
    "text": " Welcome to my channel.",
    "words": [
      {"word": "Welcome", "start": 0.1, "end": 0.5, "probability": 0.98}
    ]
  }
]
```

### frames_index.json schema
```json
[
  {"frame_file": "frame_00001.jpg", "timestamp_sec": 0.0},
  {"frame_file": "frame_00002.jpg", "timestamp_sec": 1.0}
]
```

### aligned_pairs.json schema (TO BE SAVED — currently missing)
```json
[
  {"frame_file": "frame_00001.jpg", "timestamp_sec": 0.0, "transcript_text": ""},
  {"frame_file": "frame_00002.jpg", "timestamp_sec": 1.0, "transcript_text": "Welcome to my channel."}
]
```

### annotation.json schema (manual labels)
```json
{
  "video_id": "oeqPrUmVz-o",
  "annotator": "name",
  "annotation_date": "2024-01-15",
  "thumbnail_label": null,
  "thumbnail_label_confidence": null,
  "notes": ""
}
```
Label options: `representative`, `clickbait`, `partial`, `unclear`
Confidence options: `high`, `medium`, `low`

---

## Team Workflow

### Storage
- All data lives on **Google Drive**: `MyDrive/thumbnail_project/raw/`
- Mount in Colab with:
```python
from google.colab import drive
drive.mount('/content/drive')
RAW_DIR = "/content/drive/MyDrive/thumbnail_project/raw"
```

### Task Division
| Task | GPU needed | Where to run |
|---|---|---|
| Download videos (section 2) | No | Local or Colab CPU |
| Extract metadata (section 3) | No | Local or Colab CPU |
| Extract frames (section 4) | No | Local or Colab CPU |
| Extract audio (section 5) | No | Local or Colab CPU |
| Transcription (section 6) | YES | Colab GPU (T4) |
| Alignment (section 7) | No | Local or Colab CPU |
| Manual annotation | No | Edit annotation.json directly |

### Speed estimates on Colab T4 GPU
| Video length | Whisper small | Whisper medium |
|---|---|---|
| 5 min | ~30 sec | ~90 sec |
| 15 min | ~90 sec | ~4 min |

### Storage estimates per video
| File | Size |
|---|---|
| video.mp4 | ~12 MB |
| frames/ (1fps, 5min video) | ~11 MB |
| audio.wav | ~10 MB |
| transcript.json | ~50 KB |
| thumbnail.jpg | ~200 KB |
| Total (with video) | ~33 MB |
| Total (after deleting video + audio) | ~12 MB |

**Delete `video.mp4` and `audio.wav` after processing.** They are not needed for training.

---

## Immediate Next Steps (in order)

### Step 1 — Fix the 5 notebook bugs
- Fix timestamp: `float(i)` not `float(i + 1)`
- Save aligned_pairs.json in section 7
- Implement frame filtering (blur + brightness + near-duplicate)
- Add skip logic to all processing cells
- Add `--extractor-args "youtube:player_client=android"` to yt-dlp

### Step 2 — Add Google Drive mount to notebook top

### Step 3 — Add validation cell at end
Check that every video folder has all required files before calling it done.

### Step 4 — Test on 10 videos across 3–4 categories

### Step 5 — Scale to 100 videos, verify storage is manageable

### Step 6 — Manual annotation of 300–500 videos
Fill `annotation.json` for each. Use `high` and `medium` confidence labels only.
Aim for 150 `representative` + 150 `clickbait` minimum.

### Step 7 — Begin Stage 1 model
- Two-tower contrastive model (ViT-B/16 for frames, BERT for transcript)
- InfoNCE loss on `aligned_pairs.json`
- Evaluate by scoring thumbnails and comparing to manual annotations

---

## Key Design Decisions Already Made

| Decision | Choice | Reason |
|---|---|---|
| Primary training signal | frame-transcript temporal alignment | Causally guaranteed consistent, no human labels needed |
| Do NOT use | thumbnail as training signal | Creator thumbnails are noisy — this is the problem being solved |
| Evaluation metric Stage 1 | Precision/Recall vs manual annotation labels | Clean ground truth, interpretable |
| Evaluation metric Stage 2 | Precision@θ (MSE threshold) | Same as Yu et al. — directly comparable |
| Dataset split unit | By channel, not by video | Prevents model from memorizing creator style |
| Video language | English only | Consistent Whisper performance |
| Whisper model | small for development, medium for final | Speed/accuracy tradeoff |
| Frame rate | 1 fps | Sufficient temporal coverage, manageable storage |

---

## Papers Referenced

| Paper | What it contributes |
|---|---|
| Yu et al. (arXiv 2101.00073) | Clean pipeline: predict embedding → retrieve nearest frame by MSE, Precision@θ evaluation, context gating |
| Fantini EPFL thesis (arXiv 2410.19825) | Multi-criteria scoring, variety enhancement, face quality signals, no-RNN conclusion |

---

## Recommended Project Title

> **Annotation-Free Video Thumbnail Consistency Scoring via Frame-Transcript Contrastive Learning**

---

## Repository Structure (target)

```
thumbnail-project/
  notebooks/
    Thumbnail_Test_Data_Collection_Pipeline.ipynb
  docs/
    project_summary.md
    data_collection_guide.md
    handoff.md                   ← this file
  scripts/                       ← to be created
    validate_dataset.py
    build_splits.py
  requirements.txt
  README.md
```

---

## Contact / Context

This project was designed over an extended research design conversation.
The full reasoning behind every decision is in `project_summary.md`.
Read that document before questioning any design choice — most objections were already considered and resolved.
