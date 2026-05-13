# QVHighlights Dataset — Project Data Reference

This document describes the data sources, structure, and processing pipeline used in the **Thumbnail → Query Generator (M2)** project. The project builds on the [QVHighlights dataset](https://github.com/jayleicn/moment_detr) and extends it with custom frame extraction, saliency-based filtering, and training pair construction.

---

## Source Dataset

**QVHighlights** is a large-scale dataset for natural language video moment retrieval and highlight detection. Each annotation links a natural language query to a specific temporal moment within a YouTube video, annotated with fine-grained saliency scores.

- **Paper**: *QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries* (Lei et al., 2021)
- **Repository**: [github.com/jayleicn/moment_detr](https://github.com/jayleicn/moment_detr)
- **License**: Refer to the original repository for licensing terms.

---

## Our Processed Data

All project-specific extractions, processed files, and model checkpoints are stored on Google Drive:

📁 **[Project Drive Folder](https://drive.google.com/drive/folders/1mjf-8nWMT1L1IH4O-ZzeuPj8Vv1H8p1k)**

```
thumbnail_project/
├── processed/
│   ├── train_processed.jsonl       # Cleaned & enriched training annotations
│   ├── m2_train_pairs.jsonl        # (frame_path, query, saliency) pairs for M2
│   ├── m2_split_indices.json       # Reproducible 90/10 train/val split (seed=42)
│   ├── m2_vocab_stats.json         # GPT-2 BPE tokenization stats & max_seq_len
│   ├── download_log.json           # Per-video download status log
│   ├── m2_training_curves.png      # Loss / BLEU-4 / ROUGE-L curves
│   └── m2_qualitative_grid.png     # Generated vs ground-truth visual grid
├── qvh_frames/                     # Extracted saliency-peak JPEG frames
│   └── {vid}.jpg
├── qvh_videos/                     # Downloaded source videos (MP4)
│   └── {vid}.mp4
├── checkpoints/
│   └── m2_best.pt                  # Best M2 checkpoint (by BLEU-4)
└── exports/
    ├── m2_export.pt                # Self-contained export (weights + config + metrics)
    └── m2_export_meta.json         # Human-readable metadata JSON
```

---

## Annotation Format

The original QVHighlights annotations are in [JSON Lines](https://jsonlines.org/) format (`.jsonl`), one record per line. Three splits are provided: `train`, `val`, and `test`.

### Example record

```json
{
    "qid": 8737,
    "query": "A family is playing basketball together on a green court outside.",
    "duration": 126,
    "vid": "bP5KfdFJzC4_660.0_810.0",
    "relevant_windows": [[0, 16]],
    "relevant_clip_ids": [0, 1, 2, 3, 4, 5, 6, 7],
    "saliency_scores": [
        [4, 1, 1], [4, 1, 1], [4, 2, 1], [4, 3, 2],
        [4, 3, 2], [4, 3, 3], [4, 3, 3], [4, 3, 2]
    ]
}
```

### Field descriptions

| Field | Type | Description |
|---|---|---|
| `qid` | int | Unique query identifier |
| `query` | str | Natural language query describing the video moment |
| `duration` | int | Duration of the video clip in seconds |
| `vid` | str | Video identifier: `{youtube_id}_{start_time}_{end_time}` |
| `relevant_windows` | list[list[int]] | `[[start_sec, end_sec], ...]` — temporal windows of the highlighted moment |
| `relevant_clip_ids` | list[int] | IDs of 2-second clips falling within `relevant_windows`, 0-indexed |
| `saliency_scores` | list[list[int]] | Per-clip scores from 3 annotators. `4` = Very Good, `0` = Very Bad |

> **Note**: `relevant_clip_ids`, `relevant_windows`, and `saliency_scores` are **not included** in the `test` split. See the [standalone eval README](https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/README.md) for test set evaluation.

### Retrieving source videos

The `vid` field encodes the YouTube source. To reconstruct the video URL:

```
vid format  :  {youtube_id}_{start_time}_{end_time}
Embed URL   :  https://www.youtube.com/embed/{youtube_id}?start={start_time}&end={end_time}&version=3
```

**Example** — `vid = "bP5KfdFJzC4_660.0_810.0"`:

```
https://www.youtube.com/embed/bP5KfdFJzC4?start=660&end=810&version=3
```

---

## M2 Training Pairs (`m2_train_pairs.jsonl`)

This is the primary input file for Stage 2 (M2 model training). Each record links a single extracted frame to its query and aggregated saliency score.

### Example record

```json
{
    "qid": 8737,
    "vid": "bP5KfdFJzC4_660.0_810.0",
    "frame_path": "/content/drive/MyDrive/thumbnail_project/qvh_frames/bP5KfdFJzC4_660.0_810.0.jpg",
    "query": "A family is playing basketball together on a green court outside.",
    "mean_saliency": 3.25
}
```

### Field descriptions

| Field | Type | Description |
|---|---|---|
| `qid` | int | Query ID (matches original QVHighlights `qid`) |
| `vid` | str | Video ID (matches original `vid`) |
| `frame_path` | str | Absolute path to the extracted JPEG frame on Drive |
| `query` | str | Natural language query (copied from original annotation) |
| `mean_saliency` | float | Mean saliency across all annotators and relevant clips (range: 0–4) |

The `mean_saliency` field is derived by averaging all values in the original `saliency_scores` array. It is used for stratified train/val splitting and for saliency-based analysis of model performance.

---

## Train / Val Split (`m2_split_indices.json`)

The split is fixed and reproducible via `seed=42`. It uses a **stratified 90/10 split** across 5 saliency quintiles to ensure both sets cover the full range of signal quality.

```json
{
    "seed": 42,
    "val_ratio": 0.10,
    "n_total": 1234,
    "n_train": 1111,
    "n_val": 123,
    "train_qids": [...],
    "val_qids": [...]
}
```

---

## Vocabulary Stats (`m2_vocab_stats.json`)

Generated in Stage 2 Phase A from the full training query corpus using the GPT-2 BPE tokenizer. Used to set `max_seq_len` in the M2Dataset.

```json
{
    "n_queries": 1234,
    "n_unique_queries": 1230,
    "word_vocab_size": 3812,
    "mean_word_length": 9.4,
    "gpt2_token_mean": 12.3,
    "gpt2_token_std": 3.1,
    "gpt2_token_p95": 19,
    "gpt2_token_p99": 23,
    "recommended_max_len": 25,
    "queries_exceeding_max": 3,
    "pii_hits": {},
    "offensive_hits": 0
}
```

The `recommended_max_len` value (p99 token length + 2) is used directly as `MAX_SEQ_LEN` in the M2Dataset and model `generate()` call.

---

## Frame Extraction

One frame is extracted per video at the temporal midpoint of the highest-saliency 2-second clip. Frames are saved as JPEG at native video resolution and stored in `qvh_frames/`. File names match the `vid` identifier exactly (e.g. `bP5KfdFJzC4_660.0_810.0.jpg`).

Frames with file sizes below 1 KB are flagged as corrupt during Stage 2 verification (cell A1) and excluded from training.

---

## Download Log (`download_log.json`)

Records the outcome of each video download attempt during Stage 0.

```json
[
    {"vid": "bP5KfdFJzC4_660.0_810.0", "status": "ok"},
    {"vid": "XYZ_0.0_60.0",            "status": "fail"},
    {"vid": "ABC_30.0_120.0",           "status": "timeout"}
]
```

| Status | Meaning |
|---|---|
| `ok` | Video downloaded and frame extracted successfully |
| `fail` | Video unavailable (deleted, private, or geo-blocked) |
| `timeout` | Download exceeded the time limit |

The download log is used in Stage 2 (cell A2) to check whether failed downloads are biased toward any saliency range or video category.

---

## Saliency Score Reference

Saliency scores are assigned by three independent annotators per 2-second clip.

| Score | Label |
|---|---|
| 4 | Very Good |
| 3 | Good |
| 2 | Fair |
| 1 | Bad |
| 0 | Very Bad |

The `mean_saliency` field in `m2_train_pairs.jsonl` is the mean of all per-annotator, per-clip scores across the full `relevant_windows` of a query. Higher values indicate moments that annotators agree are visually salient and well-aligned with the query.

---

## Additional Source Files

The QVHighlights repository also provides a weakly supervised pre-training file:

- **`subs_train.jsonl`** — Subtitle file for ASR-based weakly supervised pre-training. Same format as the annotation files but without `saliency_scores`. Not used in this project unless pre-training M2 on ASR data.

---

## Citation

If you use QVHighlights in your work, please cite the original paper:

```bibtex
@inproceedings{lei2021detecting,
    title={Detecting Moments and Highlights in Videos via Natural Language Queries},
    author={Lei, Jie and Berg, Tamara L and Bansal, Mohit},
    booktitle={Advances in Neural Information Processing Systems},
    year={2021}
}
```
