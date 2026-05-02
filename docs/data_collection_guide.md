# Stage 0 — Data Collection & Formatting Guide

## Objective
Define a standardized pipeline for collecting, processing, and storing raw video data to ensure consistency across the dataset.

This document focuses only on data acquisition and formatting. Any inconsistency here will propagate through the entire pipeline.

---

## 1. Data Sources

### Allowed Sources
- YouTube videos via yt-dlp
- Public datasets: YouTube-8M, ActivityNet, QVHighlights (if thumbnails available)

### Hard Requirements Per Video
- Clear visual content (not slideshow, not screen recording)
- Spoken language (for transcription — no music-only videos)
- A title
- A publicly available thumbnail
- Duration: 60 seconds minimum, 15 minutes maximum
- Resolution: 720p or higher
- Language: English only (Whisper works on all languages but keep consistent for one project)

### Recommended Category Distribution
Collect roughly equal numbers across categories to avoid bias:

| Category | Why include |
|---|---|
| Cooking / food | High thumbnail-content alignment expected |
| Gaming | Often clickbait thumbnails — important negative examples |
| Education / tutorials | High alignment, clear spoken content |
| Vlogs | Mixed — good variety |
| News / commentary | Often face-heavy thumbnails |
| Sports highlights | Action-heavy, visually distinct |
| Technology reviews | Product-focused, structured |

Aim for at least 100 videos per category. Fewer than 50 per category is not enough diversity for training.

### What to Avoid
- Videos shorter than 60 seconds (not enough transcript content)
- Vertical videos (aspect ratio problems for frame extraction)
- Slideshows or screen recordings (no meaningful visual content)
- Auto-generated captions only (lower transcript quality — prefer creator-uploaded captions or Whisper)
- Videos with no speech at all
- Age-restricted videos (download may fail silently)
- Private or unlisted videos

---

## 2. Video Download

### Tool
```bash
pip install yt-dlp
```

### Download Command
```bash
yt-dlp \
  --format "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height>=720][ext=mp4]" \
  --merge-output-format mp4 \
  --write-info-json \
  --write-thumbnail \
  --convert-thumbnails jpg \
  --output "%(id)s/%(id)s.%(ext)s" \
  <VIDEO_URL>
```

### What `--write-info-json` gives you (important — do not skip)
The info JSON contains fields you will need later:
- `id` — YouTube video ID (use as your video identifier)
- `title` — raw title
- `description` — full description text
- `categories` — YouTube category
- `tags` — creator tags
- `duration` — video length in seconds
- `view_count`, `like_count` — engagement metadata
- `upload_date` — date string
- `channel_id`, `channel` — creator identity
- `thumbnail` — thumbnail URL used

Save the full info JSON. Do not discard it. You may need description and tags later as additional text signals.

### Output Format
- Video: mp4, H.264
- Thumbnail: jpg (highest resolution available — yt-dlp fetches maxresdefault automatically)

---

## 3. Metadata Extraction

### From the info JSON, extract and save separately:

**title.txt**
```
How to cook pasta in 10 minutes
```

**description.txt**
```
Full description text exactly as written by creator.
Keep line breaks. Do not truncate.
```

**meta.json**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "How to cook pasta in 10 minutes",
  "duration_seconds": 612,
  "upload_date": "20231015",
  "channel_id": "UCxxxxxx",
  "channel_name": "Cooking With John",
  "category": "Howto & Style",
  "tags": ["pasta", "cooking", "recipe"],
  "view_count": 142000,
  "like_count": 8300,
  "language": "en",
  "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"
}
```

The `meta.json` is your index file per video. Always generate it during collection, not later.

---

## 4. Frame Extraction

### Command
```bash
ffmpeg -i input.mp4 -vf fps=1 -q:v 2 frames/frame_%05d.jpg
```

### Parameters
- FPS: 1 (default — sufficient for most videos)
- Quality: `-q:v 2` gives high quality JPEG (scale 1–31, lower is better)
- Zero-padded 5-digit index: `frame_00001.jpg` supports up to 99,999 frames

### Why 5 digits not 4
A 15-minute video at 1fps = 900 frames. At 2fps = 1800 frames. Use 5 digits to be safe across all video lengths.

### Frame-Timestamp Mapping (CRITICAL — do not skip)
You need to know which frame corresponds to which second of the video. Generate a `frames_index.json`:

```json
[
  {"frame_file": "frame_00001.jpg", "timestamp_sec": 1.0},
  {"frame_file": "frame_00002.jpg", "timestamp_sec": 2.0},
  {"frame_file": "frame_00003.jpg", "timestamp_sec": 3.0}
]
```

This mapping is essential for aligning frames with transcript segments in Stage 1. Without it you cannot create temporally aligned (frame, transcript) pairs.

Generate it with:
```python
import json

fps = 1
n_frames = 900  # replace with actual count
index = [
    {"frame_file": f"frame_{i+1:05d}.jpg", "timestamp_sec": float(i + 1)}
    for i in range(n_frames)
]
with open("frames_index.json", "w") as f:
    json.dump(index, f, indent=2)
```

### Frame Filtering (apply after extraction, before storage)
Remove frames that are clearly unusable:
- **Blurry frames:** compute Laplacian variance — discard if below threshold (e.g. < 100)
- **Dark frames:** compute mean pixel brightness — discard if mean < 30
- **Near-duplicate frames:** compare consecutive frames with perceptual hash — discard if similarity > 0.95

Do not overfilter. Keep a record of which frames were discarded and why in a `frames_discarded.json` file. You may need them later.

---

## 5. Audio Extraction

### Command
```bash
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
```

### Parameters
- `-ar 16000` — 16kHz sample rate (required by Whisper)
- `-ac 1` — mono channel (Whisper expects mono)
- `-acodec pcm_s16le` — uncompressed WAV

### Why 16kHz mono matters
Whisper internally resamples to 16kHz mono. If you store at 44kHz stereo and pass it directly, it works but wastes disk space and processing time. Store at 16kHz mono from the start.

---

## 6. Transcription

### Tool
```bash
pip install openai-whisper
```

### Command
```python
import whisper, json

model = whisper.load_model("medium")  # or "large-v2" for better accuracy
result = model.transcribe("audio.wav", language="en", word_timestamps=True)

with open("transcript.json", "w") as f:
    json.dump(result["segments"], f, indent=2)
```

### Output Format — transcript.json
```json
[
  {
    "id": 0,
    "start": 0.0,
    "end": 3.2,
    "text": " Welcome to my channel.",
    "words": [
      {"word": "Welcome", "start": 0.1, "end": 0.5, "probability": 0.98},
      {"word": "to", "start": 0.5, "end": 0.7, "probability": 0.99}
    ]
  }
]
```

### Why word-level timestamps matter
For Stage 1, you need to align transcript segments with specific frames. Word-level timestamps let you build precise `(frame_t, transcript_window_around_t)` pairs. Without them you are limited to segment-level alignment which is coarser.

### Whisper model choice
| Model | Speed | Accuracy | Recommended for |
|---|---|---|---|
| tiny | fastest | low | testing pipeline only |
| base | fast | acceptable | quick experiments |
| medium | moderate | good | default choice |
| large-v2 | slow | best | final dataset |

Use `medium` during development. Switch to `large-v2` for final data collection.

### Quality check
After transcription, verify:
- Total transcript duration ≈ video duration (within 5%)
- No empty segments
- Confidence scores reasonable (Whisper does not expose per-segment confidence directly, but word probabilities are available with `word_timestamps=True`)

---

## 7. Thumbnail

### Download
yt-dlp with `--write-thumbnail --convert-thumbnails jpg` handles this automatically at highest resolution.

### Manual download if needed
```python
import requests

url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
response = requests.get(url)
if response.status_code == 200:
    with open("thumbnail.jpg", "wb") as f:
        f.write(response.content)
# fallback if maxresdefault not available
else:
    url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
```

### Additional thumbnail metadata to record
In `meta.json`, also record:
```json
{
  "thumbnail_resolution": [1280, 720],
  "thumbnail_has_text": null,
  "thumbnail_has_face": null,
  "thumbnail_type": null
}
```

Leave `null` for now — these fields will be filled during annotation. They are placeholders for your Stage 1 labels.

---

## 8. Manual Annotation Fields (for Stage 1 labels)

This section defines the fields you will fill manually for the labeled subset (300–500 videos recommended).

Create a `annotation.json` per video:
```json
{
  "video_id": "dQw4w9WgXcQ",
  "annotator": "your_name",
  "annotation_date": "2024-01-15",
  "thumbnail_label": "representative",
  "thumbnail_label_confidence": "high",
  "notes": "thumbnail shows the finished dish, matches spoken content"
}
```

### Label options
| Label | Meaning |
|---|---|
| `representative` | Thumbnail genuinely shows what the video is about |
| `clickbait` | Thumbnail is designed for clicks, misleads about content |
| `partial` | Thumbnail is related but exaggerated or out of context |
| `unclear` | Cannot determine from available information |

### Confidence options
- `high` — obvious case, annotator is certain
- `medium` — requires judgment, annotator is fairly sure
- `low` — genuinely ambiguous

**Only use `high` and `medium` confidence labels for training. Use `low` confidence labels only for analysis.**

Annotate at least:
- 150 `representative` videos
- 150 `clickbait` videos
- As many `partial` as naturally occur

This labeled subset becomes your test set and your active learning seed.

---

## 9. Final Folder Structure

```
dataset/
  raw/
    dQw4w9WgXcQ/
      dQw4w9WgXcQ.mp4
      thumbnail.jpg
      title.txt
      description.txt
      meta.json
      annotation.json          ← only for manually labeled videos
      audio.wav
      transcript.json
      frames/
        frame_00001.jpg
        frame_00002.jpg
        ...
      frames_index.json         ← timestamp mapping, REQUIRED
      frames_discarded.json     ← record of filtered frames
  processed/                    ← generated during training, not during collection
  splits/
    train.txt                   ← list of video IDs for training
    val.txt
    test.txt
    labeled.txt                 ← IDs with annotation.json
```

---

## 10. Dataset Splits

Define splits before training, not during. Store as plain text files with one video ID per line.

### Split rules
- Split at the **video level** not the frame level — all frames from one video go to the same split
- Test set must contain only manually annotated videos
- Validation set should contain a mix of annotated and unannotated videos
- Never let the same YouTube channel appear in both train and test — this prevents the model from memorizing creator style

### Recommended split
| Split | Size | Contents |
|---|---|---|
| train | 70% | unannotated + annotated (non-test) |
| val | 10% | mix |
| test | 20% | manually annotated only |
| labeled | variable | all videos with annotation.json |

---

## 11. Data Validation Checklist

Run this check on every video before adding to the dataset:

### File existence
- [ ] `meta.json` exists and is valid JSON
- [ ] `thumbnail.jpg` exists and is non-empty
- [ ] `title.txt` exists and is non-empty
- [ ] `description.txt` exists (can be empty string if no description)
- [ ] `audio.wav` exists and is playable
- [ ] `transcript.json` exists and contains at least 3 segments
- [ ] `frames/` directory contains at least 30 frames
- [ ] `frames_index.json` exists and length matches frame count

### Content quality
- [ ] Video duration between 60s and 900s
- [ ] Thumbnail resolution at least 640×360
- [ ] Transcript total duration within 10% of video duration
- [ ] Frame filenames are sequentially numbered with no gaps
- [ ] At least one transcript segment has confidence > 0.8

### Alignment check (critical for Stage 1)
- [ ] `frames_index.json` timestamps are monotonically increasing
- [ ] Last frame timestamp ≤ video duration
- [ ] At least one transcript segment overlaps with each 10-second window of the video

---

## 12. Naming and IDs

Use the YouTube video ID as the folder name directly:
```
dQw4w9WgXcQ/     ← YouTube ID, globally unique, no collisions
```

Do not invent your own sequential IDs. YouTube IDs are already unique, collision-free, and let you trace any sample back to its source instantly.

If you use non-YouTube sources, prefix with source:
```
yt_dQw4w9WgXcQ/
vimeo_123456789/
```

---

## 13. Scale Recommendation

| Purpose | Minimum | Recommended |
|---|---|---|
| Pipeline testing | 50 videos | — |
| Stage 1 development | 500 videos | 1,000+ |
| Stage 1 training | 2,000 videos | 5,000+ |
| Manual annotation (labels) | 300 videos | 500 |
| Final evaluation test set | 100 videos | 200 |

Start with 100 videos to validate the full pipeline end-to-end before collecting at scale.

---

## 14. Common Pitfalls

| Pitfall | Consequence | Prevention |
|---|---|---|
| Missing `frames_index.json` | Cannot align frames with transcript | Generate immediately after frame extraction |
| Storing audio at 44kHz stereo | Wasted disk space, slower processing | Always extract at 16kHz mono |
| Using frame index as timestamp | Wrong alignment if extraction failed partway | Use actual ffmpeg timestamps |
| Mixing video languages | Whisper accuracy drops, embeddings misalign | English only for one project |
| Same channel in train and test | Model memorizes creator style, not content | Split by channel not by video |
| Discarding frames without logging | Cannot debug alignment issues | Always write `frames_discarded.json` |
| Low-resolution thumbnails | Degraded visual features | Always fetch maxresdefault |
| Missing description | Loses useful text signal | Store even if empty string |
| Sequential IDs instead of YouTube IDs | Cannot trace samples back to source | Use YouTube ID as folder name |
