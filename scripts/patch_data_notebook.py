"""
Patch notebooks/data_collection_preparation.ipynb to match FINAL_DESIGN.md.

What this script does (idempotent):
  1. Apply bug fixes B1..B6 from FINAL_DESIGN section 5.
  2. Insert new cells for:
       - YouTube thumbnail download (yt-dlp --write-thumbnail)
       - M1 official-format subset JSONL builder
       - M2 val pairs builder (~300 from official val)
       - consistency_eval.jsonl builder (positives + hard negatives)
       - m2_image_emb.npy cache for thumbnails / frames
       - category_map.json export
  3. Ensure clip_score is written on BOTH first-extract and cached paths.

Run:
    python scripts/patch_data_notebook.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from copy import deepcopy

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "data_collection_preparation.ipynb"
MARK = "# [PIPELINE_PATCH_v1]"                       # v1 sentinel (bug fixes + new cells)
MARK_V2 = "# [PIPELINE_PATCH_v2_center]"             # v2 sentinel (center-frame, no CLIP)
MARK_V3 = "# [PIPELINE_PATCH_v3_principled]"         # v3 sentinel (annotator-agreement selection)
MARK_V4 = "# [PIPELINE_PATCH_v4_direct_thumbnails]"  # v4 sentinel (direct-HTTP thumbnail download)
MARK_V5 = "# [PIPELINE_PATCH_v5_safe_encode]"        # v5 sentinel (defensive get_text_features)
MARK_V6 = "# [PIPELINE_PATCH_v6_bypass_get_features]"# v6 sentinel (bypass get_image_features)
MARK_V7 = "# [PIPELINE_PATCH_v7_selfcontained_m2_val]"  # v7 sentinel (self-contained m2 val builder)
MARK_V8 = "# [PIPELINE_PATCH_v8_unpack_features]"    # v8 sentinel (unpack Moment-DETR features)
MARK_V9 = "# [PIPELINE_PATCH_v9_multiframe_aug]"     # v9 sentinel (K-frame augmentation for M2)
MARK_V10 = "# [PIPELINE_PATCH_v10_taxonomy]"         # v10 sentinel (data taxonomy markdown)
MARK_V11 = "# [PIPELINE_PATCH_v11_reconcile]"        # v11 sentinel (reconcile v1 frames, no re-extract)
MARK_V12 = "# [PIPELINE_PATCH_v12_gold_set]"         # v12 sentinel (manual gold set builder)
MARK_V13 = "# [PIPELINE_PATCH_v13_gold_eval]"        # v13 sentinel (gold-set pipeline test)
MARK_V14 = "# [PIPELINE_PATCH_v14_hard_qmiss]"       # v14 sentinel (hard + query-mismatch builders)


# ---------- helpers ----------------------------------------------------------

def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def cell_src(cell) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, src: str) -> None:
    cell["source"] = src.splitlines(keepends=True)


def find_cell(nb, predicate):
    for i, c in enumerate(nb["cells"]):
        if predicate(c):
            return i
    return -1


# ---------- new cell contents ------------------------------------------------

CELL_LOAD_JSONL = f'''{MARK}
def load_jsonl(path):
    """Load JSONL with explicit utf-8 (B1)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# Load all three splits
train_raw = load_jsonl(DATA_DIR / "highlight_train_release.jsonl")
val_raw   = load_jsonl(DATA_DIR / "highlight_val_release.jsonl")
test_raw  = load_jsonl(DATA_DIR / "highlight_test_release.jsonl")

print(f"Train samples : {{len(train_raw):,}}")
print(f"Val samples   : {{len(val_raw):,}}")
print(f"Test samples  : {{len(test_raw):,}}")
print(f"Total         : {{len(train_raw)+len(val_raw)+len(test_raw):,}}")
'''


CELL_SAVE_JSONL = f'''{MARK}
def save_jsonl(records, path):
    """Save JSONL with explicit utf-8 + non-ASCII safe (B1)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print(f"Saved {{len(records):,}} records -> {{path}}")

save_jsonl(train_subset, PROCESSED_DIR / "train_processed.jsonl")
save_jsonl(val_proc,     PROCESSED_DIR / "val_processed.jsonl")
save_jsonl(test_proc,    PROCESSED_DIR / "test_processed.jsonl")
save_jsonl(train_proc,   PROCESSED_DIR / "train_full_processed.jsonl")

print("\\nProcessed files saved to Drive.")
'''


CELL_INFER_CATEGORY = f'''{MARK}
# Category inference with explicit vlog bucket + "other" fallback (B2).
# Avoids over-counting daily_vlog when no keyword matches.

TRAVEL_KEYWORDS = {{"beach","hotel","tour","plane","swim","visit","travel","ocean",
                   "mountain","temple","city","airport","restaurant","street","market",
                   "trip","resort","cruise","sightseeing","tourist"}}
NEWS_KEYWORDS   = {{"protest","police","report","interview","press","fire","flood",
                   "hurricane","crowd","rally","march","officer","military",
                   "election","government","news","reporter","anchor"}}
VLOG_KEYWORDS   = {{"camera","selfie","vlog","talks","shows","explains","unboxing",
                   "kitchen","cooking","makeup","family","baby","dog","cat","bedroom"}}

def infer_category(query):
    words = set(query.lower().split())
    if words & NEWS_KEYWORDS:   return "news"
    if words & TRAVEL_KEYWORDS: return "travel_vlog"
    if words & VLOG_KEYWORDS:   return "daily_vlog"
    return "other"

for r in train_proc:
    r["category"] = infer_category(r["query"])

from collections import Counter
cat_counts = Counter(r["category"] for r in train_proc)
print("Category distribution in full train set:")
for cat, cnt in sorted(cat_counts.items()):
    print(f"  {{cat:15s}}: {{cnt:5d}}  ({{cnt/len(train_proc)*100:.1f}}%)")

# Persist mapping qid -> category for downstream eval.
cat_map = {{r["qid"]: r["category"] for r in train_proc}}
with open(PROCESSED_DIR / "category_map.json", "w", encoding="utf-8") as f:
    json.dump(cat_map, f)
print(f"Saved category_map.json ({{len(cat_map)}} entries)")
'''


CELL_DOWNLOAD_CLIP = f'''{MARK}
# yt-dlp downloader with multi-client fallback + stderr logging (B3).

MAX_DOWNLOADS    = 2000   # set lower for a quick test run
PARALLEL_WORKERS = 6

YT_CLIENTS = ["android", "web", "ios"]  # try these in order on failure

def _run_ytdlp(cmd, timeout):
    p = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def download_clip(record):
    vid      = record["vid"]
    yt_id    = record["youtube_id"]
    start    = record["seg_start"]
    end      = record["seg_end"]
    out_path = VIDEO_DIR / f"{{vid}}.mp4"

    if out_path.exists() and out_path.stat().st_size > 10_000:
        record["video_downloaded"] = True
        return {{"vid": vid, "status": "skip", "path": str(out_path), "err": None}}

    url = f"https://www.youtube.com/watch?v={{yt_id}}"
    last_err = None
    for client in YT_CLIENTS:
        cmd = [
            "yt-dlp",
            "--download-sections", f"*{{start}}-{{end}}",
            "--format", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
            "--merge-output-format", "mp4",
            "--extractor-args", f"youtube:player_client={{client}}",
            "--quiet", "--no-warnings",
            "-o", str(out_path),
            url,
        ]
        try:
            rc, _, err = _run_ytdlp(cmd, timeout=180)
            if rc == 0 and out_path.exists():
                record["video_downloaded"] = True
                return {{"vid": vid, "status": "ok", "path": str(out_path),
                        "err": None, "client": client}}
            last_err = (err or "").strip()[-300:]
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            continue
    record["video_downloaded"] = False
    status = "timeout" if last_err == "timeout" else "fail"
    return {{"vid": vid, "status": status, "path": None, "err": last_err}}

print(f"Downloading up to {{MAX_DOWNLOADS}} clips with {{PARALLEL_WORKERS}} workers...")

to_download = train_subset[:MAX_DOWNLOADS]
results = {{"ok": 0, "skip": 0, "fail": 0, "timeout": 0}}
download_log = []

with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
    futures = {{executor.submit(download_clip, r): r for r in to_download}}
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        results[result["status"]] = results.get(result["status"], 0) + 1
        download_log.append(result)
        if i % 50 == 0 or i == len(to_download):
            print(f"  [{{i:4d}}/{{len(to_download)}}]  "
                  f"ok={{results['ok']}}  skip={{results['skip']}}  "
                  f"fail={{results['fail']}}  timeout={{results['timeout']}}")

with open(PROCESSED_DIR / "download_log.json", "w", encoding="utf-8") as f:
    json.dump(download_log, f, indent=2)

print(f"\\nDownload complete: ok={{results['ok']}}, skip={{results['skip']}}, "
      f"fail={{results['fail']}}, timeout={{results['timeout']}}")
print("Failure samples (last 5):")
for r in [r for r in download_log if r["status"] in ("fail", "timeout")][:5]:
    print(f"  {{r['vid']}}: {{r['err']}}")
'''


CELL_EXTRACT_FRAME = f'''{MARK}
# Robust frame extraction: VFR-safe seek (B4) + always write clip_score (B5)
# + accurate counter (B6).

import cv2
from PIL import Image

N_CANDIDATES = 10  # frames sampled across the 2-second best clip

def _seek_msec(cap, ms):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ms))
    ret, frame = cap.read()
    return ret, frame

def extract_frame(record):
    """
    Pipeline
    --------
    1. Seek N_CANDIDATES frames across the 2-second best clip BY MSEC (VFR-safe).
    2. Score against query with CLIP ViT-B/32.
    3. Save best as {{vid}}.jpg (224 x 224).
    4. Refresh clip_score on EVERY call (also on cached path).
    """
    vid        = record["vid"]
    query      = record["query"]
    best_cid   = record.get("best_clip_id")
    video_path = VIDEO_DIR / f"{{vid}}.mp4"
    frame_out  = FRAMES_DIR / f"{{vid}}.jpg"

    if best_cid is None:
        record["video_downloaded"] = False
        record["clip_score"] = None
        return record

    if not video_path.exists():
        record["video_downloaded"] = False
        record["clip_score"] = None
        return record

    if frame_out.exists():
        # Cached frame: still re-run CLIP scoring so clip_score is always populated.
        try:
            pil = Image.open(frame_out).convert("RGB")
            model, preprocess, device = _get_clip()
            with _clip_lock:
                score = float(clip_rank_frames([pil], query, model, preprocess, device)[0])
            record["clip_score"] = round(score, 4)
        except Exception:
            record["clip_score"] = record.get("clip_score")  # leave as-is
        record["video_downloaded"] = True
        record["frame_path"]       = str(frame_out)
        return record

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        record["video_downloaded"] = False
        record["clip_score"] = None
        return record

    clip_start_ms = best_cid * 2_000
    clip_end_ms   = clip_start_ms + 2_000
    sample_ms     = np.linspace(clip_start_ms, clip_end_ms - 1,
                                N_CANDIDATES, dtype=int)

    frames_pil, timestamps = [], []
    for ms in sample_ms:
        ret, frame = _seek_msec(cap, int(ms))
        if ret and frame is not None:
            frames_pil.append(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            timestamps.append(float(ms) / 1000.0)
    cap.release()

    if not frames_pil:
        record["video_downloaded"] = False
        record["clip_score"] = None
        return record

    try:
        model, preprocess, device = _get_clip()
        with _clip_lock:
            scores = clip_rank_frames(frames_pil, query, model, preprocess, device)
        best_i      = int(np.argmax(scores))
        best_pil    = frames_pil[best_i]
        clip_score  = float(scores[best_i])
        selected_ts = timestamps[best_i]
    except Exception:
        mid         = len(frames_pil) // 2
        best_pil    = frames_pil[mid]
        clip_score  = None
        selected_ts = timestamps[mid]

    best_pil.resize((224, 224), Image.LANCZOS).save(str(frame_out), quality=92)

    record["video_downloaded"] = True
    record["frame_path"]       = str(frame_out)
    record["clip_score"]       = (round(clip_score, 4)
                                  if clip_score is not None else None)
    record["best_timestamp"]   = round(selected_ts, 3)
    return record


print("Pre-loading CLIP model...")
_get_clip()
print("Extracting best frames with CLIP ranking (VFR-safe)...")
updated_records = []
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {{executor.submit(extract_frame, r): r for r in train_subset}}
    for i, future in enumerate(as_completed(futures), 1):
        updated_records.append(future.result())
        if i % 200 == 0 or i == len(train_subset):
            with_score = sum(1 for r in updated_records
                             if r.get("frame_path") and r.get("clip_score") is not None)
            print(f"  [{{i:4d}}/{{len(train_subset)}}]  frames-with-score: {{with_score}}")

updated_records.sort(key=lambda x: x["qid"])
train_subset = updated_records

n_with_frame = sum(1 for r in train_subset if r.get("frame_path"))
scores_all   = [r["clip_score"] for r in train_subset if r.get("clip_score") is not None]
print(f"\\nFrames extracted  : {{n_with_frame}} / {{len(train_subset)}}")
if scores_all:
    print(f"CLIP score - mean : {{sum(scores_all)/len(scores_all):.4f}}")
    print(f"CLIP score - min  : {{min(scores_all):.4f}}   max: {{max(scores_all):.4f}}")
    below = sum(1 for s in scores_all if s < 0.22)
    print(f"Low-quality (< 0.22): {{below}}  ({{100*below/len(scores_all):.1f}}%)")
'''


# ---------- NEW cells (appended) --------------------------------------------

CELL_THUMBNAIL_DL_MD = f'''{MARK}
# 13. Download Original YouTube Thumbnails (eval-time domain-gap probe)

These are **not** used for training M2. They are saved as metadata for evaluating
the domain gap between extracted frames and real stylized thumbnails (per
FINAL_DESIGN section 2, Case D-light).

Output: `qvh_thumbnails/{{youtube_id}}.jpg` and `thumbnail_index.jsonl`.
'''

CELL_THUMBNAIL_DL = f'''{MARK}
# Download real YouTube thumbnails for the subset (cheap: ~2 min total).

THUMB_DIR = PROJECT_ROOT / "qvh_thumbnails"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

def download_thumbnail(record):
    yt_id    = record["youtube_id"]
    out_jpg  = THUMB_DIR / f"{{yt_id}}.jpg"
    if out_jpg.exists() and out_jpg.stat().st_size > 1000:
        return {{"yt_id": yt_id, "status": "skip", "path": str(out_jpg)}}

    url = f"https://www.youtube.com/watch?v={{yt_id}}"
    cmd = [
        "yt-dlp", "--skip-download", "--write-thumbnail",
        "--convert-thumbnails", "jpg",
        "--extractor-args", "youtube:player_client=web",
        "-o", str(THUMB_DIR / f"{{yt_id}}.%(ext)s"),
        "--quiet", "--no-warnings",
        url,
    ]
    try:
        p = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
        if out_jpg.exists():
            return {{"yt_id": yt_id, "status": "ok", "path": str(out_jpg)}}
        return {{"yt_id": yt_id, "status": "fail",
                "err": (p.stderr or "")[-200:]}}
    except subprocess.TimeoutExpired:
        return {{"yt_id": yt_id, "status": "timeout"}}

# Deduplicate by youtube_id (one thumbnail per YT video).
unique_yt = {{r["youtube_id"]: r for r in train_subset}}.values()
print(f"Downloading thumbnails for {{len(list(unique_yt))}} unique YouTube videos...")
unique_yt = list({{r["youtube_id"]: r for r in train_subset}}.values())

thumb_results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {{ex.submit(download_thumbnail, r): r for r in unique_yt}}
    for i, fut in enumerate(as_completed(futures), 1):
        thumb_results.append(fut.result())
        if i % 100 == 0 or i == len(unique_yt):
            ok = sum(1 for r in thumb_results if r["status"] in ("ok", "skip"))
            print(f"  [{{i:4d}}/{{len(unique_yt)}}]  ok={{ok}}")

# Build thumbnail_index.jsonl keyed by qid (per-record path, not per yt_id,
# because evaluation iterates records).
index = []
for r in train_subset:
    p = THUMB_DIR / f"{{r['youtube_id']}}.jpg"
    if p.exists():
        index.append({{"qid": r["qid"], "vid": r["vid"],
                      "youtube_id": r["youtube_id"],
                      "thumbnail_path": str(p)}})

with open(PROCESSED_DIR / "thumbnail_index.jsonl", "w", encoding="utf-8") as f:
    for x in index:
        f.write(json.dumps(x, ensure_ascii=False) + "\\n")

print(f"\\nThumbnails available for {{len(index)}} / {{len(train_subset)}} records.")
'''


CELL_M1_SUBSET_MD = f'''{MARK}
# 14. Build M1 Training Subset in Official QVHighlights Format

Moment-DETR's loader expects the **original** QVH schema (not our processed one).
We emit `m1_train_subset.jsonl` by filtering the official train file using the
qids in our 2000-sample subset. We also copy `m1_val.jsonl` verbatim.

This file is consumed directly by `train.py --train_path` if we ever retrain.
For V1 we use the official checkpoint, but this file is still needed for
feature-coverage audits.
'''

CELL_M1_SUBSET = f'''{MARK}
# Filter official train JSONL by our subset qids -> m1_train_subset.jsonl.

subset_qids = {{r["qid"] for r in train_subset}}
m1_train = [r for r in train_raw if r["qid"] in subset_qids]

save_jsonl(m1_train, PROCESSED_DIR / "m1_train_subset.jsonl")
save_jsonl(val_raw,  PROCESSED_DIR / "m1_val.jsonl")

print(f"M1 train subset: {{len(m1_train)}} records (official schema)")
print(f"M1 val full    : {{len(val_raw)}} records (official schema)")

# Feature-coverage audit (R3 mitigation in FINAL_DESIGN section 9).
# We expect the official feature dir layout under REPO_DIR/features/clip_features.
feat_dir = REPO_DIR / "features" / "clip_features"
if feat_dir.exists():
    have = {{p.stem for p in feat_dir.glob("*.npz")}}
    missing = [r["vid"] for r in m1_train if r["vid"] not in have]
    print(f"Feature files present: {{len(have)}}")
    print(f"Subset records missing features: {{len(missing)}}")
    if missing[:3]:
        print(f"  examples: {{missing[:3]}}")
else:
    print("(Feature dir not unpacked yet; run feature unpack before training audit.)")
'''


CELL_M2_VAL_MD = f'''{MARK}
# 15. Build M2 Validation Pairs (300 from official val)

M2 currently has no validation split. Without one, we cannot early-stop on
CLIP-text similarity. This cell extracts frames for ~300 val records using the
same `extract_frame` pipeline, producing `m2_val_pairs.jsonl`.

Held out separately so it never leaks into M2 training.
'''

CELL_M2_VAL = f'''{MARK}
# Sample 300 val records (stratified by mean_saliency tertile) and extract frames.

M2_VAL_N = 300
random.seed(2026)

# Process val records to add best_clip_id / best_timestamp.
val_proc_full = val_proc  # already processed above
val_with_sal  = [r for r in val_proc_full if r.get("mean_saliency") is not None]

# Tertile-stratified pick so we cover low/mid/high saliency.
val_with_sal.sort(key=lambda r: r["mean_saliency"])
tertile = len(val_with_sal) // 3
buckets = [val_with_sal[:tertile],
           val_with_sal[tertile:2*tertile],
           val_with_sal[2*tertile:]]
per_bucket = M2_VAL_N // 3
val_pick = []
for b in buckets:
    val_pick.extend(random.sample(b, min(per_bucket, len(b))))
print(f"Selected {{len(val_pick)}} val records for M2 evaluation pairs.")

# Download those val clips (reuse download_clip).
print("Downloading val clips for M2 frame extraction...")
val_dl_results = []
with ThreadPoolExecutor(max_workers=6) as ex:
    futures = {{ex.submit(download_clip, r): r for r in val_pick}}
    for i, fut in enumerate(as_completed(futures), 1):
        val_dl_results.append(fut.result())
        if i % 50 == 0 or i == len(val_pick):
            ok = sum(1 for r in val_dl_results if r["status"] in ("ok", "skip"))
            print(f"  [{{i:4d}}/{{len(val_pick)}}]  ok={{ok}}")

# Extract frames for downloaded val records.
print("\\nExtracting M2 val frames...")
val_pick_extracted = []
with ThreadPoolExecutor(max_workers=6) as ex:
    futures = {{ex.submit(extract_frame, r): r for r in val_pick}}
    for i, fut in enumerate(as_completed(futures), 1):
        val_pick_extracted.append(fut.result())

m2_val_pairs = [
    {{"qid": r["qid"], "vid": r["vid"],
     "frame_path": r["frame_path"], "query": r["query"],
     "best_timestamp": r["best_timestamp"],
     "mean_saliency": r["mean_saliency"],
     "clip_score": r.get("clip_score")}}
    for r in val_pick_extracted if r.get("frame_path")
]

save_jsonl(m2_val_pairs, PROCESSED_DIR / "m2_val_pairs.jsonl")
print(f"\\nM2 val pairs: {{len(m2_val_pairs)}}")
'''


CELL_CONSISTENCY_EVAL_MD = f'''{MARK}
# 16. Build Consistency Evaluation Set (positives + hard negatives)

This is the **primary evaluation artifact** for the pipeline. Per FINAL_DESIGN
section 5 and the score in section 7:

- **Positive pair**: (frame_i, vid_i) -- the frame came from this video.
- **Hard negative**: (frame_i, vid_j) where i != j and the queries are
  semantically dissimilar (CLIP-text cosine < 0.6) so the negative isn't
  accidentally consistent.

Output: `consistency_eval.jsonl` with ~300 positives + ~300 hard negatives.
'''

CELL_CONSISTENCY_EVAL = f'''{MARK}
# Build 300 positives + 300 hard negatives using CLIP-text dissimilarity.

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

random.seed(2026)
np.random.seed(2026)

eval_pool = [p for p in m2_pairs
             if p.get("frame_path") and p.get("clip_score") is not None]
print(f"Eligible eval pool: {{len(eval_pool)}}")

N_POS = min(300, len(eval_pool))
positives = random.sample(eval_pool, N_POS)

# Encode all queries in the eval_pool with CLIP-text for dissimilarity filtering.
print("Encoding queries with CLIP-text...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_hf = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def encode_texts(texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        toks = clip_tok(texts[i:i+batch], padding=True, truncation=True,
                        return_tensors="pt").to(device)
        emb = clip_hf.get_text_features(**toks)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)

queries = [p["query"] for p in eval_pool]
q_emb   = encode_texts(queries)
qid_to_idx = {{p["qid"]: i for i, p in enumerate(eval_pool)}}

DISSIM_MAX = 0.60  # negatives must satisfy cos(q_i, q_j) < this

negatives = []
for p in positives:
    i = qid_to_idx[p["qid"]]
    sims = q_emb @ q_emb[i]
    candidates = [j for j, s in enumerate(sims)
                  if j != i and s < DISSIM_MAX
                  and eval_pool[j]["vid"] != p["vid"]]
    if not candidates:
        continue
    j = random.choice(candidates)
    neg_video = eval_pool[j]
    negatives.append({{
        "pair_id"   : f"neg_{{p['qid']}}_{{neg_video['qid']}}",
        "label"     : 0,
        "thumb_qid" : p["qid"],
        "thumb_vid" : p["vid"],
        "frame_path": p["frame_path"],
        "thumb_query": p["query"],
        "video_qid" : neg_video["qid"],
        "video_vid" : neg_video["vid"],
        "video_query_for_ref": neg_video["query"],
        "neg_query_cos": float(sims[j]),
    }})

positive_pairs = [{{
    "pair_id"   : f"pos_{{p['qid']}}",
    "label"     : 1,
    "thumb_qid" : p["qid"],
    "thumb_vid" : p["vid"],
    "frame_path": p["frame_path"],
    "thumb_query": p["query"],
    "video_qid" : p["qid"],
    "video_vid" : p["vid"],
    "video_query_for_ref": p["query"],
}} for p in positives]

consistency_eval = positive_pairs + negatives
random.shuffle(consistency_eval)

save_jsonl(consistency_eval, PROCESSED_DIR / "consistency_eval.jsonl")
print(f"\\nConsistency eval: {{len(positive_pairs)}} positives + "
      f"{{len(negatives)}} hard negatives = {{len(consistency_eval)}} pairs")
if negatives:
    cosines = [n["neg_query_cos"] for n in negatives]
    print(f"Negative query-cos range: {{min(cosines):.3f}} .. {{max(cosines):.3f}}")
'''


CELL_IMG_EMB_MD = f'''{MARK}
# 17. Cache CLIP-image embeddings for all frames + thumbnails

The consistency scorer needs `e_T = CLIP_img(thumbnail)` for every eval pair.
Compute once, cache to `.npy`, avoid re-encoding during the evaluation loop
(saves ~30 minutes when iterating on the score formula).
'''

CELL_IMG_EMB = f'''{MARK} {MARK_V5} {MARK_V6}
# Cache CLIP-image embeddings keyed by qid (m2 frames) and youtube_id (thumbs).
# v6: bypass get_image_features entirely -- some transformers versions have a
# broken visual_projection chain inside get_image_features. We call
# vision_model directly and apply visual_projection only when the dimension
# matches its expected input.

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

device    = "cuda" if torch.cuda.is_available() else "cpu"
clip_hf   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

PROJ_DIM      = clip_hf.config.projection_dim              # 512 for ViT-B/32
VISION_HIDDEN = clip_hf.config.vision_config.hidden_size   # 768 for ViT-B/32
print(f"CLIP config -> projection_dim={{PROJ_DIM}}, vision_hidden={{VISION_HIDDEN}}")


@torch.no_grad()
def _vision_pooled(pixel_values):
    """Return post-pooler feature from vision_model, regardless of API shape."""
    out = clip_hf.vision_model(pixel_values=pixel_values)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if isinstance(out, tuple):
        return out[1] if len(out) > 1 else out[0][:, 0]
    raise RuntimeError(f"Unexpected vision_model output: {{type(out)}}")


@torch.no_grad()
def encode_images(paths, batch=32):
    embs, kept = [], []
    for i in range(0, len(paths), batch):
        bp = paths[i:i+batch]
        imgs, ok = [], []
        for p in bp:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(p)
            except Exception:
                continue
        if not imgs:
            continue
        inp    = clip_proc(images=imgs, return_tensors="pt").to(device)
        pooled = _vision_pooled(inp["pixel_values"])

        # Project only if dim matches visual_projection's input.
        if pooled.shape[-1] == VISION_HIDDEN:
            emb = clip_hf.visual_projection(pooled)
        elif pooled.shape[-1] == PROJ_DIM:
            emb = pooled
        else:
            raise RuntimeError(
                f"Pooled feature has unexpected dim {{pooled.shape[-1]}}; "
                f"expected {{VISION_HIDDEN}} or {{PROJ_DIM}}."
            )
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        embs.append(emb.cpu().numpy())
        kept.extend(ok)
    if not embs:
        return np.zeros((0, PROJ_DIM), dtype=np.float32), []
    return np.concatenate(embs, axis=0), kept

# 1) Frames (m2 training pairs)
frame_paths = [p["frame_path"] for p in m2_pairs if p.get("frame_path")]
frame_qids  = [p["qid"]        for p in m2_pairs if p.get("frame_path")]
print(f"Encoding {{len(frame_paths)}} frames...")
frame_emb, kept_paths = encode_images(frame_paths)
kept_set = set(kept_paths)
kept_qids = [q for q, p in zip(frame_qids, frame_paths) if p in kept_set]
np.savez(PROCESSED_DIR / "m2_image_emb.npz",
         qids=np.asarray(kept_qids, dtype=np.int64),
         emb=frame_emb.astype(np.float32))
print(f"  Saved m2_image_emb.npz: {{frame_emb.shape}}")

# 2) Real thumbnails (one per youtube_id) -- only if step 13 ran
THUMB_DIR = PROJECT_ROOT / "qvh_thumbnails"
if THUMB_DIR.exists():
    thumb_paths = sorted(THUMB_DIR.glob("*.jpg"))
    if thumb_paths:
        print(f"Encoding {{len(thumb_paths)}} real thumbnails...")
        thumb_emb, kept_thumb = encode_images([str(p) for p in thumb_paths])
        thumb_yt = [Path(p).stem for p in kept_thumb]
        np.savez(PROCESSED_DIR / "thumbnail_image_emb.npz",
                 youtube_ids=np.asarray(thumb_yt),
                 emb=thumb_emb.astype(np.float32))
        print(f"  Saved thumbnail_image_emb.npz: {{thumb_emb.shape}}")
    else:
        print("(No thumbnails in qvh_thumbnails/ -- skip)")
else:
    print("(qvh_thumbnails/ does not exist -- skip)")
'''


CELL_FINAL_SUMMARY_MD = f'''{MARK}
# 18. Final Dataset Manifest

All artifacts produced by this notebook (after the pipeline patch):

| File | Purpose |
|---|---|
| `m1_train_subset.jsonl` | 2000-sample QVH-format subset (for M1 audits / optional retrain) |
| `m1_val.jsonl` | Full official val (for M1 eval) |
| `m2_train_pairs.jsonl` | ~1608 (frame, query) pairs for ClipCap training |
| `m2_val_pairs.jsonl` | ~300 (frame, query) pairs from val (for early stopping) |
| `consistency_eval.jsonl` | ~600 (positive + hard negative) pairs (primary eval) |
| `thumbnail_index.jsonl` | Real-YouTube-thumbnail paths keyed by qid |
| `m2_image_emb.npz` | Cached CLIP-image embeddings for frames |
| `thumbnail_image_emb.npz` | Cached CLIP-image embeddings for real thumbnails |
| `category_map.json` | qid -> category for stratification |
| `download_log.json` | yt-dlp results (debug) |
'''


CELL_FINAL_SUMMARY = f'''{MARK}
# Print final manifest summary.

import os

manifest = [
    "m1_train_subset.jsonl",
    "m1_val.jsonl",
    "m2_train_pairs.jsonl",
    "m2_val_pairs.jsonl",
    "consistency_eval.jsonl",
    "thumbnail_index.jsonl",
    "m2_image_emb.npz",
    "thumbnail_image_emb.npz",
    "category_map.json",
    "download_log.json",
]

print(f"{{'File':40s}}  {{'Status':8s}}  {{'Size':>12s}}")
print("-" * 64)
for name in manifest:
    p = PROCESSED_DIR / name
    if p.exists():
        size = p.stat().st_size
        size_s = f"{{size/1024:.1f}} KB" if size < 1e6 else f"{{size/1e6:.1f}} MB"
        print(f"{{name:40s}}  {{'OK':8s}}  {{size_s:>12s}}")
    else:
        print(f"{{name:40s}}  {{'MISSING':8s}}  {{'-':>12s}}")
'''


# ---------- v2 cells (center-frame, no CLIP) --------------------------------

CELL_SELECT_BEST_FRAME_V2 = f'''{MARK_V2}
# Tie-aware best-frame selection.
#
# QVHighlights gives us, per training record:
#   relevant_clip_ids : list of 2-second clip indices that fall in the moment
#   saliency_scores   : per-clip list of 3 annotator scores (0..4 each)
#
# When multiple relevant clips share the same maximum mean saliency, we have a
# tie. We resolve it deterministically by picking the MIDDLE of the tied clips
# (by index in the input list). Reasons:
#   - Avoids start/end bias (which "pick the first argmax" would introduce).
#   - For contiguous high-saliency runs (the common case), middle = the most
#     temporally representative frame of the moment.
#   - Reproducible: no randomness involved.
#
# Returns (best_clip_id, best_timestamp_seconds, reason_str).

def select_best_frame(relevant_clip_ids, saliency_scores):
    if not relevant_clip_ids or not saliency_scores:
        return None, None, "no_annotation"

    mean_scores = [sum(s) / max(1, len(s)) for s in saliency_scores]
    max_score = max(mean_scores)
    tied = [cid for cid, sc in zip(relevant_clip_ids, mean_scores)
            if sc == max_score]

    if len(tied) == 1:
        best_clip_id = tied[0]
        reason = f"single_max(score={{max_score:.2f}})"
    else:
        # Middle of the tied list.
        best_clip_id = tied[len(tied) // 2]
        reason = f"tied_{{len(tied)}}_picked_middle(score={{max_score:.2f}})"

    best_timestamp = best_clip_id * 2 + 1.0    # centre of the 2-second clip
    return best_clip_id, best_timestamp, reason
'''


CELL_EXTRACT_FRAME_V2 = f'''{MARK_V2}
# Center-frame extraction. No CLIP, no race condition, deterministic.
#
# Replaces the CLIP-ranked extractor: empirically the CLIP-ranking step added
# little visible quality but introduced a race condition, GPU memory churn, and
# a tunable score threshold. Switching back to the simple center-of-best-clip
# strategy gives equivalent supervision signal at ~10x the throughput.
#
# Side effects (set on the record):
#   frame_path     : str | absent
#   best_timestamp : seconds (centre of the 2-second best clip)
#   clip_score     : None  (kept for schema compatibility downstream)
#   video_downloaded: bool

import cv2
from PIL import Image
from pathlib import Path


def extract_frame(record):
    vid        = record["vid"]
    best_cid   = record.get("best_clip_id")
    video_path = VIDEO_DIR / f"{{vid}}.mp4"
    frame_out  = FRAMES_DIR / f"{{vid}}.jpg"

    # Annotated as "no relevant clip" (e.g. test split) -> nothing to extract.
    if best_cid is None:
        record["video_downloaded"] = video_path.exists()
        record["clip_score"] = None
        return record

    if not video_path.exists():
        record["video_downloaded"] = False
        record["clip_score"] = None
        return record

    # Cached: frame already on disk.
    if frame_out.exists():
        record["video_downloaded"] = True
        record["frame_path"]       = str(frame_out)
        record["clip_score"]       = None
        record["best_timestamp"]   = best_cid * 2 + 1.0
        return record

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        record["video_downloaded"] = False
        record["clip_score"] = None
        return record

    # Seek to centre of the 2-second clip via MSEC (VFR-safe).
    center_ms = best_cid * 2_000 + 1_000
    cap.set(cv2.CAP_PROP_POS_MSEC, float(center_ms))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        record["video_downloaded"] = False
        record["clip_score"] = None
        return record

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil.resize((224, 224), Image.LANCZOS).save(str(frame_out), quality=92)

    record["video_downloaded"] = True
    record["frame_path"]       = str(frame_out)
    record["clip_score"]       = None
    record["best_timestamp"]   = best_cid * 2 + 1.0
    return record


print("Extracting center frames (no CLIP)...")
updated_records = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {{executor.submit(extract_frame, r): r for r in train_subset}}
    for i, future in enumerate(as_completed(futures), 1):
        updated_records.append(future.result())
        if i % 200 == 0 or i == len(train_subset):
            done = sum(1 for r in updated_records if r.get("frame_path"))
            print(f"  [{{i:4d}}/{{len(train_subset)}}]  frames extracted: {{done}}")

updated_records.sort(key=lambda x: x["qid"])
train_subset = updated_records

n_with_frame = sum(1 for r in train_subset if r.get("frame_path"))
print(f"\\nFrames extracted: {{n_with_frame}} / {{len(train_subset)}}")
'''


CELL_RACECOND_CLEANUP_V2 = f'''{MARK_V2}
# This cell was previously used to delete clip_score == 0.0 frames caused by a
# CLIP race condition. With v2 (center-frame, no CLIP) that race no longer
# exists, so this cell is now a no-op. Left in place so cell ordering stays
# stable for the v1 -> v2 migration.
print("v2 pipeline: no race-condition cleanup needed.")
'''


CELL_CONSISTENCY_EVAL_V2 = f'''{MARK} {MARK_V2} {MARK_V5}
# v2 consistency-eval builder: removes the `clip_score is not None` filter
# because the center-frame extractor sets clip_score=None for all pairs.
# Also: defensive encode_texts() that handles transformers versions where
# get_text_features() returns BaseModelOutputWithPooling instead of a tensor.

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

random.seed(2026)
np.random.seed(2026)

# v2: keep every record that has a frame path (no clip_score filter).
eval_pool = [p for p in m2_pairs if p.get("frame_path")]
print(f"Eligible eval pool: {{len(eval_pool)}}")

N_POS = min(300, len(eval_pool))
positives = random.sample(eval_pool, N_POS)

print("Encoding queries with CLIP-text...")
device   = "cuda" if torch.cuda.is_available() else "cpu"
clip_hf  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def _to_text_tensor(result):
    """Coerce get_text_features() output to a 2-D tensor across HF versions."""
    if isinstance(result, torch.Tensor):
        return result
    if hasattr(result, "pooler_output") and result.pooler_output is not None:
        return clip_hf.text_projection(result.pooler_output)
    if hasattr(result, "last_hidden_state"):
        return clip_hf.text_projection(result.last_hidden_state[:, 0])
    raise RuntimeError(f"Unexpected get_text_features output: {{type(result)}}")


@torch.no_grad()
def encode_texts(texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        toks = clip_tok(texts[i:i+batch], padding=True, truncation=True,
                        return_tensors="pt").to(device)
        result = clip_hf.get_text_features(**toks)
        emb    = _to_text_tensor(result)
        emb    = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)

queries = [p["query"] for p in eval_pool]
q_emb   = encode_texts(queries)
qid_to_idx = {{p["qid"]: i for i, p in enumerate(eval_pool)}}

DISSIM_MAX = 0.60   # negative must satisfy cos(q_i, q_j) < this

negatives = []
for p in positives:
    i = qid_to_idx[p["qid"]]
    sims = q_emb @ q_emb[i]
    candidates = [j for j, s in enumerate(sims)
                  if j != i and s < DISSIM_MAX
                  and eval_pool[j]["vid"] != p["vid"]]
    if not candidates:
        continue
    j = random.choice(candidates)
    neg_video = eval_pool[j]
    negatives.append({{
        "pair_id"   : f"neg_{{p['qid']}}_{{neg_video['qid']}}",
        "label"     : 0,
        "thumb_qid" : p["qid"], "thumb_vid": p["vid"],
        "frame_path": p["frame_path"], "thumb_query": p["query"],
        "video_qid" : neg_video["qid"], "video_vid": neg_video["vid"],
        "video_query_for_ref": neg_video["query"],
        "neg_query_cos": float(sims[j]),
    }})

positive_pairs = [{{
    "pair_id"   : f"pos_{{p['qid']}}",
    "label"     : 1,
    "thumb_qid" : p["qid"], "thumb_vid": p["vid"],
    "frame_path": p["frame_path"], "thumb_query": p["query"],
    "video_qid" : p["qid"], "video_vid": p["vid"],
    "video_query_for_ref": p["query"],
}} for p in positives]

consistency_eval = positive_pairs + negatives
random.shuffle(consistency_eval)

save_jsonl(consistency_eval, PROCESSED_DIR / "consistency_eval.jsonl")
print(f"\\nConsistency eval: {{len(positive_pairs)}} positives + "
      f"{{len(negatives)}} hard negatives = {{len(consistency_eval)}} pairs")
if negatives:
    cosines = [n["neg_query_cos"] for n in negatives]
    print(f"Negative query-cos range: {{min(cosines):.3f}} .. {{max(cosines):.3f}}")
'''


# ---------- v3 cells (principled tie-breaking) ------------------------------

CELL_SELECT_BEST_FRAME_V3 = f'''{MARK_V3}
# Principled best-frame selection with a 3-tier tie-breaking hierarchy.
#
# Replaces v2's "middle of tied list" heuristic. The hierarchy:
#
#   Tier 1  unique_max
#           Exactly one clip has the highest mean saliency. Use it.
#
#   Tier 2  annotator_agreement
#           Multiple clips tied on mean. Pick the one with the LOWEST
#           per-annotator variance -- i.e. the clip the 3 annotators agreed on
#           most strongly. High agreement = stronger supervision signal.
#
#   Tier 3  temporal_position
#           Still tied (zero-variance agreement on multiple clips). Prefer the
#           LATER clip. Heuristic: in QVH queries describing actions, the
#           climax tends to sit at the end of the relevant window.
#
#   (Optional)  sharpness
#           If `frame_paths` is supplied (i.e. frames are pre-extracted for
#           every tied candidate), use Laplacian variance as a focus measure
#           and pick the sharpest. Off by default in V1 due to chicken-and-egg
#           with extraction.
#
# Returns: (best_clip_id, best_timestamp_seconds, selection_reason).

def laplacian_sharpness(frame_path):
    """Variance of the Laplacian -- standard cheap focus measure.
    Higher = sharper. Returns 0.0 on read failure (so it never wins a tie)."""
    import cv2
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_best_frame(relevant_clip_ids, saliency_scores,
                      relevant_windows=None, frame_paths=None):
    """3-tier hierarchical selection. See docstring above for tiers."""
    if not relevant_clip_ids or not saliency_scores:
        return None, None, "no_annotation"

    mean_scores  = [sum(s) / 3 for s in saliency_scores]
    max_score    = max(mean_scores)
    tied_indices = [i for i, s in enumerate(mean_scores) if s == max_score]

    # ---- Tier 1: unique max ------------------------------------------------
    if len(tied_indices) == 1:
        best_local = tied_indices[0]
        reason = "unique_max"

    else:
        # ---- Tier 2: annotator agreement (lowest variance) -----------------
        variances = [
            sum((x - mean_scores[i]) ** 2 for x in saliency_scores[i]) / 3
            for i in tied_indices
        ]
        min_var = min(variances)
        agreement_winners = [
            tied_indices[j] for j, v in enumerate(variances) if v == min_var
        ]

        if len(agreement_winners) == 1:
            best_local = agreement_winners[0]
            reason = "annotator_agreement"

        else:
            # ---- Tier 3a: sharpness (only if frame_paths given) ----------
            if frame_paths is not None:
                sharps = [laplacian_sharpness(frame_paths[i])
                          for i in agreement_winners]
                best_local = agreement_winners[sharps.index(max(sharps))]
                reason = "sharpness"
            # ---- Tier 3b: temporal position (later clip) -----------------
            else:
                best_local = max(agreement_winners,
                                 key=lambda i: relevant_clip_ids[i])
                reason = "temporal_position"

    best_clip_id   = relevant_clip_ids[best_local]
    best_timestamp = best_clip_id * 2 + 1.0   # centre of the 2-second clip
    return best_clip_id, best_timestamp, reason
'''


CELL_PROCESS_RECORD_V3 = f'''{MARK_V3}
# process_record() now delegates clip selection to select_best_frame() so the
# 3-tier hierarchy is the single source of truth. Also stores selection_reason
# on each record for downstream auditing (e.g. "what fraction of training pairs
# came from a unique-max clip vs a tied-agreement clip?").

def process_record(item, has_saliency=True):
    vid = item["vid"]
    parts = vid.rsplit("_", 2)
    yt_id, seg_start, seg_end = parts[0], float(parts[1]), float(parts[2])

    processed = {{
        "qid"              : item["qid"],
        "query"            : item["query"],
        "duration"         : item["duration"],
        "vid"              : vid,
        "youtube_id"       : yt_id,
        "seg_start"        : seg_start,
        "seg_end"          : seg_end,
        "relevant_windows" : item.get("relevant_windows", []),
        "relevant_clip_ids": item.get("relevant_clip_ids", []),
        "video_downloaded" : False,
        "frame_path"       : None,
    }}

    if has_saliency and item.get("saliency_scores") and item.get("relevant_clip_ids"):
        clip_ids = item["relevant_clip_ids"]
        scores   = item["saliency_scores"]
        best_cid, best_ts, reason = select_best_frame(
            relevant_clip_ids=clip_ids,
            saliency_scores=scores,
            relevant_windows=item.get("relevant_windows"),
        )
        mean_scores = [sum(s) / 3 for s in scores]
        processed["best_clip_id"]      = best_cid
        processed["best_timestamp"]    = best_ts
        processed["mean_saliency"]     = round(max(mean_scores), 3)
        processed["selection_reason"]  = reason
    else:
        processed["best_clip_id"]     = None
        processed["best_timestamp"]   = None
        processed["mean_saliency"]    = None
        processed["selection_reason"] = "no_annotation"

    return processed


# Process all splits
train_proc = [process_record(r, has_saliency=True)  for r in train_raw]
val_proc   = [process_record(r, has_saliency=True)  for r in val_raw]
test_proc  = [process_record(r, has_saliency=False) for r in test_raw]

print(f"Processed -> train: {{len(train_proc)}}, val: {{len(val_proc)}}, test: {{len(test_proc)}}")

# Audit the tie-breaking hierarchy on the train split.
from collections import Counter
reason_counts = Counter(r.get("selection_reason", "no_annotation") for r in train_proc)
print("\\nSelection reasons (train split):")
for k, v in reason_counts.most_common():
    print(f"  {{k:25s}}: {{v:5d}}  ({{100*v/len(train_proc):5.1f}}%)")

print("\\nSample processed record:")
print(json.dumps(train_proc[0], indent=2))
'''


# ---------- v4 cell (direct-HTTP thumbnail download) ------------------------

CELL_THUMBNAIL_DL_V4 = f'''{MARK_V4}
# Direct HTTP download from i.ytimg.com -- bypasses yt-dlp entirely.
# YouTube CDN thumbnails are public and don't trigger bot detection.

import urllib.request, urllib.error, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

THUMB_DIR = PROJECT_ROOT / "qvh_thumbnails"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

THUMB_URLS = [
    "https://i.ytimg.com/vi/{{yt}}/maxresdefault.jpg",   # 1280x720, may 404
    "https://i.ytimg.com/vi/{{yt}}/sddefault.jpg",       # 640x480
    "https://i.ytimg.com/vi/{{yt}}/hqdefault.jpg",       # 480x360, always exists
]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0 Safari/537.36")


def download_thumbnail(record):
    yt_id   = record["youtube_id"]
    out_jpg = THUMB_DIR / f"{{yt_id}}.jpg"
    if out_jpg.exists() and out_jpg.stat().st_size > 1000:
        return {{"yt_id": yt_id, "status": "skip", "path": str(out_jpg)}}

    last_err = None
    for tmpl in THUMB_URLS:
        url = tmpl.format(yt=yt_id)
        try:
            req = urllib.request.Request(url, headers={{"User-Agent": UA}})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            if len(data) > 1000:                           # filter 1x1 placeholder
                out_jpg.write_bytes(data)
                return {{"yt_id": yt_id, "status": "ok",
                        "path": str(out_jpg), "url": url, "size": len(data)}}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {{e.code}}"
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)[:200]
            continue
    return {{"yt_id": yt_id, "status": "fail",
            "err": last_err or "no_resolution_worked"}}


unique_yt = list({{r["youtube_id"]: r for r in train_subset}}.values())
print(f"Downloading thumbnails for {{len(unique_yt)}} unique YouTube videos...")

thumb_results = []
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = {{ex.submit(download_thumbnail, r): r for r in unique_yt}}
    for i, fut in enumerate(as_completed(futures), 1):
        thumb_results.append(fut.result())
        if i % 100 == 0 or i == len(unique_yt):
            ok   = sum(1 for r in thumb_results if r["status"] in ("ok", "skip"))
            fail = sum(1 for r in thumb_results if r["status"] == "fail")
            print(f"  [{{i:4d}}/{{len(unique_yt)}}]  ok={{ok}}  fail={{fail}}")

from collections import Counter
res_used = Counter(r.get("url", "").rsplit("/", 1)[-1].replace(".jpg", "")
                   for r in thumb_results if r["status"] == "ok")
print(f"\\nResolutions used: {{dict(res_used)}}")

index = []
for r in train_subset:
    p = THUMB_DIR / f"{{r['youtube_id']}}.jpg"
    if p.exists():
        index.append({{"qid": r["qid"], "vid": r["vid"],
                      "youtube_id": r["youtube_id"],
                      "thumbnail_path": str(p)}})

with open(PROCESSED_DIR / "thumbnail_index.jsonl", "w", encoding="utf-8") as f:
    for x in index:
        f.write(json.dumps(x, ensure_ascii=False) + "\\n")

with open(PROCESSED_DIR / "thumbnail_download_log.json", "w", encoding="utf-8") as f:
    json.dump(thumb_results, f, indent=2)

print(f"\\nThumbnails available for {{len(index)}} / {{len(train_subset)}} records.")
if any(r["status"] == "fail" for r in thumb_results):
    fails = [r for r in thumb_results if r["status"] == "fail"]
    print(f"Failures: {{len(fails)}} -- first 3:")
    for r in fails[:3]:
        print(f"  {{r['yt_id']}}: {{r.get('err')}}")
'''


# ---------- v7 cell (self-contained M2 val pairs) ---------------------------

CELL_M2_VAL_V7 = f'''{MARK} {MARK_V7}
# Self-contained M2 val pairs builder: defines its own downloader inline so a
# kernel restart between section 8 and section 15 cannot raise NameError.

import subprocess, json, random
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(2026)

M2_VAL_N   = 300
YT_CLIENTS = ["android", "web", "ios"]


def _dl(record):
    """Download one 150-second clip if missing. Returns status dict."""
    vid      = record["vid"]
    out_path = VIDEO_DIR / f"{{vid}}.mp4"
    if out_path.exists() and out_path.stat().st_size > 10_000:
        record["video_downloaded"] = True
        return {{"vid": vid, "status": "skip"}}
    url = f"https://www.youtube.com/watch?v={{record['youtube_id']}}"
    last_err = None
    for client in YT_CLIENTS:
        cmd = ["yt-dlp",
               "--download-sections",
               f"*{{record['seg_start']}}-{{record['seg_end']}}",
               "--format",
               "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
               "--merge-output-format", "mp4",
               "--extractor-args", f"youtube:player_client={{client}}",
               "--quiet", "--no-warnings",
               "-o", str(out_path), url]
        try:
            p = subprocess.run(cmd, timeout=180, capture_output=True, text=True)
            if p.returncode == 0 and out_path.exists():
                record["video_downloaded"] = True
                return {{"vid": vid, "status": "ok", "client": client}}
            last_err = (p.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            last_err = "timeout"
    record["video_downloaded"] = False
    return {{"vid": vid, "status": "fail", "err": last_err}}


# Stratified pick by mean_saliency tertile
val_with_sal = [r for r in val_proc if r.get("mean_saliency") is not None]
val_with_sal.sort(key=lambda r: r["mean_saliency"])
t = len(val_with_sal) // 3
buckets = [val_with_sal[:t], val_with_sal[t:2*t], val_with_sal[2*t:]]
val_pick = []
for b in buckets:
    val_pick.extend(random.sample(b, min(M2_VAL_N // 3, len(b))))
print(f"Selected {{len(val_pick)}} val records for M2 evaluation pairs.")

print("Downloading val clips...")
val_dl_results = []
with ThreadPoolExecutor(max_workers=6) as ex:
    futures = {{ex.submit(_dl, r): r for r in val_pick}}
    for i, fut in enumerate(as_completed(futures), 1):
        val_dl_results.append(fut.result())
        if i % 50 == 0 or i == len(val_pick):
            ok = sum(1 for r in val_dl_results if r["status"] in ("ok", "skip"))
            print(f"  [{{i:4d}}/{{len(val_pick)}}]  ok={{ok}}")

print("\\nExtracting M2 val frames...")
val_pick_extracted = []
with ThreadPoolExecutor(max_workers=6) as ex:
    futures = {{ex.submit(extract_frame, r): r for r in val_pick}}
    for i, fut in enumerate(as_completed(futures), 1):
        val_pick_extracted.append(fut.result())

m2_val_pairs = [
    {{"qid": r["qid"], "vid": r["vid"], "frame_path": r["frame_path"],
     "query": r["query"], "best_timestamp": r["best_timestamp"],
     "mean_saliency": r["mean_saliency"], "clip_score": r.get("clip_score")}}
    for r in val_pick_extracted if r.get("frame_path")
]

save_jsonl(m2_val_pairs, PROCESSED_DIR / "m2_val_pairs.jsonl")
print(f"\\nM2 val pairs: {{len(m2_val_pairs)}} / {{len(val_pick)}} requested")
'''


# ---------- v8 cells (unpack Moment-DETR features) --------------------------

CELL_UNPACK_FEATURES_MD = f'''{MARK_V8}
# 2.5 Unpack Official Moment-DETR Features (one-time, ~5 min)

The M1 baseline + the consistency scorer both need pre-extracted CLIP video and
text features from the official Moment-DETR release. Run this once after section 2.

Source: `moment_detr_features.tar.gz` (place it under `PROJECT_ROOT` first).

Outputs:
- `moment_detr/features/clip_features/*.npz`         -- per-video CLIP features
- `moment_detr/features/clip_text_features/*.npz`    -- per-query CLIP text features
'''


CELL_UNPACK_FEATURES_V8 = f'''{MARK_V8}
import tarfile

ARCHIVE  = PROJECT_ROOT / "moment_detr_features.tar.gz"
FEAT_DIR = REPO_DIR / "features"

if (FEAT_DIR / "clip_features").exists():
    print(f"Features already unpacked at {{FEAT_DIR}}")
else:
    if not ARCHIVE.exists():
        print(f"MISSING: {{ARCHIVE}}")
        print("Download the archive from Moment-DETR's release page first:")
        print("  https://github.com/jayleicn/moment_detr/blob/main/data/README.md")
    else:
        print(f"Unpacking {{ARCHIVE}} -> {{REPO_DIR}}")
        with tarfile.open(ARCHIVE) as tf:
            tf.extractall(REPO_DIR)
        print(f"Done. Feature dirs: {{[p.name for p in FEAT_DIR.iterdir()]}}")
        clip_n = len(list((FEAT_DIR / "clip_features").glob("*.npz")))
        txt_n  = len(list((FEAT_DIR / "clip_text_features").glob("*.npz")))
        print(f"  clip_features      : {{clip_n}} files")
        print(f"  clip_text_features : {{txt_n}} files")
'''


# ---------- v10..v14 cells (taxonomy, reconcile, gold set, hard+qmiss) ----

CELL_MULTIFRAME_MD_V9 = f'''{MARK_V9}
# 10.5 — Multi-frame augmentation for M2 training

For each annotated record we extract **K=3 frames** evenly spaced across the
`relevant_clip_ids` (the moment) and treat each `(frame, query)` as a separate
training example. This produces ~3× the training data of the single-frame set
and gives M2 more visual diversity per query.

Output: `processed/m2_train_pairs_multi.jsonl` -- the **★ recommended training
set** for ClipCap (M2). The single-frame `m2_train_pairs.jsonl` stays as a
fallback.

Frames land in `qvh_frames_multi/{{vid}}_clip{{N}}.jpg` (a separate folder from
`qvh_frames/`, so no file collisions).
'''


CELL_MULTIFRAME_V9 = f'''{MARK_V9}
# === Multi-frame augmentation for M2 training (full QVH set) ===
# Treats each (frame, query) as a separate training example.
# Scales with however many records are in train_processed.jsonl.

import cv2, random, json
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

random.seed(2026)

# CONFIG
K_FRAMES_PER_QUERY = 3
MIN_RELEVANT_CLIPS = 1     # skip records with no relevant clips annotated
QVH_VIDEOS         = VIDEO_DIR
QVH_MULTI_FRAMES   = PROJECT_ROOT / 'qvh_frames_multi'
QVH_MULTI_FRAMES.mkdir(parents=True, exist_ok=True)

# Load all train records that have annotation
TRAIN_PROC = PROCESSED_DIR / 'train_processed.jsonl'
train_records = []
with open(TRAIN_PROC, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('relevant_clip_ids') and len(r['relevant_clip_ids']) >= MIN_RELEVANT_CLIPS:
            train_records.append(r)

print(f"Loaded {{len(train_records):,}} train records with annotation")
print(f"Will extract K={{K_FRAMES_PER_QUERY}} frames per record")
print(f"Expected output: ~{{K_FRAMES_PER_QUERY * len(train_records):,}} (frame, query) pairs")


def sample_relevant_frames(record, k=3):
    """Extract k frames spread across the relevant_clip_ids of a record."""
    relevant = sorted(record['relevant_clip_ids'])
    video_path = QVH_VIDEOS / f"{{record['vid']}}.mp4"
    if not video_path.exists():
        return []

    # Spread k indices across the relevant clip range for visual diversity
    if len(relevant) >= k:
        idxs = [int(round((len(relevant) - 1) * i / max(1, k - 1))) for i in range(k)]
        chosen = [relevant[i] for i in idxs]
    else:
        chosen = relevant + random.choices(relevant, k=k - len(relevant))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    out = []
    for clip_id in chosen:
        out_path = QVH_MULTI_FRAMES / f"{{record['vid']}}_clip{{clip_id}}.jpg"
        if out_path.exists():
            out.append({{'path': str(out_path), 'clip_id': clip_id}})
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, float(clip_id * 2_000 + 1_000))
        ret, fr = cap.read()
        if not ret:
            continue
        pil = (Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                   .resize((224, 224), Image.LANCZOS))
        pil.save(str(out_path), quality=92)
        out.append({{'path': str(out_path), 'clip_id': clip_id}})
    cap.release()
    return out


m2_pairs_multi = []
n_missing_video = 0
for rec in tqdm(train_records, desc='Multi-frame extraction'):
    frames = sample_relevant_frames(rec, k=K_FRAMES_PER_QUERY)
    if not frames:
        n_missing_video += 1
        continue
    for f in frames:
        m2_pairs_multi.append({{
            'qid'           : rec['qid'],
            'vid'           : rec['vid'],
            'frame_path'    : f['path'],
            'query'         : rec['query'],
            'clip_id'       : f['clip_id'],
            'best_timestamp': f['clip_id'] * 2 + 1.0,
            'mean_saliency' : rec.get('mean_saliency'),
            'augment_id'    : f"{{rec['vid']}}_clip{{f['clip_id']}}",
        }})

out_path = PROCESSED_DIR / 'm2_train_pairs_multi.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for p in m2_pairs_multi:
        f.write(json.dumps(p, ensure_ascii=False) + '\\n')

print(f"\\nWrote {{out_path}}")
print(f"  records with frames extracted : {{len(train_records) - n_missing_video:,}}")
print(f"  records skipped (missing video): {{n_missing_video:,}}")
print(f"  total (frame, query) pairs    : {{len(m2_pairs_multi):,}}")
'''


CELL_TAXONOMY_MD = f'''{MARK_V10}
# 0.5 — Data Taxonomy (read me first)

Every artifact this notebook produces serves a specific role. Use this map to
understand where each folder/file is consumed downstream.

| Path | Built by | Used by |
|---|---|---|
| `qvh_videos/{{vid}}.mp4`              | Section 8 (yt-dlp download)               | All frame extractors |
| `qvh_frames/{{vid}}.jpg`              | Section 9 (center / CLIP-ranked)          | **M2 training (single-frame)**, eval positives |
| `qvh_frames_multi/{{vid}}_clip{{N}}.jpg` | v9 multi-frame aug cell                | **M2 training (multi-frame, RECOMMENDED)** |
| `qvh_frames_within_neg/{{vid}}_clip{{N}}.jpg` | Hard eval builder (v14)            | **Within-video hard negatives** in `consistency_eval_hard.jsonl` |
| `qvh_thumbnails/{{youtube_id}}.jpg`   | Section 13 (direct HTTP from i.ytimg.com) | **Optional domain-gap probe**: replace `frame_path` with `thumbnail_path` and re-run eval to measure how much real stylized thumbnails hurt AUC |
| `processed/train_processed.jsonl`     | Section 5                                  | Source for all frame extraction |
| `processed/m1_*.jsonl`                | Section 14                                  | M1 audits / optional retrain |
| `processed/m2_train_pairs.jsonl`      | Section 10                                  | M2 single-frame training (fallback) |
| `processed/m2_train_pairs_multi.jsonl`| v9 multi-frame aug                          | **★ Default M2 training set** |
| `processed/m2_val_pairs.jsonl`        | v7 self-contained M2-val builder            | M2 early stopping (CLIP-text-sim) |
| `processed/consistency_eval.jsonl`    | Section 16                                  | **E3 ceiling** AUC (cross-video) |
| `processed/consistency_eval_hard.jsonl` | v14 hard builder                          | **E4 research claim** AUC (within-video) |
| `processed/consistency_eval_query_mismatch.jsonl` | v14 qmiss builder               | **E5 alpha-sweep** (query path test) |
| `processed/manual_gold.jsonl`         | v12 gold builder + you labeling             | **E6 small-but-real** AUC (~50 pairs) |
| `processed/m2_image_emb.npz`          | Section 17                                  | Cached frame embeddings (faster eval) |
| `processed/thumbnail_image_emb.npz`   | Section 17                                  | Cached thumbnail embeddings (for domain-gap probe) |
| `processed/category_map.json`         | Section 6                                   | Stratification + reporting |

## What is the QV thumbnails section for?

You probably wondered: we downloaded `qvh_thumbnails/`, but the training pipeline
never uses them. That's intentional. **They are an evaluation-only artifact for
the domain-gap experiment** (E8 in EVALUATION.md):

1. M2 was trained on **video frames** (CLIP-similar, "natural" images).
2. In a real deployment, M2 receives **stylized YouTube thumbnails** (text
   overlays, color grading, edited composites).
3. The gap between these two distributions degrades M2's queries.
4. To **measure** this gap, re-run the consistency scorer using the real
   thumbnails as the `thumbnail_path` (instead of `qvh_frames/{{vid}}.jpg`)
   and compare AUC.

If you skip the domain-gap experiment, you can ignore `qvh_thumbnails/`
entirely — it doesn't affect training or the main AUC.
'''


CELL_RECONCILE_V11 = f'''{MARK_V11}
# Reconcile previously-extracted v1 frames with the current pipeline.
#
# Use this cell IF: you already extracted frames with the old v1 (CLIP-ranked)
# method and want to keep them. This cell DOES NOT re-extract anything --
# it just rebuilds `m2_pairs` in memory from whatever .jpg files exist in
# FRAMES_DIR, so downstream cells use them.
#
# Skip this cell if you ran the v2/v3 center-frame extractor in this same
# session (m2_pairs is already correct).

import json
from pathlib import Path


def _load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\\n')


# Reload train_subset from disk if not in memory.
if 'train_subset' not in dir():
    tp = PROCESSED_DIR / 'train_processed.jsonl'
    assert tp.exists(), f"Missing {{tp}} -- run Sections 5-7 first."
    train_subset = _load_jsonl(tp)

# Mark frame_path on each record whose .jpg exists on disk.
n_with_frame = 0
for r in train_subset:
    candidate = FRAMES_DIR / f"{{r['vid']}}.jpg"
    if candidate.exists():
        r['frame_path']       = str(candidate)
        r['video_downloaded'] = True
        if 'clip_score' not in r:
            r['clip_score'] = None
        n_with_frame += 1
    else:
        r['frame_path']       = None
        r['video_downloaded'] = False

# Rebuild m2_pairs from records that have frames.
m2_pairs = [{{
    'qid'           : r['qid'],
    'frame_path'    : r['frame_path'],
    'query'         : r['query'],
    'vid'           : r['vid'],
    'best_timestamp': r.get('best_timestamp'),
    'mean_saliency' : r.get('mean_saliency'),
    'clip_score'    : r.get('clip_score'),
}} for r in train_subset if r.get('frame_path')]

_save_jsonl(train_subset, PROCESSED_DIR / 'train_processed.jsonl')
_save_jsonl(m2_pairs,     PROCESSED_DIR / 'm2_train_pairs.jsonl')

print(f"train_subset           : {{len(train_subset):,}} records")
print(f"records with frame     : {{n_with_frame:,}}")
print(f"m2_pairs (frame, query): {{len(m2_pairs):,}}")
print(f"\\nIf m2_pairs > 0, downstream cells (multi-frame, eval builders) are ready to run.")
'''


CELL_HARD_QMISS_MD = f'''{MARK_V14}
# 16.5 — Build the harder evaluation sets

Three eval sets, in increasing difficulty:

1. `consistency_eval.jsonl`              -- cross-video negatives (already built in Section 16)
2. `consistency_eval_hard.jsonl`         -- ★ within-video negatives (built below) — the research claim
3. `consistency_eval_query_mismatch.jsonl` -- query-mismatch negatives (built below) — isolates the query path

Each has 300 positives + 300 negatives = 600 pairs total.
'''


CELL_HARD_BUILDER_V14 = f'''{MARK_V14}
# Build consistency_eval_hard.jsonl -- within-video negatives.
# For each positive, pick a frame from OUTSIDE the relevant_window of the SAME video.

import cv2, random, json
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

random.seed(2026)

WITHIN_NEG_DIR = PROJECT_ROOT / 'qvh_frames_within_neg'
WITHIN_NEG_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing positives + train records
positives = [p for p in m2_pairs if p.get('frame_path')]
train_by_qid = {{r['qid']: r for r in train_subset}}

def extract_outside_frame(record):
    n_clips = int(record.get('duration', 150) // 2) if record.get('duration') else 75
    relevant = set(record.get('relevant_clip_ids', []))
    outside = [c for c in range(n_clips) if c not in relevant]
    if not outside:
        return None
    # Prefer clips at distance >= 5 from any relevant clip (avoid transitions)
    if relevant:
        far = [c for c in outside if min(abs(c - r) for r in relevant) >= 5]
        outside = far if far else outside
    chosen = random.choice(outside)
    video_path = VIDEO_DIR / f"{{record['vid']}}.mp4"
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(chosen * 2_000 + 1_000))
    ret, fr = cap.read()
    cap.release()
    if not ret:
        return None
    out_path = WITHIN_NEG_DIR / f"{{record['vid']}}_clip{{chosen}}.jpg"
    pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((224, 224), Image.LANCZOS)
    pil.save(str(out_path), quality=92)
    return {{'path': str(out_path), 'clip_id': chosen}}

# Pick 300 positives for the hard eval (a balanced subset)
N = min(300, len(positives))
chosen = random.sample(positives, N)

hard_eval = []
for p in tqdm(chosen, desc='within-video negatives'):
    rec = train_by_qid.get(p['qid'])
    if rec is None or not rec.get('relevant_clip_ids'):
        continue
    # Positive
    hard_eval.append({{
        'pair_id'    : f"pos_{{p['qid']}}",
        'label'      : 1,
        'thumb_qid'  : p['qid'], 'thumb_vid': p['vid'],
        'frame_path' : p['frame_path'],
        'thumb_query': p['query'],
        'video_qid'  : p['qid'], 'video_vid': p['vid'],
    }})
    # Within-video negative
    neg = extract_outside_frame(rec)
    if neg is None:
        continue
    hard_eval.append({{
        'pair_id'    : f"hardneg_{{p['qid']}}_clip{{neg['clip_id']}}",
        'label'      : 0,
        'thumb_qid'  : p['qid'], 'thumb_vid': p['vid'],
        'frame_path' : neg['path'],
        'thumb_query': p['query'],
        'video_qid'  : p['qid'], 'video_vid': p['vid'],
        'out_of_window_clip_id': neg['clip_id'],
    }})

random.shuffle(hard_eval)
out_path = PROCESSED_DIR / 'consistency_eval_hard.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for x in hard_eval:
        f.write(json.dumps(x, ensure_ascii=False) + '\\n')

print(f"Hard eval written: {{out_path}}")
print(f"  positives        : {{sum(1 for p in hard_eval if p['label']==1)}}")
print(f"  within-video neg : {{sum(1 for p in hard_eval if p['label']==0)}}")
'''


CELL_QMISS_BUILDER_V14 = f'''{MARK_V14}
# Build consistency_eval_query_mismatch.jsonl -- query-mismatch negatives.
# Same frame + same video, but a WRONG query (semantically unrelated).

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

random.seed(2026)
np.random.seed(2026)

positives = [p for p in m2_pairs if p.get('frame_path')]
print(f"Encoding {{len(positives)}} queries with CLIP-text...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_hf = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def encode_texts(texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        toks = clip_tok(texts[i:i+batch], padding=True, truncation=True,
                        return_tensors="pt").to(device)
        result = clip_hf.text_model(**toks).pooler_output
        emb = clip_hf.text_projection(result)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)

all_queries = [p['query'] for p in positives]
q_emb = encode_texts(all_queries)

DISSIM_MAX = 0.40
qmiss_eval = []
N = min(300, len(positives))
chosen = random.sample(range(len(positives)), N)

for i in chosen:
    p = positives[i]
    sims = q_emb @ q_emb[i]
    cands = [j for j, s in enumerate(sims) if j != i and s < DISSIM_MAX]
    if not cands:
        continue
    j = random.choice(cands)
    wrong_query = all_queries[j]

    qmiss_eval.append({{
        'pair_id'    : f"pos_{{p['qid']}}",
        'label'      : 1,
        'thumb_qid'  : p['qid'], 'thumb_vid': p['vid'],
        'frame_path' : p['frame_path'],
        'thumb_query': p['query'],
        'video_qid'  : p['qid'], 'video_vid': p['vid'],
    }})
    qmiss_eval.append({{
        'pair_id'           : f"qmiss_{{p['qid']}}_with_q{{positives[j]['qid']}}",
        'label'             : 0,
        'thumb_qid'         : p['qid'], 'thumb_vid': p['vid'],
        'frame_path'        : p['frame_path'],
        'thumb_query'       : wrong_query,
        'video_qid'         : p['qid'], 'video_vid': p['vid'],
        'correct_query_was' : p['query'],
        'wrong_query_from'  : positives[j]['qid'],
        'cos_correct_wrong' : float(sims[j]),
    }})

random.shuffle(qmiss_eval)
out_path = PROCESSED_DIR / 'consistency_eval_query_mismatch.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for x in qmiss_eval:
        f.write(json.dumps(x, ensure_ascii=False) + '\\n')

print(f"Query-mismatch eval: {{out_path}}")
print(f"  positives        : {{sum(1 for p in qmiss_eval if p['label']==1)}}")
print(f"  query-mismatch neg: {{sum(1 for p in qmiss_eval if p['label']==0)}}")
'''


CELL_GOLD_MD = f'''{MARK_V12}
# 19 — Manual Gold Evaluation Set (the "ground truth" sanity check)

The synthetic eval sets (cross-video, within-video, query-mismatch) are built
**automatically** from QVH structure. They give us 600 pairs with zero labeling
effort -- but they may not reflect real human judgment of consistency.

The **manual gold set** is a small sample of 50 pairs that team members label
by hand. We then compute AUC on this set and compare it against the synthetic
AUC. If they're close, the synthetic eval is a faithful proxy.

This section produces:
1. `manual_gold_template.jsonl` -- 50 pairs to label, with `label_human=null`
2. `manual_gold.jsonl` -- the SAME file after each team member fills in labels

**Time investment**: ~2 hours total across 4 people (12-13 pairs each).
'''


CELL_GOLD_BUILDER_V12 = f'''{MARK_V12}
# Build the gold template: 50 pairs (25 positives + 25 negatives) sampled from
# consistency_eval.jsonl, with empty label_human fields to fill manually.

import random, json
from pathlib import Path

random.seed(2026)

CONSIST_PATH = PROCESSED_DIR / 'consistency_eval.jsonl'
assert CONSIST_PATH.exists(), f"Run Section 16 first to build {{CONSIST_PATH}}"

pairs = []
with open(CONSIST_PATH, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

pos = [p for p in pairs if p['label'] == 1]
neg = [p for p in pairs if p['label'] == 0]

# Stratified sample: 25 of each
N_PER_LABEL = 25
sampled = random.sample(pos, min(N_PER_LABEL, len(pos))) \\
        + random.sample(neg, min(N_PER_LABEL, len(neg)))
random.shuffle(sampled)

# Build template with empty label_human
template = []
for p in sampled:
    template.append({{
        'pair_id'           : p['pair_id'],
        'label_synthetic'   : p['label'],
        'label_human'       : None,           # fill: 0=inconsistent, 1=consistent, 0.5=partial
        'confidence'        : None,           # fill: 1=sure, 2=somewhat sure, 3=guessing
        'note'              : '',              # fill: optional comment
        'thumb_vid'         : p['thumb_vid'],
        'frame_path'        : p['frame_path'],
        'thumb_query'       : p['thumb_query'],
        'video_vid'         : p['video_vid'],
        'video_query_for_ref': p.get('video_query_for_ref'),
    }})

out_path = PROCESSED_DIR / 'manual_gold_template.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for r in template:
        f.write(json.dumps(r, ensure_ascii=False) + '\\n')

print(f"Wrote template: {{out_path}}")
print(f"  total pairs    : {{len(template)}}")
print(f"  synthetic pos  : {{sum(1 for r in template if r['label_synthetic']==1)}}")
print(f"  synthetic neg  : {{sum(1 for r in template if r['label_synthetic']==0)}}")
print(f"\\nSee docs/LABELING_GUIDE.md for instructions on how to fill labels.")
print(f"Save the labeled file as: {{PROCESSED_DIR}}/manual_gold.jsonl")
'''


CELL_GOLD_VIEWER_V12 = f'''{MARK_V12}
# Interactive viewer: display each gold-set pair (thumb + video frames + query).
# Use this to label by eye. Records can be filled in the template JSONL using
# any editor.

import cv2, json
from PIL import Image
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path

template_path = PROCESSED_DIR / 'manual_gold_template.jsonl'
if not template_path.exists():
    print(f"Run the previous cell first to build {{template_path}}")
else:
    pairs = []
    with open(template_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))


    def show_pair(p, n_video_frames=4):
        fig, axes = plt.subplots(1, 1 + n_video_frames, figsize=(3.5 * (1 + n_video_frames), 3.5))
        # Thumbnail
        try:
            axes[0].imshow(Image.open(p['frame_path']).convert('RGB'))
            axes[0].set_title(f"THUMB\\n{{p['thumb_vid'][:22]}}", fontsize=9)
        except Exception:
            axes[0].text(0.5, 0.5, 'thumb missing', ha='center')
        axes[0].axis('off')

        # Video frames sampled from the .mp4
        vp = VIDEO_DIR / f"{{p['video_vid']}}.mp4"
        if vp.exists():
            cap = cv2.VideoCapture(str(vp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            for j in range(n_video_frames):
                target = int(total * (j + 0.5) / n_video_frames) if total else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, fr = cap.read()
                ax = axes[1 + j]
                if ret:
                    img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    ax.imshow(img); ax.set_title(f"VIDEO @ ~{{int(100*(j+0.5)/n_video_frames)}}%", fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center')
                ax.axis('off')
            cap.release()
        else:
            for j in range(n_video_frames):
                axes[1 + j].text(0.5, 0.5, 'no .mp4', ha='center'); axes[1 + j].axis('off')

        qtext = textwrap.fill(p['thumb_query'], 100)
        synth = 'POSITIVE' if p['label_synthetic'] == 1 else 'NEGATIVE'
        ref = p.get('video_query_for_ref') or '(same as thumb_query for positives)'
        fig.suptitle(
            f"pair_id={{p['pair_id']}}  |  synthetic_label={{synth}}\\n"
            f"Thumb's query : {{qtext}}\\n"
            f"Video's actual: {{textwrap.fill(ref, 90)}}",
            fontsize=10, y=1.05)
        plt.tight_layout(); plt.show()


    # Show the first 5 pairs as a demo. Re-run to see more, or batch them all.
    for p in pairs[:5]:
        show_pair(p)
        print(f"  pair_id  : {{p['pair_id']}}")
        print(f"  TO LABEL : edit manual_gold_template.jsonl, set:")
        print(f"             label_human = 0 (inconsistent) or 1 (consistent) or 0.5 (partial)")
        print(f"             confidence  = 1, 2, or 3")
        print('-' * 70)
'''


CELL_GOLD_EVAL_MD = f'''{MARK_V13}
# 20 — Gold-set pipeline test (run after labeling is complete)

After all 50 pairs have `label_human` filled in (saved as `manual_gold.jsonl`),
this cell runs the consistency scorer on them and computes:

- **Gold AUC**: AUC using `label_human` (binarize 0.5 -> 0)
- **Synthetic AUC**: AUC using `label_synthetic`
- **Inter-rater agreement** if multiple labelers contributed
- **Synthetic vs gold correlation**: does the synthetic eval reflect human judgment?

If `gold AUC` ≈ `synthetic AUC` (within ~0.05), the synthetic eval is a faithful
proxy and we can confidently report the 600-pair synthetic numbers in the paper.
'''


CELL_GOLD_EVAL_V13 = f'''{MARK_V13}
# Run pipeline on the labeled gold set and compute AUC.
# Requires: src.scoring.* modules; bootstrap cell (in modular notebooks) sets sys.path.
# For data_collection_preparation.ipynb, ensure REPO_DIR is on sys.path first.

import sys, json
from pathlib import Path

REPO_DIR = '/content/drive/MyDrive/thumbnail_project/thumbnail-consistency'
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np

GOLD_PATH = PROCESSED_DIR / 'manual_gold.jsonl'
if not GOLD_PATH.exists():
    print(f"Gold file missing: {{GOLD_PATH}}")
    print("Label the template first (see docs/LABELING_GUIDE.md), then save")
    print("the result here as 'manual_gold.jsonl'.")
else:
    gold = []
    with open(GOLD_PATH, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                gold.append(json.loads(line))

    # Filter records that have a human label
    labeled = [r for r in gold if r.get('label_human') is not None]
    print(f"Total gold pairs   : {{len(gold)}}")
    print(f"Labeled (human)    : {{len(labeled)}}")
    if not labeled:
        print("No human labels filled yet. Stop here.")
    else:
        # Convert 0.5 (partial) -> 0 for AUC binarization
        h_lab = np.array([1 if r['label_human'] == 1 else 0 for r in labeled])
        s_lab = np.array([r['label_synthetic'] for r in labeled])

        # Agreement
        agree = (h_lab == s_lab).mean()
        print(f"\\nHuman vs synthetic label agreement: {{agree:.3f}}")
        if agree < 0.75:
            print("  WARNING: low agreement. Synthetic eval may not be a good proxy.")

        # Show breakdown
        from collections import Counter
        print(f"\\nHuman label distribution    : {{Counter(h_lab.tolist())}}")
        print(f"Synthetic label distribution: {{Counter(s_lab.tolist())}}")

        # Now run the scorer (requires src/* on sys.path + features extracted)
        # If you haven't run the smoke test yet, you can compute scores here too.
        # For now we just report the agreement number and let you compute AUC
        # in the run_evaluation.ipynb notebook with EVAL_PATH = manual_gold.jsonl.

        print(f"\\nNext step: open notebooks/run_evaluation.ipynb and change")
        print(f"  EVAL_PATH = '/content/drive/.../processed/manual_gold.jsonl'")
        print(f"Then re-run the scoring loop and compare AUC vs the synthetic AUC.")
'''


# ---------- patcher ----------------------------------------------------------

def patch_existing(nb) -> int:
    """Replace bug-fixed cells in place. Returns number of cells patched."""
    n = 0

    # Cell: load_jsonl + load splits (B1)
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "load_jsonl" in cell_src(c)
                                and "highlight_train_release" in cell_src(c))
    if i >= 0:
        set_src(nb["cells"][i], CELL_LOAD_JSONL)
        n += 1

    # Cell: save_jsonl + save splits (B1)
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def save_jsonl" in cell_src(c)
                                and "train_processed.jsonl" in cell_src(c))
    if i >= 0:
        set_src(nb["cells"][i], CELL_SAVE_JSONL)
        n += 1

    # Cell: infer_category (B2)
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "TRAVEL_KEYWORDS" in cell_src(c)
                                and "NEWS_KEYWORDS" in cell_src(c))
    if i >= 0:
        set_src(nb["cells"][i], CELL_INFER_CATEGORY)
        n += 1

    # Cell: download_clip (B3)
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def download_clip" in cell_src(c))
    if i >= 0:
        set_src(nb["cells"][i], CELL_DOWNLOAD_CLIP)
        n += 1

    # Cell: extract_frame (B4 + B5 + B6)
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def extract_frame" in cell_src(c))
    if i >= 0:
        set_src(nb["cells"][i], CELL_EXTRACT_FRAME)
        n += 1

    return n


def append_new_cells(nb) -> int:
    """Append the new pipeline cells if not already present (idempotent)."""
    already = any(MARK in cell_src(c) and "13. Download Original YouTube"
                  in cell_src(c) for c in nb["cells"])
    if already:
        return 0

    new = [
        markdown_cell(CELL_THUMBNAIL_DL_MD),
        code_cell(CELL_THUMBNAIL_DL),
        markdown_cell(CELL_M1_SUBSET_MD),
        code_cell(CELL_M1_SUBSET),
        markdown_cell(CELL_M2_VAL_MD),
        code_cell(CELL_M2_VAL),
        markdown_cell(CELL_CONSISTENCY_EVAL_MD),
        code_cell(CELL_CONSISTENCY_EVAL),
        markdown_cell(CELL_IMG_EMB_MD),
        code_cell(CELL_IMG_EMB),
        markdown_cell(CELL_FINAL_SUMMARY_MD),
        code_cell(CELL_FINAL_SUMMARY),
    ]
    nb["cells"].extend(new)
    return len(new)


def apply_v2_center_frame(nb) -> int:
    """Switch extraction to center-frame (no CLIP) + tie-aware select_best_frame
    + update the consistency_eval builder to drop the clip_score filter.

    Idempotent: cells already carrying MARK_V2 are left alone.
    """
    n = 0

    # 1) select_best_frame -- usually lives in cell 10 ("import cv2" + def).
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def select_best_frame" in cell_src(c))
    if i >= 0 and MARK_V2 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_SELECT_BEST_FRAME_V2)
        n += 1

    # 2) extract_frame -- patched in v1; replace again with the no-CLIP version.
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def extract_frame" in cell_src(c))
    if i >= 0 and MARK_V2 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_EXTRACT_FRAME_V2)
        n += 1

    # 3) Race-condition cleanup cell -- turn into a no-op.
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "clip_score == 0.0" in cell_src(c))
    if i >= 0 and MARK_V2 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_RACECOND_CLEANUP_V2)
        n += 1

    # 4) consistency_eval builder -- swap filter AND ensure v5 defensive
    #    encode_texts is present. We re-apply if MARK_V5 is missing, even if
    #    the cell already carries an older v2 mark.
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "consistency_eval.jsonl" in cell_src(c)
                                and "negatives = []" in cell_src(c))
    if i >= 0 and MARK_V5 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_CONSISTENCY_EVAL_V2)
        n += 1

    return n


def apply_v3_principled_selection(nb) -> int:
    """Switch select_best_frame to the 3-tier annotator-agreement hierarchy
    and route process_record() through it.

    Idempotent: cells already carrying MARK_V3 are left alone.
    """
    n = 0

    # 1) select_best_frame -- v2 set this cell to "middle of tied". Replace
    #    with the principled hierarchy.
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def select_best_frame" in cell_src(c))
    if i >= 0 and MARK_V3 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_SELECT_BEST_FRAME_V3)
        n += 1

    # 2) process_record -- delegate to select_best_frame and record reason.
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def process_record" in cell_src(c))
    if i >= 0 and MARK_V3 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_PROCESS_RECORD_V3)
        n += 1

    return n


def apply_v4_direct_thumbnails(nb) -> int:
    """Swap the yt-dlp-based thumbnail downloader with direct i.ytimg.com HTTP."""
    n = 0
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def download_thumbnail" in cell_src(c)
                                and "yt-dlp" in cell_src(c))
    if i >= 0 and MARK_V4 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_THUMBNAIL_DL_V4)
        n += 1
    return n


def apply_v5_safe_encode(nb) -> int:
    """v5: add defensive _to_image_tensor to the CLIP-image embedding cell.
    v6 supersedes this by bypassing get_image_features entirely. We force
    re-application when MARK_V6 is missing, so the cell content gets the
    bypass even if it already carries an older v5 mark.
    """
    n = 0
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "def encode_images" in cell_src(c))
    if i >= 0 and MARK_V6 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_IMG_EMB)
        n += 1
    return n


def apply_v7_selfcontained_m2_val(nb) -> int:
    """Replace the M2 val builder with a self-contained version (no dependency
    on download_clip being in scope)."""
    n = 0
    # Signature: the v1 cell that picks val records and calls download_clip
    i = find_cell(nb, lambda c: c["cell_type"] == "code"
                                and "M2_VAL_N" in cell_src(c)
                                and "val_pick" in cell_src(c))
    if i >= 0 and MARK_V7 not in cell_src(nb["cells"][i]):
        set_src(nb["cells"][i], CELL_M2_VAL_V7)
        n += 1
    return n


def apply_v8_unpack_features(nb) -> int:
    """Append a one-time feature-unpacking cell if it isn't already present."""
    already = any(MARK_V8 in cell_src(c) for c in nb["cells"])
    if already:
        return 0
    # Append at the end -- order-independent (just unpacks an archive).
    nb["cells"].append(markdown_cell(CELL_UNPACK_FEATURES_MD))
    nb["cells"].append(code_cell(CELL_UNPACK_FEATURES_V8))
    return 2


def apply_v9_multiframe_aug(nb) -> int:
    """Append the multi-frame augmentation cell (and its markdown header)."""
    if any(MARK_V9 in cell_src(c) for c in nb["cells"]):
        return 0
    nb["cells"].append(markdown_cell(CELL_MULTIFRAME_MD_V9))
    nb["cells"].append(code_cell(CELL_MULTIFRAME_V9))
    return 2


def apply_v10_taxonomy(nb) -> int:
    """Append a data taxonomy markdown explaining each artifact."""
    if any(MARK_V10 in cell_src(c) for c in nb["cells"]):
        return 0
    nb["cells"].append(markdown_cell(CELL_TAXONOMY_MD))
    return 1


def apply_v11_reconcile(nb) -> int:
    """Append a reconciliation cell that uses existing v1 frames without re-extracting."""
    if any(MARK_V11 in cell_src(c) for c in nb["cells"]):
        return 0
    nb["cells"].append(code_cell(CELL_RECONCILE_V11))
    return 1


def apply_v14_hard_qmiss(nb) -> int:
    """Append hard-eval + query-mismatch builder cells if not present."""
    if any(MARK_V14 in cell_src(c) for c in nb["cells"]):
        return 0
    nb["cells"].append(markdown_cell(CELL_HARD_QMISS_MD))
    nb["cells"].append(code_cell(CELL_HARD_BUILDER_V14))
    nb["cells"].append(code_cell(CELL_QMISS_BUILDER_V14))
    return 3


def apply_v12_gold_set(nb) -> int:
    """Append the manual gold-set builder + viewer cells."""
    if any(MARK_V12 in cell_src(c) for c in nb["cells"]):
        return 0
    nb["cells"].append(markdown_cell(CELL_GOLD_MD))
    nb["cells"].append(code_cell(CELL_GOLD_BUILDER_V12))
    nb["cells"].append(code_cell(CELL_GOLD_VIEWER_V12))
    return 3


def apply_v13_gold_eval(nb) -> int:
    """Append the gold-set pipeline test cell."""
    if any(MARK_V13 in cell_src(c) for c in nb["cells"]):
        return 0
    nb["cells"].append(markdown_cell(CELL_GOLD_EVAL_MD))
    nb["cells"].append(code_cell(CELL_GOLD_EVAL_V13))
    return 2


def main():
    if not NB_PATH.exists():
        print(f"Notebook not found: {NB_PATH}", file=sys.stderr)
        sys.exit(1)

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    n_patched = patch_existing(nb)
    n_added   = append_new_cells(nb)
    n_v2      = apply_v2_center_frame(nb)
    n_v3      = apply_v3_principled_selection(nb)
    n_v4      = apply_v4_direct_thumbnails(nb)
    n_v5      = apply_v5_safe_encode(nb)
    n_v7      = apply_v7_selfcontained_m2_val(nb)
    n_v8      = apply_v8_unpack_features(nb)
    n_v9      = apply_v9_multiframe_aug(nb)
    n_v10     = apply_v10_taxonomy(nb)
    n_v11     = apply_v11_reconcile(nb)
    n_v14     = apply_v14_hard_qmiss(nb)
    n_v12     = apply_v12_gold_set(nb)
    n_v13     = apply_v13_gold_eval(nb)

    # Backup before write.
    backup = NB_PATH.with_suffix(".ipynb.bak")
    if not backup.exists():
        backup.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"v1 bug-fix patches applied : {n_patched}")
    print(f"v1 new cells appended      : {n_added}")
    print(f"v2 center-frame patches    : {n_v2}")
    print(f"v3 principled selection    : {n_v3}")
    print(f"v4 direct-HTTP thumbnails  : {n_v4}")
    print(f"v5/v6 safe CLIP encode     : {n_v5}")
    print(f"v7 self-contained m2 val   : {n_v7}")
    print(f"v8 unpack features cells   : {n_v8}")
    print(f"v9 multi-frame aug cells   : {n_v9}")
    print(f"v10 data taxonomy MD       : {n_v10}")
    print(f"v11 reconcile v1 frames    : {n_v11}")
    print(f"v14 hard + qmiss builders  : {n_v14}")
    print(f"v12 gold-set builder       : {n_v12}")
    print(f"v13 gold-set evaluator     : {n_v13}")
    print(f"Total cells now            : {len(nb['cells'])}")
    print(f"Wrote: {NB_PATH}")


if __name__ == "__main__":
    main()
