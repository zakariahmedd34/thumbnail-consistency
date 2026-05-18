"""Chunk a long video's clip-feature sequence into <=75-clip windows.

For QVHighlights segments (duration == 150s -> 75 clips) the chunker is a no-op.
For longer videos, we slide a 75-clip window with stride 60 (overlap 15 clips
= 30 seconds), so a 5-minute video gives 3 chunks.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def chunk_indices(
    n_clips: int,
    window: int = 75,
    stride: int = 60,
) -> List[Tuple[int, int]]:
    """Return list of (start_clip, end_clip_exclusive) windows covering [0, n_clips)."""
    if n_clips <= window:
        return [(0, n_clips)]
    out = []
    s = 0
    while s < n_clips:
        e = min(s + window, n_clips)
        out.append((s, e))
        if e == n_clips:
            break
        s += stride
    return out


def chunk_features(
    v_feat: np.ndarray,  # (n_clips, D)
    window: int = 75,
    stride: int = 60,
) -> List[np.ndarray]:
    """Return list of sub-arrays, each (T_w, D) with T_w <= window."""
    idxs = chunk_indices(v_feat.shape[0], window=window, stride=stride)
    return [v_feat[s:e] for s, e in idxs]
