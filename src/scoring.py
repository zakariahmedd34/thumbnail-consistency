"""Thumbnail-video consistency score.

    C(T, V) = (1/K) * sum_k  max_w [ alpha * cos(e_qk, e_w,k) + (1 - alpha) * cos(e_T, e_w,k) ]

  e_T     CLIP image embedding of the thumbnail
  e_qk    CLIP text embedding of query k (queries come from an M2 generator)
  e_w,k   video window w pooled with softmax(M1 saliency for query k / tau),
          keeping only the top_p most salient clips

The pure-numpy functions at the top are unit-tested; ConsistencyScorer at the
bottom wires them to CLIP and Moment-DETR.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


# ---------- pure math ---------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def saliency_weighted_pool(v_feat: np.ndarray, saliency: np.ndarray,
                           tau: float = 0.1, top_p: float = 0.25) -> np.ndarray:
    """(D,) average of clip features weighted by softmax(saliency / tau) over the top_p clips."""
    assert v_feat.ndim == 2 and saliency.ndim == 1 and v_feat.shape[0] == saliency.shape[0]
    k = max(1, int(np.ceil(top_p * len(saliency))))
    keep = np.argpartition(-saliency, k - 1)[:k]
    masked = np.full(len(saliency), -np.inf)
    masked[keep] = saliency[keep]
    w = softmax(masked / tau)
    return (w[:, None] * v_feat).sum(axis=0)


def per_chunk_score(e_q: np.ndarray, e_T: np.ndarray, e_w: np.ndarray, alpha: float = 0.5) -> float:
    return alpha * cosine(e_q, e_w) + (1.0 - alpha) * cosine(e_T, e_w)


def aggregate_per_query(per_chunk: Sequence[float]) -> float:
    return float(max(per_chunk)) if len(per_chunk) else float("nan")


def aggregate_consistency(per_query: Sequence[float]) -> float:
    return float(np.mean(per_query)) if len(per_query) else float("nan")


def consistency_score(e_T: np.ndarray, e_q_list: Sequence[np.ndarray], chunks: Sequence[dict],
                      alpha: float = 0.5, tau: float = 0.1, top_p: float = 0.25) -> dict:
    """chunks: [{"v_feat": (T_w, D), "saliency": [K arrays of (T_w,)]}, ...]"""
    per_q_chunk = []
    for k, e_q in enumerate(e_q_list):
        per_q_chunk.append([
            per_chunk_score(e_q, e_T, saliency_weighted_pool(c["v_feat"], c["saliency"][k], tau, top_p), alpha)
            for c in chunks])
    per_q = [aggregate_per_query(s) for s in per_q_chunk]
    return {"C": aggregate_consistency(per_q), "per_query": per_q, "per_query_chunk": per_q_chunk}


def chunk_indices(n_clips: int, window: int = 75, stride: int = 60) -> List[Tuple[int, int]]:
    """(start, end) clip windows covering [0, n_clips). Short videos give one window."""
    if n_clips <= window:
        return [(0, n_clips)]
    out, s = [], 0
    while True:
        e = min(s + window, n_clips)
        out.append((s, e))
        if e == n_clips:
            return out
        s += stride


def chunk_features(v_feat: np.ndarray, window: int = 75, stride: int = 60) -> List[np.ndarray]:
    return [v_feat[s:e] for s, e in chunk_indices(v_feat.shape[0], window, stride)]


# ---------- model-backed scorer ------------------------------------------------

@dataclass
class ScoringConfig:
    alpha: float = 0.5
    tau: float = 0.1
    top_p: float = 0.25
    chunk_window: int = 75
    chunk_stride: int = 60

    @classmethod
    def from_config(cls, **overrides) -> "ScoringConfig":
        from src.config import cfg
        vals = {**cfg()["scoring"], **{k: v for k, v in overrides.items() if v is not None}}
        return cls(**vals)


class ConsistencyScorer:
    """Scores (thumbnail, video, queries). Load once, call score() many times."""

    def __init__(self, m1=None, feature_dir: str | Path | None = None, cfg: ScoringConfig | None = None):
        from src import config
        from src.moment_detr import MomentDETR
        self.m1 = m1 or MomentDETR(config.m1_checkpoint(), config.moment_detr_repo())
        self.feature_dir = Path(feature_dir or config.features_dir())
        self.cfg = cfg or ScoringConfig.from_config()

    def score(self, thumbnail_path: str, vid: str, queries: Sequence[str]) -> dict:
        from src.encoders import encode_image, encode_text, encode_text_tokens
        from src.moment_detr import load_video_features

        queries = [q for q in queries if q and q.strip()]
        if not queries:
            raise ValueError("no non-empty queries to score")
        e_T_t, ok = encode_image([str(thumbnail_path)])
        if not ok:
            raise ValueError(f"could not read thumbnail image: {thumbnail_path}")
        e_T = e_T_t[0].numpy()
        e_q = encode_text(list(queries)).numpy()
        t_tokens = encode_text_tokens(list(queries))   # M1 input: per-token features

        v_feat = load_video_features(self.feature_dir, vid)
        chunks, max_fg, best_windows = [], 0.0, []
        for s, e in chunk_indices(len(v_feat), self.cfg.chunk_window, self.cfg.chunk_stride):
            preds = self.m1.predict(v_feat[s:e], t_tokens)
            chunks.append({"v_feat": v_feat[s:e], "saliency": [p["saliency"] for p in preds]})
            for p in preds:
                top = p["windows"][0]
                if top[2] > max_fg:
                    max_fg = top[2]
                best_windows.append([top[0] + s * 2.0, top[1] + s * 2.0, top[2]])

        out = consistency_score(e_T, list(e_q), chunks, self.cfg.alpha, self.cfg.tau, self.cfg.top_p)
        out.update({
            "vid": vid, "thumbnail_path": str(thumbnail_path), "queries": list(queries),
            "n_chunks": len(chunks), "max_fg_prob": float(max_fg),
            "thumb_video_max_cos": float((v_feat / np.linalg.norm(v_feat, axis=1, keepdims=True) @ e_T).max()),
            "top_moments": best_windows,
        })
        return out
