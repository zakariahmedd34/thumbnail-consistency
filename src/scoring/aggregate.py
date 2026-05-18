"""Pure-math aggregation for the consistency score.

Implements the equations in FINAL_DESIGN Appendix A:

  e_clip_w  = sum_t softmax(s[t] / tau) * v[t]           (top-p mask applied)
  c(k, w)   = alpha * cos(e_qk, e_clip_w) + (1-alpha) * cos(e_T, e_clip_w)
  C(T, V)   = (1/K) * sum_k max_w c(k, w)                # multi-query mean of per-query max

All functions accept numpy arrays so they can be unit-tested without torch.
"""
from __future__ import annotations
from typing import Sequence, Optional
import numpy as np


# ---------- primitives -------------------------------------------------------

def _l2(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------- saliency-weighted pooling ---------------------------------------

def saliency_weighted_pool(
    v_feat: np.ndarray,           # (T, D) video clip features
    saliency: np.ndarray,         # (T,) per-clip saliency scores
    tau: float = 0.1,
    top_p: float = 0.25,
) -> np.ndarray:
    """Return (D,) saliency-softmax-weighted pooled video feature.

    Top-p of T clips by saliency are kept; the rest get weight 0.
    This stabilizes the pooling against long videos with diffuse saliency.
    """
    assert v_feat.ndim == 2 and saliency.ndim == 1
    assert v_feat.shape[0] == saliency.shape[0], "T mismatch"
    T = v_feat.shape[0]

    k = max(1, int(np.ceil(top_p * T)))
    # Keep top-k saliency indices; set others to -inf so softmax zeros them.
    keep_idx = np.argpartition(-saliency, k - 1)[:k]
    masked = np.full_like(saliency, -np.inf, dtype=np.float64)
    masked[keep_idx] = saliency[keep_idx].astype(np.float64)

    w = softmax(masked / tau)                  # (T,)
    pooled = (w[:, None] * v_feat).sum(axis=0) # (D,)
    return pooled


# ---------- per (query, chunk) score ----------------------------------------

def per_chunk_score(
    e_q: np.ndarray,              # (D,) CLIP-text emb of the query
    e_T: np.ndarray,              # (D,) CLIP-image emb of the thumbnail
    e_clip_w: np.ndarray,         # (D,) pooled video feature for chunk w
    alpha: float = 0.5,
) -> float:
    """c(k, w) = alpha * cos(e_q, e_clip_w) + (1-alpha) * cos(e_T, e_clip_w)."""
    return alpha * cosine(e_q, e_clip_w) + (1.0 - alpha) * cosine(e_T, e_clip_w)


# ---------- multi-query / multi-chunk aggregation ---------------------------

def aggregate_per_query(per_chunk: Sequence[float]) -> float:
    """max_w c(k, w) for one query across W chunks."""
    return float(max(per_chunk)) if len(per_chunk) else float("nan")


def aggregate_consistency(
    per_query: Sequence[float],
) -> float:
    """C(T, V) = mean_k max_w c(k, w). Multi-query mean of per-query max."""
    if not per_query:
        return float("nan")
    return float(np.mean(per_query))


# ---------- optional confidence damping --------------------------------------

def confidence_damp(score: float, max_fg_prob: float) -> float:
    """Apply foreground-prob damping (FINAL_DESIGN eq. 4)."""
    return float(score * (0.5 + 0.5 * max_fg_prob))


# ---------- the full per-pair score (composable) ----------------------------

def consistency_score(
    e_T: np.ndarray,                       # (D,) thumbnail emb
    e_q_list: Sequence[np.ndarray],        # K * (D,) per-query text embs
    chunks: Sequence[dict],                # each dict: v_feat (T,D), saliency_per_q [K * (T,)]
    alpha: float = 0.5,
    tau: float = 0.1,
    top_p: float = 0.25,
    max_fg_prob: Optional[float] = None,   # for optional damping; None disables
) -> dict:
    """Compute C(T, V) plus per-query / per-chunk breakdowns.

    Each `chunk` dict has keys:
      - "v_feat"      : (T_w, D)
      - "saliency"    : list of K arrays (T_w,), one per query
      - (optional) "max_fg_prob": float (for damping)

    Returns:
      {
        "C": float, "C_damped": float|None,
        "per_query": [K floats], "per_query_chunk": [K x W floats]
      }
    """
    K = len(e_q_list)
    per_q_chunk = []
    per_q = []
    for k in range(K):
        chunk_scores = []
        for w_idx, ck in enumerate(chunks):
            pooled = saliency_weighted_pool(
                ck["v_feat"], ck["saliency"][k], tau=tau, top_p=top_p)
            c_kw = per_chunk_score(e_q_list[k], e_T, pooled, alpha=alpha)
            chunk_scores.append(c_kw)
        per_q_chunk.append(chunk_scores)
        per_q.append(aggregate_per_query(chunk_scores))

    C = aggregate_consistency(per_q)
    C_damped = confidence_damp(C, max_fg_prob) if max_fg_prob is not None else None
    return {
        "C": C, "C_damped": C_damped,
        "per_query": per_q, "per_query_chunk": per_q_chunk,
    }
