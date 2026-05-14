"""Evaluation metrics for the consistency pipeline."""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (no sklearn dep)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert set(np.unique(labels).tolist()).issubset({0, 1}), "labels must be 0/1"
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[:len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def cohens_d(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s_p = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) +
                   (len(neg) - 1) * neg.var(ddof=1)) /
                  (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / (s_p + 1e-12))


def welch_t(pos: Sequence[float], neg: Sequence[float]) -> Tuple[float, float]:
    """Welch's t statistic and p-value (two-sided) via scipy if available."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    try:
        from scipy import stats
        t, p = stats.ttest_ind(pos, neg, equal_var=False)
        return float(t), float(p)
    except ImportError:
        # Fallback: only the t statistic, no p-value
        mp, mn = pos.mean(), neg.mean()
        vp, vn = pos.var(ddof=1), neg.var(ddof=1)
        t = (mp - mn) / np.sqrt(vp / len(pos) + vn / len(neg) + 1e-12)
        return float(t), float("nan")


def summarize_separation(
    scores: Sequence[float], labels: Sequence[int]
) -> dict:
    """Headline numbers for the consistency-eval set."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    auc = roc_auc(scores, labels)
    d = cohens_d(pos, neg)
    t, p = welch_t(pos, neg)
    return {
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "mean_pos": float(pos.mean()) if len(pos) else float("nan"),
        "mean_neg": float(neg.mean()) if len(neg) else float("nan"),
        "auc": auc,
        "cohens_d": d,
        "welch_t": t,
        "welch_p": p,
    }


def confusion_at_threshold(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = (scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {"threshold": float(threshold),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1)}
