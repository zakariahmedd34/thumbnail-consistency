"""Unit tests for src.scoring.aggregate — pure math, no GPU."""
from __future__ import annotations
import numpy as np
import pytest

from src.scoring.aggregate import (
    cosine, softmax, saliency_weighted_pool, per_chunk_score,
    aggregate_per_query, aggregate_consistency, confidence_damp,
    consistency_score,
)
from src.scoring.chunking import chunk_indices, chunk_features
from src.eval.metrics import roc_auc, cohens_d, summarize_separation


# ---------- primitives ------------------------------------------------------

def test_cosine_identical_is_one():
    v = np.array([0.3, 0.7, -0.1, 0.5])
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-9)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cosine_opposite_is_minus_one():
    a = np.array([1.0, 1.0])
    b = -a
    assert cosine(a, b) == pytest.approx(-1.0, abs=1e-9)


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0, -1.0])
    w = softmax(x)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w >= 0).all()


# ---------- saliency pool ---------------------------------------------------

def test_pool_picks_high_saliency_clip():
    # Build a tiny case: 4 clips, salient one points to a known direction.
    T, D = 4, 8
    rng = np.random.default_rng(0)
    target = rng.normal(size=D)
    other  = rng.normal(size=D)
    v = np.stack([other, other, target, other])
    sal = np.array([0.1, 0.1, 5.0, 0.1])  # clip 2 dominates
    pooled = saliency_weighted_pool(v, sal, tau=0.1, top_p=0.25)
    # With top_p=0.25 -> k=1, only clip 2 survives.
    assert cosine(pooled, target) == pytest.approx(1.0, abs=1e-6)


def test_pool_top_p_threshold():
    T, D = 10, 4
    v = np.eye(T, D)
    v = v[:, :D] if T >= D else np.tile(np.eye(D), (T // D + 1, 1))[:T]
    sal = np.linspace(0.0, 1.0, T)
    pooled_quarter = saliency_weighted_pool(v, sal, tau=0.1, top_p=0.25)
    pooled_full    = saliency_weighted_pool(v, sal, tau=0.1, top_p=1.0)
    # Differ as soon as top_p changes (sanity).
    assert not np.allclose(pooled_quarter, pooled_full)


# ---------- per chunk + aggregation -----------------------------------------

def test_per_chunk_score_endpoints():
    e_q = np.array([1.0, 0.0])
    e_T = np.array([0.0, 1.0])
    e_v = np.array([1.0, 0.0])
    # alpha=0.5: 0.5 * cos(qq) + 0.5 * cos(Tv) = 0.5 * 1 + 0.5 * 0 = 0.5
    assert per_chunk_score(e_q, e_T, e_v, alpha=0.5) == pytest.approx(0.5, abs=1e-9)
    assert per_chunk_score(e_q, e_T, e_v, alpha=1.0) == pytest.approx(1.0, abs=1e-9)
    assert per_chunk_score(e_q, e_T, e_v, alpha=0.0) == pytest.approx(0.0, abs=1e-9)


def test_aggregate_per_query_is_max():
    assert aggregate_per_query([0.1, 0.9, 0.3]) == 0.9


def test_aggregate_consistency_is_mean():
    assert aggregate_consistency([1.0, 0.0]) == pytest.approx(0.5, abs=1e-9)


def test_confidence_damp_monotone():
    base = 0.6
    damp_0 = confidence_damp(base, 0.0)
    damp_1 = confidence_damp(base, 1.0)
    assert damp_0 == pytest.approx(0.5 * base)
    assert damp_1 == pytest.approx(base)


# ---------- full path -------------------------------------------------------

def test_consistency_score_shapes_and_aggregation():
    D, T = 8, 6
    rng = np.random.default_rng(42)
    target = rng.normal(size=D)
    e_T = target / np.linalg.norm(target)
    e_q = e_T.copy()
    # Build 2 chunks. In chunk 0 the salient clip points to e_T; chunk 1 random.
    v0 = rng.normal(size=(T, D)); v0[3] = target * 4
    v1 = rng.normal(size=(T, D))
    sal0 = np.array([0.1, 0.1, 0.2, 5.0, 0.1, 0.1])
    sal1 = rng.normal(size=T)
    chunks = [
        {"v_feat": v0, "saliency": [sal0]},
        {"v_feat": v1, "saliency": [sal1]},
    ]
    out = consistency_score(e_T=e_T, e_q_list=[e_q], chunks=chunks)
    assert isinstance(out["C"], float)
    assert len(out["per_query"]) == 1
    assert len(out["per_query_chunk"]) == 1
    assert len(out["per_query_chunk"][0]) == 2
    # max over chunks is from chunk 0 (the planted one) and should be high.
    assert out["per_query"][0] > 0.7


# ---------- chunking --------------------------------------------------------

def test_chunk_indices_short_video_single_window():
    assert chunk_indices(50, window=75, stride=60) == [(0, 50)]
    assert chunk_indices(75, window=75, stride=60) == [(0, 75)]


def test_chunk_indices_long_video_overlapping():
    idx = chunk_indices(180, window=75, stride=60)
    assert idx[0] == (0, 75)
    assert idx[1] == (60, 135)
    assert idx[2] == (120, 180)


def test_chunk_features_shapes():
    arr = np.zeros((200, 4))
    out = chunk_features(arr, window=75, stride=60)
    assert all(c.shape[1] == 4 for c in out)
    assert sum(1 for c in out if c.shape[0] <= 75) == len(out)


# ---------- metrics ---------------------------------------------------------

def test_roc_auc_perfect_separation():
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    labels = [1, 1, 1, 0, 0, 0]
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_inverted():
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
    labels = [1, 1, 1, 0, 0, 0]
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_random_around_half():
    rng = np.random.default_rng(0)
    n = 500
    s = rng.uniform(0, 1, size=2 * n)
    l = [1] * n + [0] * n
    auc = roc_auc(s, l)
    assert 0.4 < auc < 0.6


def test_summarize_separation_keys():
    out = summarize_separation([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0])
    for k in ("auc", "cohens_d", "welch_t", "welch_p",
              "mean_pos", "mean_neg", "n_pos", "n_neg"):
        assert k in out
