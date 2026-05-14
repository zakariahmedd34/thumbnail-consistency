"""End-to-end thumbnail-video consistency scorer.

Public API:
    scorer = ConsistencyScorer(m1_predictor, feature_dir, cfg)
    out    = scorer.score(thumbnail_path, vid, queries)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np

from src.models.encoders import encode_image, encode_text
from src.models.moment_detr import (
    MomentDETRFeaturePredictor, load_video_features,
)
from src.scoring.aggregate import (
    consistency_score, saliency_weighted_pool,
    per_chunk_score, aggregate_per_query, aggregate_consistency,
    confidence_damp,
)
from src.scoring.chunking import chunk_indices


@dataclass
class ScoringConfig:
    alpha: float = 0.5       # weight on cos(e_q, e_clip); 1-alpha on cos(e_T, e_clip)
    tau: float = 0.1         # softmax temperature on saliency
    top_p: float = 0.25      # fraction of clips kept by saliency
    chunk_window: int = 75   # M1 max input length
    chunk_stride: int = 60
    damp_with_fg: bool = False


class ConsistencyScorer:
    def __init__(
        self,
        m1: MomentDETRFeaturePredictor,
        feature_dir: str | Path,
        cfg: ScoringConfig | None = None,
    ):
        self.m1 = m1
        self.feature_dir = Path(feature_dir)
        self.cfg = cfg or ScoringConfig()

    # -- low-level: score a single (thumb, video, queries) tuple -------------

    def score(
        self,
        thumbnail_path: str,
        vid: str,
        queries: Sequence[str],
        precomputed_e_T: Optional[np.ndarray] = None,
        precomputed_e_q: Optional[np.ndarray] = None,
    ) -> dict:
        cfg = self.cfg

        # 1. Embed thumbnail.
        if precomputed_e_T is None:
            e_T_tensor, _ = encode_image([str(thumbnail_path)])
            assert e_T_tensor.shape[0] == 1, f"Failed to embed {thumbnail_path}"
            e_T = e_T_tensor[0].numpy()
        else:
            e_T = precomputed_e_T

        # 2. Embed queries (CLIP-text).
        if precomputed_e_q is None:
            e_q_arr = encode_text(list(queries)).numpy()
        else:
            e_q_arr = precomputed_e_q
        e_q_list = [e_q_arr[i] for i in range(e_q_arr.shape[0])]

        # 3. Load video features and chunk.
        v_feat = load_video_features(self.feature_dir, vid)  # (n_clips, 512)
        idxs = chunk_indices(v_feat.shape[0], cfg.chunk_window, cfg.chunk_stride)

        # 4. For each chunk and each query, run M1 to get saliency.
        #    We give M1 the query *text* via its own CLIP text encoder
        #    (this matches how M1 was trained).
        chunks = []
        max_fg = 0.0
        for (s, e) in idxs:
            v_w = v_feat[s:e]
            # Run M1 once for this chunk against all K queries.
            t_feats = self._embed_queries_for_m1(list(queries))
            preds = self.m1.predict_from_features(
                v_feat=v_w, t_feats=t_feats, queries=list(queries))
            sal_list = [np.asarray(p["pred_saliency_scores"], dtype=np.float32)
                        for p in preds]
            # Top fg prob across top-3 spans across queries (for damping).
            chunk_max_fg = max(
                max((row[2] for row in p["pred_relevant_windows"][:3]), default=0.0)
                for p in preds
            )
            max_fg = max(max_fg, chunk_max_fg)
            chunks.append({"v_feat": v_w, "saliency": sal_list})

        # 5. Aggregate.
        result = consistency_score(
            e_T=e_T, e_q_list=e_q_list, chunks=chunks,
            alpha=cfg.alpha, tau=cfg.tau, top_p=cfg.top_p,
            max_fg_prob=max_fg if cfg.damp_with_fg else None,
        )
        result.update({
            "vid": vid,
            "thumbnail_path": str(thumbnail_path),
            "queries": list(queries),
            "n_chunks": len(chunks),
            "max_fg_prob": max_fg,
        })
        return result

    # -- helper: encode query text using CLIP-text (frozen) ------------------

    def _embed_queries_for_m1(self, queries: List[str]) -> List[np.ndarray]:
        """Return list of (L_q, 512) CLIP token-level features per query.

        Note: M1 was trained on per-token last_hidden_state, not pooled.
        For V1 we approximate with pooled-then-broadcast: a single (1, 512)
        per query. This is the same approximation used by the upstream
        `MomentDETRPredictor.encode_text` in some configurations.
        """
        emb = encode_text(queries).numpy()       # (N, 512)
        return [emb[i:i + 1] for i in range(emb.shape[0])]
