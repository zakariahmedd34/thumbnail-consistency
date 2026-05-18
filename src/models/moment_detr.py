"""Wrapper around the official Moment-DETR checkpoint.

Two inference paths:

  (a) Raw video file       -> use the upstream `MomentDETRPredictor` from
      `Moment_DETR/run_on_video/run.py`. Useful for new videos and for
      ad-hoc smoke tests.

  (b) Pre-extracted features (the official `.npz` files we already have)
      -> use `predict_from_features(...)` directly. This is the fast path
      used by the evaluation loop; it avoids re-encoding video frames.

Both paths return the SAME dict structure:

  {
    "query":                  str,
    "pred_relevant_windows":  [[start_s, end_s, fg_score], ... 10 rows],
    "pred_saliency_scores":   [s_1, ..., s_L],   # length = n_clips
  }
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Sequence
import numpy as np
import torch
import torch.nn.functional as F


# ---------- internal helpers -------------------------------------------------

def _ensure_md_on_path(repo_root: str | Path) -> None:
    """Add Moment_DETR/moment_detr to sys.path so its internal imports resolve."""
    md = Path(repo_root) / "Moment_DETR" / "moment_detr"
    if str(md) not in sys.path:
        sys.path.insert(0, str(md))


# ---------- feature-path inference (FAST, primary) ---------------------------

class MomentDETRFeaturePredictor:
    """Run Moment-DETR using pre-extracted CLIP video + CLIP text features."""

    def __init__(self, ckpt_path: str, repo_root: str | Path,
                 device: str | None = None):
        _ensure_md_on_path(repo_root)
        from moment_detr.config import TestOptions
        from moment_detr.model import build_model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # The official ckpt also stores its training opts.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        opt = ckpt["opt"]
        # CRITICAL: opt.device was 'cuda' when the ckpt was trained. On CPU
        # runtimes, build_model() calls torch.cuda.* unconditionally and crashes
        # with "Torch not compiled with CUDA enabled" if we don't override it.
        opt.device = self.device
        # build_model expects a Namespace-like with the right attrs.
        model, _ = build_model(opt)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        self.model = model
        self.opt = opt
        self.clip_len = 2.0

    @torch.no_grad()
    def predict_from_features(
        self,
        v_feat: np.ndarray,            # (n_clips, 512) L2-normalized CLIP video feats
        t_feats: Sequence[np.ndarray], # list of (L_q, 512) CLIP text feats per query
        queries: Sequence[str] | None = None,
    ) -> List[Dict]:
        """Run M1 on N queries against ONE video. Returns list of N pred dicts."""
        from moment_detr.span_utils import span_cxw_to_xx
        from utils.tensor_utils import pad_sequences_1d

        n_clips = v_feat.shape[0]
        assert n_clips <= 75, f"video too long ({n_clips} clips > 75); use chunking"

        # video features with TEF
        v = torch.from_numpy(v_feat).float()
        v = F.normalize(v, dim=-1, eps=1e-5)
        tef_st = torch.arange(0, n_clips, 1.0) / n_clips
        tef_ed = tef_st + 1.0 / n_clips
        tef = torch.stack([tef_st, tef_ed], dim=1)
        v = torch.cat([v, tef], dim=1).to(self.device)             # (T, 514)
        n_q = len(t_feats)
        v = v.unsqueeze(0).repeat(n_q, 1, 1)                        # (N, T, 514)
        v_mask = torch.ones(n_q, n_clips, device=self.device)

        # text features
        t_tensors = [torch.from_numpy(t).float() for t in t_feats]
        t_padded, t_mask = pad_sequences_1d(
            t_tensors, dtype=torch.float32, device=self.device, fixed_length=None)
        t_padded = F.normalize(t_padded, dim=-1, eps=1e-5)

        outputs = self.model(
            src_vid=v, src_vid_mask=v_mask,
            src_txt=t_padded, src_txt_mask=t_mask,
        )
        prob = F.softmax(outputs["pred_logits"], -1)
        fg_scores = prob[..., 0]                                    # (N, 10)
        pred_spans = outputs["pred_spans"]                          # (N, 10, 2)
        saliency = outputs["saliency_scores"].float().cpu().numpy() # (N, T)

        duration = n_clips * self.clip_len
        results = []
        for i in range(n_q):
            spans = span_cxw_to_xx(pred_spans[i]).cpu() * duration  # (10, 2)
            rows = torch.cat([spans, fg_scores[i].cpu()[:, None]], dim=1).tolist()
            rows = sorted(rows, key=lambda r: r[2], reverse=True)
            rows = [[round(x, 4) for x in r] for r in rows]
            results.append({
                "query": queries[i] if queries else None,
                "pred_relevant_windows": rows,
                "pred_saliency_scores": [round(float(s), 4) for s in saliency[i].tolist()],
            })
        return results


# ---------- raw-video path (slower, ad-hoc) ----------------------------------

class MomentDETRVideoPredictor:
    """Thin shim over the upstream MomentDETRPredictor."""

    def __init__(self, ckpt_path: str, repo_root: str | Path,
                 clip_model: str = "ViT-B/32", device: str | None = None):
        _ensure_md_on_path(repo_root)
        from run_on_video.run import MomentDETRPredictor
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.impl = MomentDETRPredictor(
            ckpt_path=ckpt_path, clip_model_name_or_path=clip_model, device=dev)

    def predict(self, video_path: str, queries: List[str]) -> List[Dict]:
        return self.impl.localize_moment(video_path=video_path, query_list=queries)


# ---------- feature cache ----------------------------------------------------

def load_video_features(feature_dir: str | Path, vid: str) -> np.ndarray:
    """Load (n_clips, 512) CLIP video features by vid (matches feature filenames)."""
    path = Path(feature_dir) / f"{vid}.npz"
    return np.load(path)["features"]


def load_text_features(feature_dir: str | Path, qid: int) -> np.ndarray:
    """Load (≤32, 512) CLIP text features for a qid."""
    path = Path(feature_dir) / f"qid{qid}.npz"
    arr = np.load(path)
    # official files contain both 'last_hidden_state' (preferred) and 'pooler_output'.
    if "last_hidden_state" in arr:
        return arr["last_hidden_state"]
    return arr["features"]
