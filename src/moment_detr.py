"""Moment-DETR (M1) inference on pre-extracted CLIP features.

M1 is the frozen official checkpoint from github.com/jayleicn/moment_detr
(cloned into Moment_DETR/moment_detr). Given a video's per-clip CLIP features
and a text query, it returns a saliency score per 2-second clip plus 10
candidate moment spans.

Inputs must match training:
  video: (T <= 75, 512) CLIP image features, one per 2 s clip (+ 2 TEF dims added here)
  text : (L <= 32, 512) per-token CLIP text features (see encoders.encode_text_tokens)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CLIP_SECONDS = 2.0
MAX_CLIPS = 75


def _import_upstream(repo: Path) -> None:
    if not (repo / "moment_detr" / "model.py").exists():
        raise FileNotFoundError(
            f"Moment-DETR code not found at {repo}. Run:\n"
            f"  git clone https://github.com/jayleicn/moment_detr.git {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


class MomentDETR:
    def __init__(self, ckpt_path: str | Path, repo: str | Path, device: str | None = None):
        _import_upstream(Path(repo))
        from moment_detr.model import build_model

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Moment-DETR checkpoint not found: {ckpt_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        opt = ckpt["opt"]
        opt.device = self.device  # ckpt stores 'cuda'; build_model crashes on CPU otherwise
        model, _ = build_model(opt)
        model.load_state_dict(ckpt["model"])
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, v_feat: np.ndarray, t_feats: Sequence[np.ndarray]) -> List[Dict]:
        """Run M1 for N queries against one video window.

        Returns N dicts: {"saliency": (T,) array, "windows": [[start_s, end_s, fg_prob], ...]}
        sorted by fg_prob (highest first).
        """
        from moment_detr.span_utils import span_cxw_to_xx
        from utils.tensor_utils import pad_sequences_1d

        n_clips = v_feat.shape[0]
        if n_clips > MAX_CLIPS:
            raise ValueError(f"{n_clips} clips > {MAX_CLIPS}; chunk the video first")

        v = F.normalize(torch.from_numpy(v_feat).float(), dim=-1, eps=1e-5)
        tef_st = torch.arange(0, n_clips, 1.0) / n_clips
        tef = torch.stack([tef_st, tef_st + 1.0 / n_clips], dim=1)  # temporal endpoint features
        v = torch.cat([v, tef], dim=1).to(self.device)
        n_q = len(t_feats)
        v = v.unsqueeze(0).repeat(n_q, 1, 1)
        v_mask = torch.ones(n_q, n_clips, device=self.device)

        t, t_mask = pad_sequences_1d([torch.from_numpy(x).float() for x in t_feats],
                                     dtype=torch.float32, device=self.device, fixed_length=None)
        t = F.normalize(t, dim=-1, eps=1e-5)

        out = self.model(src_vid=v, src_vid_mask=v_mask, src_txt=t, src_txt_mask=t_mask)
        fg = F.softmax(out["pred_logits"], -1)[..., 0].cpu()           # (N, 10)
        spans = span_cxw_to_xx(out["pred_spans"]).cpu() * n_clips * CLIP_SECONDS
        sal = out["saliency_scores"].float().cpu().numpy()             # (N, T)

        results = []
        for i in range(n_q):
            rows = torch.cat([spans[i], fg[i][:, None]], dim=1).tolist()
            rows.sort(key=lambda r: r[2], reverse=True)
            results.append({"saliency": sal[i], "windows": [[round(x, 4) for x in r] for r in rows]})
        return results


def load_video_features(feature_dir: str | Path, vid: str) -> np.ndarray:
    """(n_clips, 512) official CLIP video features for a QVHighlights vid."""
    path = Path(feature_dir) / f"{vid}.npz"
    if not path.exists():
        raise FileNotFoundError(f"video features not found: {path}")
    return np.load(path)["features"].astype(np.float32)
