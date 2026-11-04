"""Frozen CLIP ViT-B/32 encoders (loaded once, shared by everything).

Three outputs are used in the project:
  encode_image(paths)        -> (N, 512) L2-normalised image embeddings
  encode_text(texts)         -> (N, 512) L2-normalised pooled text embeddings
  encode_text_tokens(texts)  -> list of (L_i, 512) per-token features for Moment-DETR

Why two text encoders: Moment-DETR was trained on the per-token
`last_hidden_state` of CLIP's text model (official `clip_text_features/*.npz`),
not on the single pooled vector. Feeding it the pooled vector drops its
saliency HIT@1 from ~0.93 to ~0.43 (measured on 150 annotated QVH queries).
HF's last_hidden_state matches the official features (cosine 1.000).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

CLIP_NAME = "openai/clip-vit-base-patch32"
M1_MAX_TOKENS = 32  # Moment-DETR max_q_l


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_clip():
    """Return (model, processor, tokenizer, device). Frozen, eval mode."""
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    dev = device()
    model = CLIPModel.from_pretrained(CLIP_NAME).to(dev).eval()
    model.requires_grad_(False)
    return model, CLIPProcessor.from_pretrained(CLIP_NAME), CLIPTokenizer.from_pretrained(CLIP_NAME), dev


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def embed_pil(images: List[Image.Image]) -> torch.Tensor:
    """(N, 512) normalised embeddings for already-loaded PIL images."""
    model, proc, _, dev = get_clip()
    px = proc(images=images, return_tensors="pt")["pixel_values"].to(dev)
    pooled = model.vision_model(pixel_values=px).pooler_output
    return _l2(model.visual_projection(pooled)).cpu()


def encode_image(paths: List[str], batch: int = 32) -> Tuple[torch.Tensor, List[str]]:
    """Batch-encode image files. Returns (N, 512) embeddings and the paths that loaded.

    Unreadable files are skipped (and left out of the returned path list).
    """
    embs, kept = [], []
    for i in range(0, len(paths), batch):
        imgs, ok = [], []
        for p in paths[i:i + batch]:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(p)
            except (OSError, ValueError):
                continue
        if imgs:
            embs.append(embed_pil(imgs))
            kept.extend(ok)
    if not embs:
        return torch.zeros((0, 512)), []
    return torch.cat(embs), kept


@torch.no_grad()
def encode_text(texts: List[str], batch: int = 64) -> torch.Tensor:
    """(N, 512) normalised pooled text embeddings (CLIP's image-text space)."""
    model, _, tok, dev = get_clip()
    out = []
    for i in range(0, len(texts), batch):
        t = tok(texts[i:i + batch], padding=True, truncation=True, return_tensors="pt").to(dev)
        pooled = model.text_model(**t).pooler_output
        out.append(_l2(model.text_projection(pooled)).cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_text_tokens(texts: List[str], max_tokens: int = M1_MAX_TOKENS) -> List[np.ndarray]:
    """Per-token CLIP text features, the input format Moment-DETR was trained on."""
    model, _, tok, dev = get_clip()
    feats = []
    for q in texts:
        t = tok([q], truncation=True, max_length=77, return_tensors="pt").to(dev)
        h = model.text_model(**t).last_hidden_state[0, :max_tokens]
        feats.append(h.float().cpu().numpy())
    return feats
