"""Cached CLIP encoder loaders.

All three encoders are loaded once and shared:
  - CLIP-image (vision): used for thumbnail + frame embeddings, and inside M2.
  - CLIP-text:           used to encode generated queries before M1.
  - GPT-2 small + tokenizer: used by ClipCap (M2).

Lazy + singleton: first call loads to GPU; subsequent calls return cached.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Tuple, List
import torch
from PIL import Image

_CLIP_NAME = "openai/clip-vit-base-patch32"
_GPT2_NAME = "gpt2"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_clip():
    """Return (model, processor, tokenizer, device). Frozen, eval mode."""
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    dev = _device()
    model = CLIPModel.from_pretrained(_CLIP_NAME).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    proc = CLIPProcessor.from_pretrained(_CLIP_NAME)
    tok = CLIPTokenizer.from_pretrained(_CLIP_NAME)
    return model, proc, tok, dev


@lru_cache(maxsize=1)
def get_gpt2():
    """Return (model, tokenizer, device). Frozen, eval mode."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    dev = _device()
    model = GPT2LMHeadModel.from_pretrained(_GPT2_NAME).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    tok = GPT2Tokenizer.from_pretrained(_GPT2_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, dev


@torch.no_grad()
def _vision_pooled(model, pixel_values):
    """Return post-pooler feature from vision_model, regardless of HF version."""
    out = model.vision_model(pixel_values=pixel_values)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if isinstance(out, tuple):
        return out[1] if len(out) > 1 else out[0][:, 0]
    raise RuntimeError(f"Unexpected vision_model output: {type(out)}")


@torch.no_grad()
def _text_pooled(model, input_ids, attention_mask):
    """Return post-pooler feature from text_model, regardless of HF version."""
    out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if isinstance(out, tuple):
        return out[1] if len(out) > 1 else out[0][:, 0]
    raise RuntimeError(f"Unexpected text_model output: {type(out)}")


@torch.no_grad()
def encode_image(paths: List[str], batch: int = 32) -> Tuple["torch.Tensor", List[str]]:
    """Batch-encode images. Returns (N, 512) L2-normalized + list of successful paths.

    Bypasses model.get_image_features() because some transformers versions return
    BaseModelOutputWithPooling there instead of a tensor, breaking `.norm()`.
    """
    model, proc, _, dev = get_clip()
    vision_hidden = model.config.vision_config.hidden_size   # 768 for ViT-B/32
    proj_dim      = model.config.projection_dim              # 512 for ViT-B/32

    embs, kept = [], []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs, ok = [], []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(p)
            except Exception:
                continue
        if not imgs:
            continue
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        pooled = _vision_pooled(model, inp["pixel_values"])
        # Apply visual_projection only if pooled feature is in vision_hidden space
        if pooled.shape[-1] == vision_hidden:
            e = model.visual_projection(pooled)            # 768 -> 512
        elif pooled.shape[-1] == proj_dim:
            e = pooled                                     # already 512
        else:
            raise RuntimeError(
                f"Unexpected pooled image dim {pooled.shape[-1]}; "
                f"expected {vision_hidden} or {proj_dim}."
            )
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        embs.append(e.cpu())
        kept.extend(ok)
    if not embs:
        return torch.zeros((0, proj_dim)), []
    return torch.cat(embs, dim=0), kept


@torch.no_grad()
def encode_text(texts: List[str], batch: int = 64) -> "torch.Tensor":
    """Batch-encode texts. Returns (N, 512) L2-normalized.

    Bypasses model.get_text_features() for the same HF-version reason as
    encode_image.
    """
    model, _, tok, dev = get_clip()
    text_hidden = model.config.text_config.hidden_size       # 512 for ViT-B/32
    proj_dim    = model.config.projection_dim                # 512

    embs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        toks = tok(chunk, padding=True, truncation=True, return_tensors="pt").to(dev)
        pooled = _text_pooled(model, toks["input_ids"], toks["attention_mask"])
        if pooled.shape[-1] == text_hidden:
            e = model.text_projection(pooled)
        elif pooled.shape[-1] == proj_dim:
            e = pooled
        else:
            raise RuntimeError(
                f"Unexpected pooled text dim {pooled.shape[-1]}; "
                f"expected {text_hidden} or {proj_dim}."
            )
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)
