"""ClipCap M2 generator — the trained component in V1.

Wraps the existing src.models.clipcap.ClipCapM2 inside the unified interface.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Union
from .base import M2Generator
from . import register


@register("clipcap")
class ClipCapGenerator(M2Generator):
    """Frozen-GPT-2 + trained prefix-MLP captioner."""

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        prefix_len: int = 10,
        gpt2_name: str = "gpt2",
    ):
        import torch
        from src.models.clipcap import ClipCapM2
        from src.models.encoders import get_clip

        self.model = ClipCapM2(prefix_len=prefix_len, gpt2_name=gpt2_name)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict)
        self.clip_model, self.clip_proc, _, self.device = get_clip()
        self.model.to(self.device).eval()
        self._ckpt_path = str(ckpt_path)

    @property
    def name(self) -> str:
        return "clipcap"

    @property
    def model_version(self) -> str:
        return Path(self._ckpt_path).name

    def generate(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.1,
        max_new_tokens: int = 25,
        **kwargs,
    ) -> List[str]:
        import torch
        from PIL import Image

        # Encode the thumbnail via the same defensive vision_model bypass as
        # src/models/encoders.py.
        img = Image.open(thumbnail).convert("RGB")
        with torch.no_grad():
            inp = self.clip_proc(images=img, return_tensors="pt").to(self.device)
            out = self.clip_model.vision_model(pixel_values=inp["pixel_values"])
            pooled = (
                out.pooler_output
                if (hasattr(out, "pooler_output") and out.pooler_output is not None)
                else out.last_hidden_state[:, 0]
            )
            proj_dim      = self.clip_model.config.projection_dim
            vision_hidden = self.clip_model.config.vision_config.hidden_size
            if pooled.shape[-1] == vision_hidden:
                e_T = self.clip_model.visual_projection(pooled)
            elif pooled.shape[-1] == proj_dim:
                e_T = pooled
            else:
                raise RuntimeError(f"Unexpected pooled dim: {pooled.shape}")
            e_T = e_T / e_T.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            # ClipCapM2.generate returns List[List[str]] indexed by batch dim.
            outs = self.model.generate(
                e_T,
                num_return_sequences=k,
                do_sample=k > 1,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=2,
            )
        # outs[0] is the K-string list for the single batch element.
        return [s.strip() for s in outs[0]]
