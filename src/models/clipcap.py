"""ClipCap-style M2: CLIP image -> prefix MLP -> frozen GPT-2 -> query string.

The ★ TRAINED component in V1. ~12M params in `prefix`, GPT-2 stays frozen.

Inputs
------
  - clip_feat : (B, 512)  CLIP image embedding (already pooled, L2-normalized)
  - input_ids : (B, T)    GPT-2 token ids (teacher forcing during train)
  - attn      : (B, T)    attention mask for input_ids

Outputs
-------
  - logits over GPT-2 vocab, aligned with input_ids tokens
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn


class ClipCapM2(nn.Module):
    def __init__(
        self,
        clip_dim: int = 512,
        prefix_len: int = 10,
        gpt2_name: str = "gpt2",
        freeze_gpt2: bool = True,
    ):
        super().__init__()
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        if freeze_gpt2:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        self.prefix_len = prefix_len
        d = self.gpt2.config.n_embd  # 768 for gpt2-small

        # Two-layer MLP: 512 -> prefix_len * 768, then refine.
        hidden = max(512, d * prefix_len // 2)
        self.prefix = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d * prefix_len),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _expand_prefix(self, clip_feat: torch.Tensor) -> torch.Tensor:
        b = clip_feat.size(0)
        p = self.prefix(clip_feat)
        return p.view(b, self.prefix_len, -1)  # (B, P, d)

    def forward(
        self,
        clip_feat: torch.Tensor,        # (B, 512)
        input_ids: torch.Tensor,        # (B, T)
        attention_mask: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        prefix = self._expand_prefix(clip_feat)
        tok_emb = self.gpt2.transformer.wte(input_ids)
        inp = torch.cat([prefix, tok_emb], dim=1)
        pre_mask = torch.ones(prefix.size(0), self.prefix_len,
                              device=clip_feat.device, dtype=attention_mask.dtype)
        attn = torch.cat([pre_mask, attention_mask], dim=1)
        out = self.gpt2(inputs_embeds=inp, attention_mask=attn)
        # Drop the prefix slice from logits; caller compares against input_ids.
        return out.logits[:, self.prefix_len:, :]

    @torch.no_grad()
    def generate(
        self,
        clip_feat: torch.Tensor,        # (B, 512)
        max_new_tokens: int = 25,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 1.1,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 2,
    ) -> List[List[str]]:
        """Return List[B] of List[K] strings."""
        self.eval()
        prefix = self._expand_prefix(clip_feat)  # (B, P, d)
        b = prefix.size(0)
        out_per_b: List[List[str]] = []

        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        for i in range(b):
            p_i = prefix[i:i + 1]  # (1, P, d)
            input_ids = torch.tensor([[bos]], device=clip_feat.device)
            input_emb = self.gpt2.transformer.wte(input_ids)
            inp_emb = torch.cat([p_i, input_emb], dim=1)
            pre_mask = torch.ones(1, self.prefix_len + 1,
                                  device=clip_feat.device, dtype=torch.long)
            gen = self.gpt2.generate(
                inputs_embeds=inp_emb,
                attention_mask=pre_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # `inputs_embeds` path: `gen` contains ONLY newly-generated token ids.
            strs = [self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                    for seq in gen]
            out_per_b.append(strs)
        return out_per_b

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
