"""PyTorch Dataset for M2 (frame, query) pairs.

Loads frame paths + queries from the JSONL produced by data_collection_preparation,
encodes frames lazily through frozen CLIP (or uses cached embeddings if available).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class M2PairsDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],   # parsed JSONL records
        tokenizer,                       # GPT-2 tokenizer (from ClipCapM2.tokenizer)
        clip_processor,                  # transformers CLIPProcessor
        clip_model,                      # frozen CLIP model (image features only)
        device: str = "cuda",
        max_caption_tokens: int = 25,
        cached_emb_path: Optional[str] = None,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.clip_proc = clip_processor
        self.clip_model = clip_model
        self.device = device
        self.max_caption_tokens = max_caption_tokens

        # Optional precomputed embeddings keyed by qid (huge speedup).
        self._cache = None
        if cached_emb_path and Path(cached_emb_path).exists():
            arr = np.load(cached_emb_path)
            self._cache = {int(q): arr["emb"][i]
                           for i, q in enumerate(arr["qids"].tolist())}

    def __len__(self) -> int:
        return len(self.records)

    @torch.no_grad()
    def _embed(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        inp = self.clip_proc(images=img, return_tensors="pt").to(self.device)
        e = self.clip_model.get_image_features(**inp)
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return e.squeeze(0).cpu()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        qid = r["qid"]

        if self._cache is not None and qid in self._cache:
            clip_feat = torch.from_numpy(self._cache[qid]).float()
        else:
            clip_feat = self._embed(r["frame_path"])

        # Caption: <bos> query <eos>
        eos = self.tokenizer.eos_token
        text = (eos + " " + r["query"].strip() + " " + eos)
        toks = self.tokenizer(
            text, max_length=self.max_caption_tokens,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "clip_feat": clip_feat,                            # (512,)
            "input_ids": toks["input_ids"].squeeze(0),         # (T,)
            "attention_mask": toks["attention_mask"].squeeze(0),
            "qid": qid,
            "query": r["query"],
        }


def collate_m2(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {
        "clip_feat":      torch.stack([b["clip_feat"] for b in batch]),
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "qid":            torch.tensor([b["qid"] for b in batch]),
        "query":          [b["query"] for b in batch],
    }
    return out
