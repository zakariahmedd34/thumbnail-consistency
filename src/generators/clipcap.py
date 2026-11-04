"""ClipCap generator: CLIP image embedding -> trained prefix MLP -> frozen GPT-2 -> query.

The prefix MLP is the only trained component in the project. Train it with
    python -m src.cli train-clipcap
which writes checkpoints/clipcap_best.pt (the path in config.yaml).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List

from . import Generator, register


def _build_model(prefix_len: int = 10, gpt2_name: str = "gpt2", clip_dim: int = 512):
    import torch.nn as nn
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    class ClipCapM2(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
            self.gpt2.requires_grad_(False)
            self.prefix_len = prefix_len
            d = self.gpt2.config.n_embd
            hidden = max(512, d * prefix_len // 2)
            self.prefix = nn.Sequential(nn.Linear(clip_dim, hidden), nn.Tanh(),
                                        nn.Linear(hidden, d * prefix_len))

        def _prefix(self, clip_feat):
            return self.prefix(clip_feat).view(clip_feat.size(0), self.prefix_len, -1)

        def forward(self, clip_feat, input_ids, attention_mask):
            import torch
            prefix = self._prefix(clip_feat)
            inp = torch.cat([prefix, self.gpt2.transformer.wte(input_ids)], dim=1)
            pre_mask = torch.ones(prefix.shape[:2], device=clip_feat.device, dtype=attention_mask.dtype)
            out = self.gpt2(inputs_embeds=inp, attention_mask=torch.cat([pre_mask, attention_mask], 1))
            return out.logits[:, self.prefix_len:, :]

        def generate(self, clip_feat, k: int = 1, max_new_tokens: int = 25, do_sample: bool = True,
                     top_p: float = 0.9, temperature: float = 1.0, num_beams: int = 1) -> List[List[str]]:
            import torch
            self.eval()
            prefix = self._prefix(clip_feat)
            bos = self.tokenizer.eos_token_id
            outs = []
            with torch.no_grad():
                for i in range(prefix.size(0)):
                    ids = torch.tensor([[bos]], device=clip_feat.device)
                    emb = torch.cat([prefix[i:i + 1], self.gpt2.transformer.wte(ids)], dim=1)
                    gen = self.gpt2.generate(
                        inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=emb.device, dtype=torch.long),
                        max_new_tokens=max_new_tokens, num_return_sequences=k, do_sample=do_sample,
                        top_p=top_p if do_sample else None, temperature=temperature if do_sample else None,
                        num_beams=num_beams, no_repeat_ngram_size=2,
                        pad_token_id=bos, eos_token_id=bos)
                    outs.append([self.tokenizer.decode(s, skip_special_tokens=True).strip() for s in gen])
            return outs

    return ClipCapM2()


@register("clipcap")
class ClipCap(Generator):
    def __init__(self, ckpt_path: str, prefix_len: int = 10, gpt2_name: str = "gpt2"):
        import torch
        from src.encoders import get_clip
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"ClipCap checkpoint not found: {ckpt_path}. Train it with: python -m src.cli train-clipcap")
        self.model = _build_model(prefix_len, gpt2_name)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt.get("model", ckpt))
        self.device = get_clip()[3]
        self.model.to(self.device).eval()

    def generate(self, image_path, k: int = 1, **_):
        from src.encoders import encode_image
        emb, ok = encode_image([str(image_path)])
        if not ok:
            raise ValueError(f"could not read image: {image_path}")
        emb = emb.to(self.device)
        # Query 1: beam search (most likely caption). Extra queries: nucleus sampling for diversity.
        queries = self.model.generate(emb, k=1, do_sample=False, num_beams=3)[0]
        if k > 1:
            queries += self.model.generate(emb, k=k - 1, do_sample=True, top_p=0.9, temperature=1.0)[0]
        return {"queries": queries, "meta": {}}


# ---------- training -----------------------------------------------------------

def train(epochs: int | None = None, out_path: str | None = None, max_train: int | None = None) -> Path:
    """Train the prefix MLP on (frame, query) pairs. Returns the checkpoint path."""
    import torch
    from src import config
    from src.encoders import encode_image, encode_text, get_clip
    import numpy as np

    c = config.cfg()["clipcap_training"]
    epochs = epochs or c["epochs"]
    config.set_seed(c["seed"])
    dev = get_clip()[3]
    out_path = Path(out_path or config.generator_kwargs("clipcap")["ckpt_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = _build_model(config.cfg()["generators"]["clipcap"].get("prefix_len", 10)).to(dev)
    tok = model.tokenizer

    def load_split(key, emb_key=None):
        recs = config.load_jsonl(config.data_path(config.cfg()["data"][key]))
        cache = {}
        if emb_key:
            arr = np.load(config.data_path(config.cfg()["data"][emb_key]))
            emb, qids = arr["emb"], arr["qids"]  # read once (each arr[...] access re-reads the file)
            cache = {int(q): emb[i] for i, q in enumerate(qids.tolist())}
        need = [r for r in recs if int(r["qid"]) not in cache]
        if need:
            emb, ok = encode_image([str(config.data_path(r["frame_path"])) for r in need])
            okset = dict(zip(ok, emb.numpy()))
            for r in need:
                p = str(config.data_path(r["frame_path"]))
                if p in okset:
                    cache[int(r["qid"])] = okset[p]
        recs = [r for r in recs if int(r["qid"]) in cache]
        feats = torch.tensor(np.stack([cache[int(r["qid"])] for r in recs]), dtype=torch.float32)
        text = [tok.eos_token + " " + r["query"].strip() + tok.eos_token for r in recs]
        t = tok(text, max_length=c["max_caption_tokens"], truncation=True, padding="max_length", return_tensors="pt")
        return recs, feats, t["input_ids"], t["attention_mask"]

    tr, tr_f, tr_ids, tr_m = load_split("clipcap_train", "clipcap_train_emb")
    va, va_f, _, _ = load_split("clipcap_val")
    if max_train:
        tr_f, tr_ids, tr_m = tr_f[:max_train], tr_ids[:max_train], tr_m[:max_train]
    print(f"train pairs={len(tr_f)}  val pairs={len(va)}  device={dev}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=c["weight_decay"])
    bs = c["batch_size"]
    steps = math.ceil(len(tr_f) / bs) * epochs
    warm = max(1, int(steps * c["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=c["label_smoothing"])

    def val_sim(n=200):
        gens = [g[0] for g in model.generate(va_f[:n].to(dev), k=1, do_sample=False, num_beams=3, max_new_tokens=20)]
        e_g, e_t = encode_text(gens).numpy(), encode_text([r["query"] for r in va[:n]]).numpy()
        return float((e_g * e_t).sum(1).mean()), gens[:3]

    best, since_best, history = -1.0, 0, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = torch.randperm(len(tr_f))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            ids, m = tr_ids[idx].to(dev), tr_m[idx].to(dev)
            logits = model(tr_f[idx].to(dev), ids[:, :-1], m[:, :-1])
            # Labels: next token; padding masked by attention mask (NOT by token id, because
            # GPT-2's pad token == its end token, and the model must learn to emit the end token).
            labels = ids[:, 1:].masked_fill(m[:, 1:] == 0, -100)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c["grad_clip"])
            opt.step(); sched.step(); tot += loss.item() * len(idx)
        sim, examples = val_sim()
        history.append({"epoch": ep, "train_loss": tot / len(tr_f), "val_clip_sim": sim, "seconds": time.time() - t0})
        print(f"epoch {ep:02d}  loss {tot / len(tr_f):.4f}  val CLIP-text sim {sim:.4f}  e.g. {examples[0]!r}")
        if sim > best:
            best, since_best = sim, 0
            torch.save({"model": model.state_dict(), "history": history, "best_sim": best}, out_path)
        else:
            since_best += 1
            if since_best >= c["early_stop_patience"]:
                print(f"early stop (best val sim {best:.4f})")
                break
    (out_path.parent / "clipcap_history.json").write_text(json.dumps(history, indent=2))
    print(f"saved best checkpoint -> {out_path}")
    return out_path
