"""Vision-language-model generator (Qwen2-VL by default; any HF image-text-to-text chat model)."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from . import Generator, register

PROMPT = "Describe what is happening in this image in one short sentence."


def clean_caption(text: str) -> str:
    """First sentence, no markdown/quotes/prefixes, single spaces."""
    t = text.strip().strip("`*\"' ")
    t = re.sub(r"^(caption|answer|description)\s*:\s*", "", t, flags=re.I)
    t = re.split(r"(?<=[.!?])\s|\n", t, maxsplit=1)[0]
    return re.sub(r"\s+", " ", t).strip().strip("\"'")


@register("vlm")
class VLM(Generator):
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", load_in_4bit: bool = True,
                 max_pixels: int | None = 401408, max_new_tokens: int = 40,
                 min_grounding: float = 0.0, prompt: str = PROMPT):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        kw: Dict[str, Any] = {"device_map": "auto", "dtype": torch.float16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        pkw = {"max_pixels": max_pixels} if max_pixels and "qwen" in model_id.lower() else {}
        self.processor = AutoProcessor.from_pretrained(model_id, **pkw)
        self.model_id, self.max_new_tokens = model_id, max_new_tokens
        self.min_grounding, self.prompt = min_grounding, prompt

    def raw(self, image_path: str, prompt: str | None = None, do_sample: bool = False,
            temperature: float = 1.0, max_new_tokens: int | None = None, n: int = 1) -> List[str]:
        """Raw decoded model outputs (no cleaning) - useful for debugging."""
        import torch
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt or self.prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        gen_kw = dict(max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=do_sample,
                      num_return_sequences=n)
        if do_sample:
            gen_kw.update(temperature=temperature, top_p=0.9)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)
        start = inputs["input_ids"].shape[1]
        return [self.processor.decode(o[start:], skip_special_tokens=True).strip() for o in out]

    def generate(self, image_path, k: int = 1, **_):
        raw = self.raw(str(image_path), do_sample=False)
        if k > 1:
            raw += self.raw(str(image_path), do_sample=True, temperature=0.7, n=k - 1)
        return {"queries": [clean_caption(r) for r in raw], "meta": {"raw": raw}}
