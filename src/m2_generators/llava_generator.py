"""LLaVA-1.6 (Mistral 7B) M2 generator — aspect+query variant.

Same prompt/parser as QwenVLAspectGenerator; different visual backbone (CLIP-L)
gives complementary error patterns useful for ensembling.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Union
from .base import M2Generator
from . import register
from .qwen_vl_generator import _ASPECT_PROMPT, QwenVLAspectGenerator


@register("llava16_aspect")
class LLaVAAspectGenerator(M2Generator):
    """LLaVA-1.6 aspect-then-query generator."""

    PROMPT = _ASPECT_PROMPT

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        load_in_4bit: bool = True,
    ):
        import torch
        from transformers import (
            LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig,
        )
        kwargs: Dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.float16}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model.eval()
        self._model_id = model_id

    @property
    def name(self) -> str:
        return "llava16_aspect"

    @property
    def model_version(self) -> str:
        return self._model_id

    def _run(self, thumbnail, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
        import torch
        from PIL import Image
        image = Image.open(thumbnail).convert("RGB")
        conv = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.PROMPT},
            ],
        }]
        prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature,
            )
        return self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def generate(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[str]:
        queries: List[str] = []
        for _ in range(k):
            raw = self._run(thumbnail, max_new_tokens, do_sample=(k > 1), temperature=temperature)
            parsed = QwenVLAspectGenerator._parse(raw)
            queries.append(parsed["query"] or raw[:200])
        return queries

    def generate_with_metadata(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        import time
        t0 = time.time()
        parsed_list = []
        for _ in range(k):
            raw = self._run(
                thumbnail,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                do_sample=(k > 1),
                temperature=kwargs.get("temperature", 1.0),
            )
            parsed_list.append(QwenVLAspectGenerator._parse(raw))
        queries = [p["query"] or p["raw"][:200] for p in parsed_list]
        return {
            "queries": queries,
            "m2_name": self.name,
            "elapsed_ms": (time.time() - t0) * 1000.0,
            "metadata": {
                "aspects": [p["aspects"] for p in parsed_list],
                "raw_outputs": [p["raw"] for p in parsed_list],
            },
        }
