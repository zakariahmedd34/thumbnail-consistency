"""Qwen2-VL M2 generators (simple-query and aspect+query variants).

Both share the same Qwen2-VL 7B model loaded in 4-bit. The two classes
differ in their prompt and parsing.
"""
from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from .base import M2Generator
from . import register

_SIMPLE_PROMPT = (
    "Describe what is happening in this image in one short sentence. "
    "Start with a subject (a person, people, an object). "
    "Only describe what is visually present. "
    "Do not ask a question. Do not add opinions or assumptions."
)

_ASPECT_PROMPT = """Look at this image carefully.

Step 1 - Extract ONLY what is visually present:
- living_being : people, animals, or characters visible
- action       : what they are doing
- object       : notable objects or items
- location     : setting or environment
- other        : any other relevant visible detail

Step 2 - Write ONE concise query sentence from those aspects only.
Do NOT add anything not visible in the image.

Respond ONLY in this exact format:
<aspects>
{"living_being": [...], "action": [...], "object": [...], "location": [...], "other": [...]}
</aspects>
<query>
YOUR QUERY HERE
</query>"""


class _QwenVLBase(M2Generator):
    """Shared loader for Qwen2-VL. Subclasses override PROMPT + parser."""

    PROMPT: str = ""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        load_in_4bit: bool = True,
    ):
        import torch
        from transformers import (
            Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
        )

        kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self._model_id = model_id

    @property
    def model_version(self) -> str:
        return self._model_id

    def _run(self, thumbnail, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        img = Image.open(thumbnail).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": self.PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature,
            )
        trimmed = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


@register("qwen2vl_simple")
class QwenVLSimpleGenerator(_QwenVLBase):
    """Single-sentence query generator. Deterministic by default."""

    PROMPT = _SIMPLE_PROMPT

    @property
    def name(self) -> str:
        return "qwen2vl_simple"

    def generate(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        max_new_tokens: int = 60,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[str]:
        # k=1 -> deterministic. k>1 -> nucleus sample with temperature.
        do_sample = k > 1
        queries: List[str] = []
        for _ in range(k):
            q = self._run(thumbnail, max_new_tokens, do_sample, temperature)
            queries.append(q)
        return queries


@register("qwen2vl_aspect")
class QwenVLAspectGenerator(_QwenVLBase):
    """Aspect-then-query generator. Exposes structured aspects in metadata."""

    PROMPT = _ASPECT_PROMPT

    @property
    def name(self) -> str:
        return "qwen2vl_aspect"

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        aspects, query = None, None
        am = re.search(r"<aspects>\s*(\{.*?\})\s*</aspects>", raw, re.DOTALL)
        if am:
            try:
                aspects = json.loads(am.group(1))
            except json.JSONDecodeError:
                aspects = {"parse_error": am.group(1)[:200]}
        qm = re.search(r"<query>\s*(.+?)\s*</query>", raw, re.DOTALL)
        if qm:
            query = qm.group(1).strip()
        return {"aspects": aspects, "query": query, "raw": raw}

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
            parsed = self._parse(raw)
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
            parsed_list.append(self._parse(raw))
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
