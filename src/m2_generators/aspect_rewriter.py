"""Aspect-rewriter M2 wrapper.

Takes an upstream M2 generator's query, runs Qwen2.5-3B (text-only) to extract
structured aspects (living_being, action, object, location, other), then
reconstructs a normalized query. Useful for:
  1. Ablating how much rewriting helps a given upstream generator
  2. Style-normalizing across heterogeneous M2 outputs
"""
from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Union, Any, Dict
from .base import M2Generator
from . import register

_EMPTY_ASPECTS = {
    "living_being": [], "action": [], "object": [], "location": [], "other": [],
}

_SYSTEM_PROMPT = """You are an expert visual scene parser.

Task:
Extract ONLY visually observable concepts from captions.

Return STRICT JSON with EXACTLY these keys:
{"living_being": [], "action": [], "object": [], "location": [], "other": []}

Rules:
- Return ONLY JSON
- No explanations / markdown
- Keep phrases short, lowercase, deduplicated
- Remove abstract / emotional / opinion words

Definitions:
- living_being: humans, animals, roles, occupations
- action      : physically observable actions
- object      : physical objects/things
- location    : places/scenes/environments
- other       : visual attributes (colors, sizes, materials)
"""


def _clean(t: str) -> str:
    t = t.strip().replace("```json", "").replace("```", "")
    t = re.sub(r"^assistant", "", t).strip()
    return t


def parse_aspects(text: str) -> Dict[str, list]:
    """Extract a normalized aspects dict from raw LLM output."""
    text = _clean(text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m is None:
        return dict(_EMPTY_ASPECTS)
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return dict(_EMPTY_ASPECTS)
    final = {}
    for k in _EMPTY_ASPECTS:
        vals = data.get(k, [])
        if not isinstance(vals, list):
            vals = []
        vals = [str(v).strip().lower() for v in vals if str(v).strip()]
        final[k] = list(dict.fromkeys(vals))  # dedup, preserve order
    return final


def aspects_to_query(aspects: Dict[str, list]) -> str:
    """Reconstruct a natural sentence from aspect dict."""
    parts = []
    living    = aspects.get("living_being", [])
    actions   = aspects.get("action", [])
    objects   = aspects.get("object", [])
    locations = aspects.get("location", [])
    others    = aspects.get("other", [])
    if living:
        parts.append(" and ".join(living))
    if actions:
        parts.append(" ".join(actions))
    if objects:
        parts.append(" ".join(objects))
    if locations:
        parts.append("in " + " and ".join(locations))
    if others:
        parts.append("with " + ", ".join(others))
    sentence = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return sentence


@register("aspect_rewriter")
class AspectRewriter(M2Generator):
    """Wraps another M2 generator and rewrites its output through aspect extraction."""

    def __init__(
        self,
        upstream: Union[str, M2Generator],
        upstream_kwargs: Dict[str, Any] | None = None,
        llm_id: str = "Qwen/Qwen2.5-3B-Instruct",
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from . import build_m2

        if isinstance(upstream, str):
            self.upstream = build_m2(upstream, **(upstream_kwargs or {}))
        else:
            self.upstream = upstream

        self.tok = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.llm.eval()
        self._llm_id = llm_id

    @property
    def name(self) -> str:
        return f"aspect_rewriter:{self.upstream.name}"

    @property
    def model_version(self) -> str:
        return f"{self.upstream.model_version}+{self._llm_id}"

    def _extract_aspects(self, query: str) -> Dict[str, list]:
        import torch
        user = f'Caption:\n"{query}"\n\nReturn ONLY JSON.'
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            out = self.llm.generate(
                **inputs, max_new_tokens=180, do_sample=False,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.eos_token_id,
            )
        decoded = self.tok.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return parse_aspects(decoded)

    def generate(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        **kwargs,
    ) -> List[str]:
        raw_queries = self.upstream.generate(thumbnail, k=k, **kwargs)
        rewritten = []
        for q in raw_queries:
            aspects = self._extract_aspects(q)
            rewritten.append(aspects_to_query(aspects) or q)
        return rewritten

    def generate_with_metadata(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        import time
        t0 = time.time()
        upstream_meta = self.upstream.generate_with_metadata(thumbnail, k=k, **kwargs)
        raw_queries = upstream_meta["queries"]
        aspects_list, rewritten = [], []
        for q in raw_queries:
            a = self._extract_aspects(q)
            aspects_list.append(a)
            rewritten.append(aspects_to_query(a) or q)
        return {
            "queries": rewritten,
            "m2_name": self.name,
            "elapsed_ms": (time.time() - t0) * 1000.0,
            "metadata": {
                "upstream_raw_queries": raw_queries,
                "extracted_aspects": aspects_list,
                "upstream_metadata": upstream_meta.get("metadata", {}),
            },
        }
