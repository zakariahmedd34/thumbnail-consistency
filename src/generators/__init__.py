"""M2 generators: thumbnail image -> K text queries describing it.

    from src.generators import build, available
    gen = build("vlm")                 # kwargs default to config.yaml -> generators.vlm
    out = gen.generate("thumb.jpg", k=3)
    out["queries"]                     # list of K strings
    out["meta"]                        # generator-specific diagnostics

Generators
  gt      ground-truth QVHighlights query (needs qid) - upper bound
  random  random query from the eval pool            - control, AUC should be ~0.5
  clipcap ClipCap (trained CLIP-prefix + frozen GPT-2) - needs a trained checkpoint
  vlm     a Hugging Face vision-language model (Qwen2-VL by default)

Heavy dependencies are imported inside each generator's __init__, so listing or
using the light generators never loads torch/transformers.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

_REGISTRY: Dict[str, Callable[..., "Generator"]] = {}


class Generator:
    name: str = "base"
    gpu_required: bool = True

    def generate(self, image_path: str, k: int = 1, **kwargs: Any) -> Dict[str, Any]:
        """Return {"queries": [k strings], "meta": {...}}."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<generator {self.name}>"


def register(name: str):
    def wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"generator {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def available() -> List[str]:
    return sorted(_REGISTRY)


def build(name: str, **kwargs: Any) -> Generator:
    """Instantiate a generator; missing kwargs are filled from config.yaml."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown generator {name!r}; available: {available()}")
    from src.config import generator_kwargs
    return _REGISTRY[name](**{**generator_kwargs(name), **kwargs})


from . import baselines, clipcap, vlm  # noqa: E402,F401  (registers generators)
