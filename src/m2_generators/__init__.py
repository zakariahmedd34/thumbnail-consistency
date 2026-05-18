"""M2 generator registry and factory.

Usage
-----
>>> from src.m2_generators import build_m2, available
>>> available()
['aspect_rewriter', 'clipcap', 'gt', 'llava16_aspect', 'qwen2vl_aspect',
 'qwen2vl_simple', 'random']
>>> m2 = build_m2('qwen2vl_simple')
>>> queries = m2.generate('thumb.jpg', k=3)
"""
from __future__ import annotations
from typing import Dict, Type, List, Any
from .base import M2Generator

_REGISTRY: Dict[str, Type[M2Generator]] = {}


def register(name: str):
    """Decorator: register a generator class under `name`.

    Example
    -------
    @register("qwen2vl_simple")
    class QwenVLSimpleGenerator(M2Generator):
        ...
    """
    def wrap(cls: Type[M2Generator]) -> Type[M2Generator]:
        if name in _REGISTRY:
            raise ValueError(f"M2 generator '{name}' is already registered.")
        if not issubclass(cls, M2Generator):
            raise TypeError(f"{cls.__name__} must subclass M2Generator.")
        _REGISTRY[name] = cls
        return cls
    return wrap


def build_m2(name: str, **kwargs: Any) -> M2Generator:
    """Instantiate the registered generator by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown M2 generator: {name!r}. "
            f"Available: {available()}"
        )
    return _REGISTRY[name](**kwargs)


def available() -> List[str]:
    """Sorted list of registered M2 generator names."""
    return sorted(_REGISTRY.keys())


# --- Eagerly import concrete implementations so the registry is populated ---
# Each module uses @register to add itself.
# Failures here mean the corresponding generator's dependencies aren't installed.
# We catch ImportError so the registry stays usable for the generators that
# DO have their deps available (e.g. ClipCap without transformers VLMs).

def _safe_import(modname: str) -> None:
    try:
        __import__(f"src.m2_generators.{modname}", fromlist=["_"])
    except ImportError:
        pass

for _m in [
    "clipcap_generator",
    "qwen_vl_generator",
    "llava_generator",
    "aspect_rewriter",
    "ground_truth_generator",
    "random_generator",
]:
    _safe_import(_m)

del _safe_import, _m


__all__ = ["M2Generator", "register", "build_m2", "available"]
