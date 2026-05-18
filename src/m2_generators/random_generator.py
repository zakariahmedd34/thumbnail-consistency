"""Random query M2 generator (control baseline).

Samples a random query from a pool. AUC should be ~0.50.
"""
from __future__ import annotations
import json
import random as _random
from pathlib import Path
from typing import List, Union
from .base import M2Generator
from . import register


@register("random")
class RandomQueryGenerator(M2Generator):
    """Picks a random query from a fixed pool. Used as a control baseline."""

    def __init__(
        self,
        pool_jsonl: Union[str, Path],
        seed: int = 2026,
    ):
        self._pool: List[str] = []
        with open(pool_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                q = r.get("query") or r.get("thumb_query")
                if q:
                    self._pool.append(q)
        if not self._pool:
            raise ValueError(f"No queries in {pool_jsonl}")
        self._rng = _random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    @property
    def gpu_required(self) -> bool:
        return False

    def generate(self, thumbnail, k: int = 1, **kwargs) -> List[str]:
        return [self._rng.choice(self._pool) for _ in range(k)]
