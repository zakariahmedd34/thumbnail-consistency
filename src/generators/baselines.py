"""Reference generators that need no model: ground truth (ceiling) and random (control)."""
from __future__ import annotations

import random
from typing import Optional

from . import Generator, register


def _queries_by_qid(jsonl_path=None) -> dict:
    """qid -> human query, from one eval set (name or path) or, if None, from all eval sets."""
    from src.config import cfg, eval_set, load_jsonl
    names = [jsonl_path] if jsonl_path else list(cfg()["data"]["eval_sets"])
    rows = [r for n in names for r in load_jsonl(eval_set(str(n)))]
    out = {}
    for r in rows:
        if "wrong_query_from" in r:  # query_mismatch negatives carry a deliberately wrong query
            continue
        qid = r.get("qid", r.get("thumb_qid"))
        q = r.get("query") or r.get("thumb_query")
        if qid is not None and q:
            out[int(qid)] = q
    if not out:
        raise ValueError(f"no qid->query rows in {jsonl_path}")
    return out


@register("gt")
class GroundTruth(Generator):
    """The human-written QVHighlights query for the thumbnail's qid."""
    gpu_required = False

    def __init__(self, lookup_jsonl: str | None = None):
        self._by_qid = _queries_by_qid(lookup_jsonl)

    def generate(self, image_path, k: int = 1, qid: Optional[int] = None, **_):
        if qid is None:
            raise ValueError("generator 'gt' requires `qid`")
        if int(qid) not in self._by_qid:
            raise KeyError(f"qid {qid} not in lookup table")
        return {"queries": [self._by_qid[int(qid)]] * k, "meta": {"qid": int(qid)}}


@register("random")
class RandomQuery(Generator):
    """A random query from the pool, ignoring the image."""
    gpu_required = False

    def __init__(self, pool_jsonl: str | None = "easy", seed: int = 2026):
        self._pool = list(_queries_by_qid(pool_jsonl).values())
        self._rng = random.Random(seed)

    def generate(self, image_path, k: int = 1, **_):
        return {"queries": [self._rng.choice(self._pool) for _ in range(k)], "meta": {}}
