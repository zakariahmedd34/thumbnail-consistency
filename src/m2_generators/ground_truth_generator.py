"""Ground-truth M2 generator (ceiling baseline).

Looks up the QVHighlights human-authored query for each qid.
This is the upper bound: every other generator is trying to match this.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Union, Optional
from .base import M2Generator
from . import register


@register("gt")
class GTQueryGenerator(M2Generator):
    """Returns the ground-truth QVHighlights query.

    Requires the caller to pass `qid=<int>` in generate(). Use this as the
    ceiling baseline in ablations.
    """

    def __init__(self, lookup_jsonl: Union[str, Path]):
        self._by_qid: dict = {}
        with open(lookup_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r.get("qid") or r.get("thumb_qid")
                q   = r.get("query") or r.get("thumb_query")
                if qid is not None and q:
                    self._by_qid[int(qid)] = q
        if not self._by_qid:
            raise ValueError(f"No qid->query mappings found in {lookup_jsonl}")

    @property
    def name(self) -> str:
        return "gt"

    @property
    def gpu_required(self) -> bool:
        return False

    def generate(
        self,
        thumbnail,
        k: int = 1,
        qid: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        if qid is None:
            raise ValueError(
                "GTQueryGenerator requires `qid` kwarg "
                "(the QVHighlights query id of the thumbnail)."
            )
        if int(qid) not in self._by_qid:
            raise KeyError(f"qid {qid} not in lookup table.")
        q = self._by_qid[int(qid)]
        return [q] * k
