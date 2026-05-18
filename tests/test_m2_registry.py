"""Contract tests for src.m2_generators.

Verifies that:
  - The registry is populated with all expected generator names
  - Every registered class is a subclass of M2Generator
  - Lightweight generators (gt, random) round-trip through generate()

VLM-based generators are NOT instantiated here because they require GPU + heavy
model downloads. Their contract is verified by importing the module without
constructing an instance.
"""
from __future__ import annotations
import json, tempfile, os
from pathlib import Path
import pytest

from src.m2_generators import available, build_m2
from src.m2_generators.base import M2Generator


# Expected registry membership (these classes register on import).
EXPECTED = {"gt", "random", "clipcap", "qwen2vl_simple", "qwen2vl_aspect",
            "llava16_aspect", "aspect_rewriter"}


def test_registry_has_expected_names():
    """Every documented generator should appear in available()."""
    have = set(available())
    missing = EXPECTED - have
    assert not missing, f"Missing M2 generators: {missing}. Have: {sorted(have)}"


def test_every_registered_is_subclass():
    """Every registered name must build into an M2Generator subclass (when deps available)."""
    from src.m2_generators import _REGISTRY
    for name, cls in _REGISTRY.items():
        assert issubclass(cls, M2Generator), f"{name} is not an M2Generator subclass"


@pytest.fixture
def tiny_eval_jsonl(tmp_path):
    """Tiny eval-jsonl shaped like consistency_eval.jsonl for GT/random tests."""
    p = tmp_path / "eval.jsonl"
    records = [
        {"qid": 1, "thumb_qid": 1, "query": "a man walks in a park"},
        {"qid": 2, "thumb_qid": 2, "query": "a dog plays with a ball"},
        {"qid": 3, "thumb_qid": 3, "query": "a child laughs on a swing"},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_gt_generator_roundtrip(tiny_eval_jsonl):
    m2 = build_m2("gt", lookup_jsonl=tiny_eval_jsonl)
    assert m2.name == "gt"
    assert m2.gpu_required is False

    out = m2.generate("any/path/to/thumb.jpg", k=2, qid=1)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0] == "a man walks in a park"
    assert out[1] == out[0]


def test_gt_generator_missing_qid_raises(tiny_eval_jsonl):
    m2 = build_m2("gt", lookup_jsonl=tiny_eval_jsonl)
    with pytest.raises(ValueError, match="requires `qid`"):
        m2.generate("any.jpg", k=1)


def test_random_generator_roundtrip(tiny_eval_jsonl):
    m2 = build_m2("random", pool_jsonl=tiny_eval_jsonl, seed=42)
    assert m2.name == "random"
    assert m2.gpu_required is False

    out = m2.generate("thumb.jpg", k=5)
    assert isinstance(out, list)
    assert len(out) == 5
    # All queries should come from the pool
    pool = {"a man walks in a park", "a dog plays with a ball",
            "a child laughs on a swing"}
    assert set(out).issubset(pool)


def test_random_generator_deterministic_with_seed(tiny_eval_jsonl):
    m2a = build_m2("random", pool_jsonl=tiny_eval_jsonl, seed=42)
    m2b = build_m2("random", pool_jsonl=tiny_eval_jsonl, seed=42)
    a = m2a.generate("x.jpg", k=10)
    b = m2b.generate("x.jpg", k=10)
    assert a == b, "Same seed must produce identical outputs"


def test_generate_with_metadata_schema(tiny_eval_jsonl):
    """The default metadata wrapper must return the canonical keys."""
    m2 = build_m2("gt", lookup_jsonl=tiny_eval_jsonl)
    meta = m2.generate_with_metadata("any.jpg", k=1, qid=1)
    for key in ("queries", "m2_name", "elapsed_ms", "metadata"):
        assert key in meta, f"Missing key in metadata: {key}"
    assert isinstance(meta["queries"], list)
    assert isinstance(meta["m2_name"], str)
    assert isinstance(meta["elapsed_ms"], float)
    assert isinstance(meta["metadata"], dict)


def test_build_unknown_raises():
    with pytest.raises(ValueError, match="Unknown M2 generator"):
        build_m2("does_not_exist")
