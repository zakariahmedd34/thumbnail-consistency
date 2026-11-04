"""Fast tests (no models, no GPU): generator registry, baselines, config paths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config
from src.generators import available, build


@pytest.fixture
def tiny_jsonl(tmp_path):
    p = tmp_path / "eval.jsonl"
    rows = [{"thumb_qid": 1, "thumb_query": "a man walks in a park"},
            {"thumb_qid": 2, "thumb_query": "a dog plays with a ball"}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_registry_lists_all_generators():
    assert set(available()) == {"gt", "random", "clipcap", "vlm"}


def test_unknown_generator_raises():
    with pytest.raises(ValueError, match="unknown generator"):
        build("nope")


def test_gt_returns_query_for_qid(tiny_jsonl):
    out = build("gt", lookup_jsonl=tiny_jsonl).generate("x.jpg", k=2, qid=1)
    assert out["queries"] == ["a man walks in a park"] * 2


def test_gt_requires_qid(tiny_jsonl):
    with pytest.raises(ValueError, match="qid"):
        build("gt", lookup_jsonl=tiny_jsonl).generate("x.jpg")


def test_random_is_seeded(tiny_jsonl):
    a = build("random", pool_jsonl=tiny_jsonl, seed=1).generate("x.jpg", k=5)["queries"]
    b = build("random", pool_jsonl=tiny_jsonl, seed=1).generate("x.jpg", k=5)["queries"]
    assert a == b and len(a) == 5


def test_legacy_colab_paths_map_to_local_data_dir():
    p = config.data_path("/content/drive/MyDrive/thumbnail_project/qvh_frames/abc.jpg")
    assert p == config.data_dir() / "qvh_frames" / "abc.jpg"


def test_relative_data_paths_resolve_under_data_dir():
    assert config.data_path("processed/x.jsonl") == config.data_dir() / "processed" / "x.jsonl"


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("THUMB_DATA_DIR", str(tmp_path))
    assert config.data_path("/content/drive/MyDrive/thumbnail_project/a.jpg") == tmp_path / "a.jpg"


def test_config_has_required_sections():
    c = config.cfg()
    for key in ["data", "moment_detr", "scoring", "generators", "app"]:
        assert key in c
    assert 0.0 <= c["scoring"]["alpha"] <= 1.0
