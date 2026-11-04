"""End-to-end tests with the real CLIP + Moment-DETR models.

Skipped automatically when the checkpoint or features are missing (e.g. in CI).
Run just these with:  python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from src import config

pytestmark = pytest.mark.skipif(
    not (config.m1_checkpoint().exists() and config.features_dir().is_dir()
         and any(config.features_dir().glob("*.npz"))),
    reason="Moment-DETR checkpoint or video features not available")


@pytest.fixture(scope="module")
def pairs():
    rows = config.load_jsonl(config.eval_set("hard"))
    pos = next(r for r in rows if r["label"] == 1)
    neg = next(r for r in rows if r["label"] == 0 and r["video_vid"] == pos["video_vid"]) \
        if any(r["label"] == 0 and r["video_vid"] == pos["video_vid"] for r in rows) \
        else next(r for r in rows if r["label"] == 0)
    return pos, neg


@pytest.fixture(scope="module")
def scorer():
    from src.scoring import ConsistencyScorer
    return ConsistencyScorer()


def test_text_tokens_match_official_features():
    """Moment-DETR's input text features must match the official extraction."""
    from src.encoders import encode_text_tokens
    tdir = config.features_dir().parent / "clip_text_features"
    row = config.load_jsonl(config.eval_set("easy"))[0]
    path = tdir / f"qid{row['thumb_qid']}.npz"
    if not path.exists():
        pytest.skip("official text features not extracted")
    official = np.load(path)["last_hidden_state"]
    ours = encode_text_tokens([row["thumb_query"]])[0]
    n = min(len(ours), len(official))
    cos = (ours[:n] * official[:n]).sum(1) / (np.linalg.norm(ours[:n], axis=1) * np.linalg.norm(official[:n], axis=1))
    assert cos.min() > 0.99


def test_score_output_shape(scorer, pairs):
    pos, _ = pairs
    out = scorer.score(str(config.data_path(pos["frame_path"])), pos["video_vid"], [pos["thumb_query"], "a dog"])
    assert np.isfinite(out["C"]) and -1 <= out["C"] <= 1
    assert len(out["per_query"]) == 2
    assert out["n_chunks"] == 1


def test_correct_query_beats_wrong_query(scorer, pairs):
    pos, _ = pairs
    img = str(config.data_path(pos["frame_path"]))
    right = scorer.score(img, pos["video_vid"], [pos["thumb_query"]])["C"]
    wrong = scorer.score(img, pos["video_vid"], ["a plate of sushi on a wooden table"])["C"]
    assert right > wrong


def test_flask_score_endpoint(pairs):
    from app.server import create_app
    pos, _ = pairs
    client = create_app().test_client()
    assert client.get("/health").get_json()["assets"]["moment_detr_checkpoint"] is True
    r = client.post("/v1/score", json={"thumbnail_path": pos["frame_path"], "vid": pos["video_vid"],
                                       "queries": [pos["thumb_query"]]})
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["score"] is not None and body["score"] > 0.3
    assert body["verdict"] in ("consistent", "inconsistent")
    # multipart upload path
    with open(config.data_path(pos["frame_path"]), "rb") as f:
        data = {"thumbnail": (io.BytesIO(f.read()), "t.jpg"), "vid": pos["video_vid"], "gen": "gt",
                "qid": str(pos["thumb_qid"]), "k": "1"}
    r = client.post("/v1/score", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["queries"] == [pos["thumb_query"]]
