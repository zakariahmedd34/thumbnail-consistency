"""Flask app: browser form + JSON API around the scoring pipeline.

GET  /                 form for uploading a thumbnail and choosing a video id
GET  /health           which files/models are available
GET  /v1/generators    registered generators
POST /v1/score         JSON {thumbnail_path, vid, gen, k, qid, queries} or multipart (thumbnail file + fields)

Models load lazily on the first request that needs them and are then reused.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

from src import config

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, default_gen: str, k: int, threshold: float):
        self.default_gen, self.k, self.threshold = default_gen, k, threshold
        self._lock = threading.Lock()
        self._gens: Dict[str, Any] = {}
        self._scorer = None

    def generator(self, name: str):
        with self._lock:
            if name not in self._gens:
                from src.generators import build
                log.info("loading generator %s", name)
                self._gens[name] = build(name)
            return self._gens[name]

    def scorer(self):
        with self._lock:
            if self._scorer is None:
                from src.scoring import ConsistencyScorer
                log.info("loading Moment-DETR + CLIP")
                self._scorer = ConsistencyScorer()
            return self._scorer

    def run(self, thumbnail: str, vid: Optional[str], gen: Optional[str] = None, k: Optional[int] = None,
            qid: Optional[int] = None, queries: Optional[List[str]] = None) -> Dict[str, Any]:
        t0 = time.time()
        gen = gen or self.default_gen
        k = int(k or self.k)
        if queries:
            gen_out = {"queries": queries, "meta": {"source": "provided"}}
        else:
            gen_out = self.generator(gen).generate(thumbnail, k=k, **({"qid": qid} if qid is not None else {}))
        result: Dict[str, Any] = {"generator": None if queries else gen, "queries": gen_out["queries"],
                                  "generator_meta": gen_out["meta"], "score": None, "verdict": None}
        if vid:
            s = self.scorer().score(thumbnail, vid, gen_out["queries"])
            result.update({
                "score": s["C"], "per_query": s["per_query"], "max_fg_prob": s["max_fg_prob"],
                "top_moments": s["top_moments"],
                "verdict": "consistent" if s["C"] >= self.threshold else "inconsistent",
                "threshold": self.threshold,
            })
        result["seconds"] = round(time.time() - t0, 2)
        return result


def create_app() -> Flask:
    acfg = config.cfg().get("app", {})
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    pipe = Pipeline(acfg.get("default_generator", "vlm"), acfg.get("k", 3), acfg.get("threshold", 0.5))
    upload_dir = config.outputs_dir() / "uploads"

    def assets() -> Dict[str, bool]:
        fd = config.features_dir()
        return {"moment_detr_checkpoint": config.m1_checkpoint().exists(),
                "video_features": fd.is_dir() and any(fd.glob("*.npz"))}

    @app.get("/")
    def index():
        from src.generators import available
        return render_template("index.html", gens=available(), pipe=pipe, assets=assets())

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "assets": assets(), "default_generator": pipe.default_gen,
                        "k": pipe.k, "threshold": pipe.threshold})

    @app.get("/v1/generators")
    def generators():
        from src.generators import available
        return jsonify({"available": available()})

    @app.post("/v1/score")
    def score():
        if request.files.get("thumbnail"):
            f = request.files["thumbnail"]
            upload_dir.mkdir(parents=True, exist_ok=True)
            thumb = str(upload_dir / f"{uuid.uuid4().hex}{Path(f.filename or 'x.jpg').suffix.lower() or '.jpg'}")
            f.save(thumb)
            body = {k: v for k, v in request.form.items() if v != ""}
            if body.get("queries"):
                body["queries"] = [q.strip() for q in body["queries"].split("\n") if q.strip()]
        else:
            body = request.get_json(silent=True) or {}
            thumb = body.get("thumbnail_path")
            if not thumb:
                return jsonify({"error": "send a 'thumbnail' file (multipart) or 'thumbnail_path' (JSON)"}), 400
            thumb = str(config.data_path(thumb))
        if not Path(thumb).exists():
            return jsonify({"error": f"thumbnail not found: {thumb}"}), 400
        try:
            out = pipe.run(thumb, body.get("vid"), body.get("gen"), body.get("k"),
                           int(body["qid"]) if body.get("qid") not in (None, "") else None, body.get("queries"))
        except (ValueError, KeyError) as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": f"missing file: {e}"}), 404
        except Exception as e:  # noqa: BLE001 - report anything else as a 500 with the reason
            log.exception("scoring failed")
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        return jsonify(out)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "upload too large (max 20 MB)"}), 413

    return app
