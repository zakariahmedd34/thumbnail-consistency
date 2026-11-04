"""Command-line entry point.

    python -m src.cli check                               # what is installed / missing
    python -m src.cli generate --image IMG --gen vlm -k 3  # M2 only: thumbnail -> queries
    python -m src.cli score --image IMG --vid VID --gen gt --qid 123
    python -m src.cli score --image IMG --vid VID --queries "a man cooks pasta"
    python -m src.cli evaluate --set easy --gen gt random --limit 100
    python -m src.cli train-clipcap --epochs 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def _version(mod: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version({"yaml": "pyyaml", "PIL": "pillow"}.get(mod, mod))
    except PackageNotFoundError:
        return ""


def cmd_check(_args) -> int:
    from src import config
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok' if good else 'MISSING'}] {name:28s} {detail}")

    print("Python packages")
    for mod in ["numpy", "yaml", "PIL", "torch", "transformers", "scipy", "flask"]:
        try:
            m = __import__(mod)
            line(mod, True, _version(mod))
        except ImportError:
            line(mod, False, "pip install -r requirements.txt")
    try:
        import bitsandbytes  # noqa: F401
        print("  [ok] bitsandbytes                 (4-bit VLM loading available)")
    except ImportError:
        print("  [--] bitsandbytes                 optional; needed for load_in_4bit VLMs")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) + f" ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)" \
            if torch.cuda.is_available() else "none (CPU only - VLM generators will be very slow)"
        print(f"  [--] GPU                          {gpu}")
    except ImportError:
        pass

    print("Files")
    md = config.moment_detr_repo()
    line("Moment-DETR code", (md / "moment_detr" / "model.py").exists(), str(md))
    line("Moment-DETR checkpoint", config.m1_checkpoint().exists(), str(config.m1_checkpoint()))
    fd = config.features_dir()
    n = len(list(fd.glob("*.npz"))) if fd.is_dir() else 0
    line("video features", n > 0, f"{n} files in {fd}")
    for name in config.cfg()["data"]["eval_sets"]:
        p = config.eval_set(name)
        if not p.exists():
            line(f"eval set '{name}'", False, str(p)); continue
        rows = config.load_jsonl(p)
        frames = sum(config.data_path(r["frame_path"]).exists() for r in rows)
        feats = sum((fd / f"{r['video_vid']}.npz").exists() for r in rows)
        line(f"eval set '{name}'", frames == len(rows) and feats == len(rows),
             f"{len(rows)} pairs, {frames} images, {feats} video features")
    ck = Path(config.generator_kwargs("clipcap")["ckpt_path"])
    print(f"  [{'ok' if ck.exists() else '--'}] ClipCap checkpoint           "
          f"{ck if ck.exists() else 'optional; train with: python -m src.cli train-clipcap'}")
    print("\nAll required pieces present." if ok else "\nSome required pieces are missing (see above).")
    return 0 if ok else 1


def _build_generator(name: str, eval_path: str | None = None):
    from src.generators import build
    kw = {}
    if name == "gt" and eval_path:
        kw["lookup_jsonl"] = eval_path
    if name == "random" and eval_path:
        kw["pool_jsonl"] = eval_path
    return build(name, **kw)


def cmd_generate(args) -> int:
    gen = _build_generator(args.gen)
    t0 = time.time()
    out = gen.generate(args.image, k=args.k, **({"qid": args.qid} if args.qid is not None else {}))
    out["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_score(args) -> int:
    from src.scoring import ConsistencyScorer, ScoringConfig
    if args.queries:
        queries, meta = args.queries, {}
    else:
        out = _build_generator(args.gen).generate(args.image, k=args.k,
                                                  **({"qid": args.qid} if args.qid is not None else {}))
        queries, meta = out["queries"], out["meta"]
    scorer = ConsistencyScorer(cfg=ScoringConfig.from_config(alpha=args.alpha))
    res = scorer.score(args.image, args.vid, queries)
    res.pop("per_query_chunk", None)
    res["generator_meta"] = meta
    text = json.dumps(res, indent=2, ensure_ascii=False, default=str)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8"); print(f"wrote {args.output}")
    print(text)
    return 0


def cmd_evaluate(args) -> int:
    from src import config
    from src.metrics import summarize_separation
    from src.scoring import ConsistencyScorer, ScoringConfig

    eval_path = config.eval_set(args.set)
    pairs = config.load_jsonl(eval_path)
    if args.limit:
        # keep the label balance when limiting
        pos = [p for p in pairs if p["label"] == 1][: args.limit // 2]
        neg = [p for p in pairs if p["label"] == 0][: args.limit - len(pos)]
        pairs = pos + neg
    out_dir = config.outputs_dir() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = ConsistencyScorer(cfg=ScoringConfig.from_config(alpha=args.alpha))
    print(f"eval set '{args.set}': {len(pairs)} pairs  ({eval_path})")

    rows = []
    for gname in args.gen:
        # Generated queries are cached per (generator, eval set) so reruns only re-score.
        qcache = out_dir / f"queries_{gname}_{Path(str(args.set)).stem}.jsonl"
        cached = {r["pair_id"]: r for r in config.load_jsonl(qcache)} \
            if qcache.exists() and gname not in ("random", "gt") else {}
        gen = None
        results, fails, t0 = [], 0, time.time()
        for i, p in enumerate(pairs):
            img = str(config.data_path(p["frame_path"]))
            try:
                if gname == "gt":
                    # The row's own human query. (A qid lookup is wrong for the query_mismatch
                    # set, where a positive and its negative share a qid but differ in query.)
                    queries = [p["thumb_query"]] * args.k
                elif p["pair_id"] in cached and len(cached[p["pair_id"]]["queries"]) >= args.k:
                    queries = cached[p["pair_id"]]["queries"][: args.k]
                else:
                    gen = gen or _build_generator(gname, str(eval_path))
                    queries = gen.generate(img, k=args.k, qid=p.get("thumb_qid"))["queries"]
                    cached[p["pair_id"]] = {"pair_id": p["pair_id"], "queries": queries}
                s = scorer.score(img, p["video_vid"], queries)
                results.append({"pair_id": p["pair_id"], "label": p["label"], "score": s["C"], "queries": queries})
            except Exception as e:  # keep going; report failures at the end
                fails += 1
                if fails <= 5:
                    print(f"  FAIL {p['pair_id']}: {type(e).__name__}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  {gname}: {i + 1}/{len(pairs)}  ({time.time() - t0:.0f}s)")
        if gname not in ("random", "gt"):
            config.save_jsonl(cached.values(), qcache)
        config.save_jsonl(results, out_dir / f"scores_{gname}_{Path(str(args.set)).stem}.jsonl")
        st = summarize_separation([r["score"] for r in results], [r["label"] for r in results])
        st.update({"generator": gname, "set": args.set, "n_failed": fails, "seconds": round(time.time() - t0, 1)})
        rows.append(st)
        print(f"  {gname:8s} AUC={st['auc']:.3f}  d={st['cohens_d']:.2f}  "
              f"mean+={st['mean_pos']:.3f} mean-={st['mean_neg']:.3f}  n={len(results)} failed={fails}")

    table = out_dir / f"summary_{Path(str(args.set)).stem}.csv"
    keys = ["generator", "set", "n_pos", "n_neg", "n_failed", "auc", "cohens_d", "mean_pos", "mean_neg",
            "welch_t", "welch_p", "seconds"]
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"wrote {table}")
    return 0 if all(r["n_failed"] == 0 for r in rows) else 1


def cmd_train_clipcap(args) -> int:
    from src.generators.clipcap import train
    train(epochs=args.epochs, max_train=args.max_train)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify dependencies, model files and data")

    g = sub.add_parser("generate", help="thumbnail -> queries (M2 only)")
    g.add_argument("--image", required=True); g.add_argument("--gen", default="vlm")
    g.add_argument("-k", type=int, default=3); g.add_argument("--qid", type=int)

    s = sub.add_parser("score", help="score one (thumbnail, video) pair")
    s.add_argument("--image", required=True); s.add_argument("--vid", required=True, help="QVH video id, e.g. ABC_60.0_210.0")
    s.add_argument("--gen", default="vlm"); s.add_argument("--queries", nargs="+", help="skip M2 and use these queries")
    s.add_argument("-k", type=int, default=3); s.add_argument("--qid", type=int); s.add_argument("--alpha", type=float)
    s.add_argument("--output")

    e = sub.add_parser("evaluate", help="AUC of generators on an eval set")
    e.add_argument("--set", default="easy", help="easy | hard | query_mismatch | path to a JSONL")
    e.add_argument("--gen", nargs="+", default=["gt", "random"])
    e.add_argument("-k", type=int, default=1); e.add_argument("--limit", type=int)
    e.add_argument("--alpha", type=float)

    t = sub.add_parser("train-clipcap", help="train the ClipCap prefix MLP")
    t.add_argument("--epochs", type=int); t.add_argument("--max-train", type=int)

    args = ap.parse_args(argv)
    return {"check": cmd_check, "generate": cmd_generate, "score": cmd_score,
            "evaluate": cmd_evaluate, "train-clipcap": cmd_train_clipcap}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
