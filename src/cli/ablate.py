"""CLI: run multiple M2 generators against the same eval set and tabulate AUC.

Usage
-----
python -m src.cli.ablate \
    --eval-jsonl data/processed/consistency_eval_hard.jsonl \
    --m2-list clipcap qwen2vl_simple llava16_aspect gt random \
    --features-dir /content/features_local/features/clip_features \
    --m1-ckpt /content/.../model_best.ckpt \
    --output-dir outputs/predictions \
    --table outputs/figures/ablation_table.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-jsonl",  required=True)
    ap.add_argument("--m2-list",     required=True, nargs="+")
    ap.add_argument("--features-dir",required=True)
    ap.add_argument("--m1-ckpt",     required=True)
    ap.add_argument("--repo-root",   default=".")
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--table",       required=True, help="Output CSV with one row per M2")
    ap.add_argument("--alpha",       type=float, default=0.5)
    ap.add_argument("--tau",         type=float, default=0.1)
    ap.add_argument("--top-p",       type=float, default=0.25)
    ap.add_argument("--k",           type=int,   default=3)
    args = ap.parse_args()

    from tqdm.auto import tqdm
    from src.m2_generators import build_m2
    from src.models.moment_detr import MomentDETRFeaturePredictor
    from src.scoring.consistency import ConsistencyScorer, ScoringConfig
    from src.eval.metrics import summarize_separation

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.table).parent.mkdir(parents=True, exist_ok=True)

    # Load eval pairs once
    pairs = [json.loads(l) for l in open(args.eval_jsonl, encoding="utf-8") if l.strip()]
    print(f"Eval pairs: {len(pairs)}")

    m1 = MomentDETRFeaturePredictor(ckpt_path=args.m1_ckpt, repo_root=args.repo_root)
    cfg = ScoringConfig(alpha=args.alpha, tau=args.tau, top_p=args.top_p, k_queries=args.k)

    rows = []
    for m2_name in args.m2_list:
        print(f"\n=== {m2_name} ===")
        # GT and random need a lookup/pool argument; pass the eval file itself.
        kwargs = {}
        if m2_name == "gt":
            kwargs = {"lookup_jsonl": args.eval_jsonl}
        elif m2_name == "random":
            kwargs = {"pool_jsonl": args.eval_jsonl}
        m2 = build_m2(m2_name, **kwargs)

        scorer = ConsistencyScorer(m1=m1, m2=m2, feature_dir=args.features_dir, cfg=cfg)  # type: ignore[arg-type]
        results = []
        for r in tqdm(pairs, desc=m2_name):
            extra = {"qid": r.get("thumb_qid")} if m2_name == "gt" else {}
            try:
                out = scorer.score(
                    thumbnail_path=r["frame_path"],
                    vid=r["video_vid"],
                    **extra,
                )
                results.append({"pair_id": r["pair_id"], "label": r["label"],
                                "score": out["consistency_score"]})
            except Exception as e:
                print(f"  FAIL pair {r['pair_id']}: {e}")
                continue

        # Save per-pair scores
        per_path = out_dir / f"scores_{m2_name}.jsonl"
        with open(per_path, "w", encoding="utf-8") as f:
            for x in results:
                f.write(json.dumps(x) + "\n")

        # Compute summary
        s = [x["score"] for x in results]
        l = [x["label"] for x in results]
        stats = summarize_separation(s, l)
        stats["m2"] = m2_name
        stats["n_scored"] = len(results)
        rows.append(stats)
        print(f"  AUC={stats['auc']:.3f}  d={stats['cohens_d']:.2f}  n={len(results)}")

    # Save table
    import csv
    with open(args.table, "w", newline="", encoding="utf-8") as f:
        keys = ["m2", "n_scored", "n_pos", "n_neg",
                "mean_pos", "mean_neg", "auc", "cohens_d", "welch_t", "welch_p"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nWrote {args.table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
