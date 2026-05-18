"""CLI: score one (thumbnail, video) pair with a chosen M2 generator.

Usage
-----
python -m src.cli.score \
    --thumbnail /path/to/thumb.jpg \
    --video-vid Exr8FNpynKo_210.0_360.0 \
    --m2 qwen2vl_simple \
    --features-dir /content/features_local/features/clip_features \
    --m1-ckpt /content/drive/.../model_best.ckpt \
    --output result.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thumbnail",   required=True, help="Path to thumbnail image")
    ap.add_argument("--video-vid",   required=True, help="QVH vid string, e.g. ABC_60.0_210.0")
    ap.add_argument("--m2",          required=True, help="M2 generator name (see src.m2_generators.available())")
    ap.add_argument("--m2-config",   default=None,  help="YAML with m2_kwargs (optional)")
    ap.add_argument("--features-dir",required=True, help="Path to clip_features/*.npz directory")
    ap.add_argument("--m1-ckpt",     required=True, help="Path to Moment-DETR model_best.ckpt")
    ap.add_argument("--repo-root",   default=".",   help="Project root (for moment_detr imports)")
    ap.add_argument("--k",           type=int, default=3, help="Number of queries to generate")
    ap.add_argument("--alpha",       type=float, default=0.5)
    ap.add_argument("--tau",         type=float, default=0.1)
    ap.add_argument("--top-p",       type=float, default=0.25)
    ap.add_argument("--qid",         type=int, default=None, help="Required for --m2 gt")
    ap.add_argument("--output",      default=None, help="Write result JSON here (default: stdout)")
    args = ap.parse_args()

    # Late imports (avoid loading models if --help is the only thing called)
    from src.m2_generators import build_m2
    from src.models.moment_detr import MomentDETRFeaturePredictor
    from src.scoring.consistency import ConsistencyScorer, ScoringConfig

    m2_kwargs = {}
    if args.m2_config:
        import yaml
        with open(args.m2_config, encoding="utf-8") as f:
            m2_kwargs = (yaml.safe_load(f) or {}).get("m2_kwargs", {})

    m2 = build_m2(args.m2, **m2_kwargs)
    m1 = MomentDETRFeaturePredictor(ckpt_path=args.m1_ckpt, repo_root=args.repo_root)
    cfg = ScoringConfig(alpha=args.alpha, tau=args.tau, top_p=args.top_p)
    scorer = ConsistencyScorer(m1=m1, m2=m2, feature_dir=args.features_dir, cfg=cfg)  # type: ignore[arg-type]

    extra = {}
    if args.qid is not None:
        extra["qid"] = args.qid
    result = scorer.score(thumbnail_path=args.thumbnail, vid=args.video_vid, **extra)

    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
