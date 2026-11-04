"""Config loading, path resolution, and small IO helpers.

Everything reads settings through `cfg()` and paths through the helpers here,
so there is exactly one place that decides where files live.

Path rules
----------
* `config.yaml` -> `data.*` paths are relative to the data dir.
* The data dir is `data/` in the repo, or `$THUMB_DATA_DIR` if set.
* The processed JSONL files were written on Colab and contain absolute paths like
  `/content/drive/MyDrive/thumbnail_project/qvh_frames/X.jpg`. `data_path()`
  rewrites that prefix to the local data dir, so the files work unchanged.
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
LEGACY_PREFIX = "/content/drive/MyDrive/thumbnail_project/"


@lru_cache(maxsize=1)
def cfg() -> Dict[str, Any]:
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir() -> Path:
    env = os.environ.get("THUMB_DATA_DIR")
    return Path(env) if env else REPO_ROOT / cfg()["data"]["data_dir"]


def data_path(p: str | Path) -> Path:
    """Resolve a data path: legacy Colab prefix -> data dir; relative -> data dir."""
    s = str(p).replace("\\", "/")
    if s.startswith(LEGACY_PREFIX):
        return data_dir() / s[len(LEGACY_PREFIX):]
    p = Path(p)
    return p if p.is_absolute() else data_dir() / p


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def features_dir() -> Path:
    return data_path(cfg()["data"]["features_dir"])


def eval_set(name: str) -> Path:
    sets = cfg()["data"]["eval_sets"]
    if name in sets:
        return data_path(sets[name])
    return Path(name)  # allow passing a file path directly


def m1_checkpoint() -> Path:
    return repo_path(cfg()["moment_detr"]["checkpoint"])


def moment_detr_repo() -> Path:
    return repo_path(cfg()["moment_detr"]["repo"])


def outputs_dir() -> Path:
    return repo_path(cfg()["outputs_dir"])


def generator_kwargs(name: str) -> Dict[str, Any]:
    """Config kwargs for a generator; path-like values resolved against the repo."""
    kw = dict(cfg().get("generators", {}).get(name, {}) or {})
    if "ckpt_path" in kw:
        kw["ckpt_path"] = str(repo_path(kw["ckpt_path"]))
    return kw


# ---------- small IO helpers -------------------------------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
