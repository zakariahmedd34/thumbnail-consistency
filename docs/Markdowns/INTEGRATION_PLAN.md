# M2-Generator Integration Plan

> Integrate the three VLM notebooks (`vlm_query_generation`, `vlm_aspect_extraction_test`,
> `aspect_extraction_pipeline`) as interchangeable M2 generators inside the existing
> consistency-scoring pipeline. Refactor the project so the only thing that varies
> across experiments is the M2 module.

---

## Phase 1 — Analysis (what each piece actually does)

### 1.1 Existing project structure (verified)

```
src/
├── models/
│   ├── clipcap.py           Prefix-MLP + frozen GPT-2 (trained component)
│   ├── encoders.py          Cached CLIP-img / CLIP-text / GPT-2 loaders
│   └── moment_detr.py       M1 wrapper (feature path + raw-video)
├── scoring/
│   ├── aggregate.py         Pure-math: pool, per_chunk, multi-query agg
│   ├── chunking.py          75-clip window splitter
│   └── consistency.py       End-to-end ConsistencyScorer
├── eval/metrics.py          AUC, Cohen's d, Welch's t
├── utils/{io,seed}.py
└── data/m2_dataset.py       PyTorch Dataset for training pairs
```

### 1.2 The three VLM notebooks (verified by reading every cell)

| Notebook | Model | Input | Output | Role |
|---|---|---|---|---|
| `vlm_query_generation.ipynb` | **Qwen2-VL 7B** (4-bit) | thumbnail | **single query string** | Direct M2 replacement, simplest |
| `vlm_aspect_extraction_test.ipynb` (cells 10-11) | **Qwen2-VL 7B** (4-bit) | thumbnail | **{aspects: dict, query: str}** | M2 with structured intermediate |
| `vlm_aspect_extraction_test.ipynb` (cells 15-16) | **LLaVA-1.6 Mistral 7B** (4-bit) | thumbnail | **{aspects: dict, query: str}** | Same as above, different backbone |
| `aspect_extraction_pipeline.ipynb` | **Qwen2.5-3B** (text-only) | **existing query text** | rewritten query | NOT an M2 — text-side post-processor / data augmenter |

### 1.3 Shared properties

| | Output schema | Inference cost (T4) | VRAM |
|---|---|---|---|
| ClipCap | `List[str]` (K queries) | ~50 ms | ~700 MB |
| Qwen2-VL-Simple | `str` (one query) | ~3-5 s | ~5 GB |
| Qwen2-VL-Aspect | `{aspects, query}` | ~5-8 s | ~5 GB |
| LLaVA-1.6 | `{aspects, query}` | ~5-7 s | ~6 GB |
| Aspect rewriter | `str` (cleaned query) | ~1 s | ~3 GB |

### 1.4 Reusability map

| Component | Reuse status |
|---|---|
| CLIP-text encoding | ✅ Reuse `src/models/encoders.py::encode_text` for every generator's output |
| Moment-DETR M1 | ✅ Single shared instance for all eval runs |
| Saliency pooling | ✅ Pure-math, no change needed |
| Score aggregation | ✅ Same `consistency_score(...)` for everyone |
| Eval sets | ✅ Same three JSONL files (`consistency_eval{,_hard,_query_mismatch}.jsonl`) |
| AUC / metrics | ✅ Same `src/eval/metrics.py` |
| Image preprocessing | ⚠️ Each VLM has its own processor — must be encapsulated inside the generator |
| Tokenizer | ⚠️ Per-model, encapsulated |
| Quantization config | ⚠️ Per-model (bnb 4-bit nf4 standard for both VLMs) |

### 1.5 Incompatibilities to resolve

| Issue | Fix |
|---|---|
| ClipCap generates K=3 nucleus samples; VLMs return 1 deterministic query | Wrap VLMs with optional `temperature>0` resampling to produce K queries |
| Aspect-pipeline generators return dicts | Adapter that flattens `{aspects, query}` → `[query]` for downstream uniformity |
| Each VLM has its own image processor | Hide behind the generator interface — never expose externally |
| 4-bit quantization needs `bitsandbytes` (~CUDA-only) | Document as GPU-only at the registry level; raise on CPU |

---

## Phase 2 — Approach Comparison

### 2.1 Master comparison table

| Method | Reads thumb? | Reads frames? | Uses CLIP? | Caption gen? | LLM decode? | Train? | Cost (sec/img) | VRAM | Output format |
|---|---|---|---|---|---|---|---|---|---|
| **GT-query** | ❌ | ❌ | ❌ (text already exists) | ❌ | ❌ | ❌ | ~0 | 0 | string |
| **Random-query** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ~0 | 0 | string |
| **Thumb-only (α=0)** | ✅ | ❌ | ✅ image | ❌ | ❌ | ❌ | ~10 ms | 600 MB | (no query — uses e_T direct) |
| **ClipCap** | ✅ | ❌ | ✅ image | ✅ (GPT-2) | ✅ | ✅ (prefix MLP only) | ~50 ms | 700 MB | string |
| **CapCLIP** (vendored upstream) | ✅ | ❌ | ✅ image | ✅ (CLIP-decoder) | ❌ | ✅ (decoder head) | ~80 ms | 800 MB | string |
| **Qwen2-VL-simple** | ✅ | ❌ | indirect (image tokens) | ✅ | ✅ | ❌ | ~4 s | 5 GB | string |
| **Qwen2-VL-aspect** | ✅ | ❌ | indirect | ✅ | ✅ | ❌ | ~6 s | 5 GB | {aspects, query} |
| **LLaVA-1.6** | ✅ | ❌ | indirect | ✅ | ✅ | ❌ | ~6 s | 6 GB | {aspects, query} |
| **Aspect-rewriter** (text-only) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ~1 s | 3 GB | string (post-processed) |
| **Fusion** (Qwen+aspects) | ✅ | ❌ | indirect | ✅ | ✅ | ❌ | ~7 s | 5 GB | string (sentence-from-aspects) |

### 2.2 Strengths / weaknesses / failure modes

| Method | Strengths | Weaknesses | Expected failure mode |
|---|---|---|---|
| **ClipCap** | Fast, GPU-cheap, deterministic, trainable on small data | Limited vocabulary; bound to training distribution; tends to produce generic queries | Generic captions ("a man talks to camera") on out-of-domain thumbs |
| **Qwen2-VL-simple** | World knowledge, fluent English, generalizes across domains | Heavy compute; latency makes batch eval slow; may hallucinate details | Confident hallucinations (mentions things not in image) |
| **Qwen2-VL-aspect** | Structured output enables post-processing/ablation per aspect | Slower; JSON parse failures degrade pipeline | Parse failure → falls back to raw query |
| **LLaVA-1.6** | Different visual backbone (CLIP-ViT-L) than Qwen; complementary errors | Same compute cost; ~10% JSON parse failure rate in our test | Captions slightly more verbose than Qwen |
| **Aspect-rewriter** (text-only) | Normalizes any generator's output to consistent style; cheap | Doesn't see image — can't correct visual errors | Compounds upstream errors |
| **Fusion** | Combines visual grounding (VLM) + style normalization (rewriter) | Two-stage = slower; risk of error accumulation | Same as VLM but slightly slower |

### 2.3 Expected effect on consistency AUC (predictions, not measurements)

On `consistency_eval_hard.jsonl` (within-video, the research-claim eval):

| M2 generator | Predicted AUC range | Reasoning |
|---|---|---|
| GT-query (ceiling) | 0.638 (already measured) | Upper bound |
| ClipCap-k3 | 0.55-0.65 | Trained on the exact domain; close to GT |
| Qwen2-VL-simple | 0.55-0.68 | Generalizes better; may exceed ClipCap when query is rich |
| Qwen2-VL-aspect → query | 0.55-0.68 | Similar to simple; structured intermediate may help |
| LLaVA-1.6 → query | 0.55-0.67 | Comparable to Qwen2-VL |
| Aspect-rewriter(Qwen) | 0.58-0.70 | Could outperform raw VLM if rewriter cleans noise |
| Random | ~0.50 | Control |

These are research questions. The ablation table is the deliverable.

---

## Phase 3 — Unified Integration Design

### 3.1 Abstract base class

```python
# src/m2_generators/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path

class M2Generator(ABC):
    """Interchangeable thumbnail -> query generator."""

    @abstractmethod
    def generate(self, thumbnail: str | Path, k: int = 1, **kwargs) -> List[str]:
        """Return a list of K query strings for the given thumbnail."""

    def generate_with_metadata(self, thumbnail: str | Path, k: int = 1, **kwargs) -> Dict[str, Any]:
        """Default: return queries + minimal metadata. Override for richer info."""
        import time
        t0 = time.time()
        queries = self.generate(thumbnail, k=k, **kwargs)
        return {
            'queries'      : queries,
            'm2_name'      : self.name,
            'elapsed_ms'   : (time.time() - t0) * 1000.0,
            'metadata'     : {},
        }

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in configs and output filenames."""

    @property
    def gpu_required(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"M2Generator({self.name})"
```

### 3.2 Registry / factory

```python
# src/m2_generators/__init__.py
from typing import Dict, Type, Any
from .base import M2Generator

_REGISTRY: Dict[str, Type[M2Generator]] = {}

def register(name: str):
    def wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return wrap

def build_m2(name: str, **kwargs) -> M2Generator:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown M2 generator: {name}. "
                         f"Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)

def available() -> list:
    return sorted(_REGISTRY.keys())

# Eagerly import to populate registry
from . import clipcap_generator        # noqa: F401
from . import qwen_vl_generator        # noqa: F401
from . import llava_generator          # noqa: F401
from . import aspect_rewriter          # noqa: F401
from . import ground_truth_generator   # noqa: F401
from . import random_generator         # noqa: F401
```

### 3.3 Concrete implementations (skeletons)

```python
# src/m2_generators/clipcap_generator.py
import torch
from pathlib import Path
from typing import List
from .base import M2Generator
from . import register

@register("clipcap")
class ClipCapGenerator(M2Generator):
    def __init__(self, ckpt_path: str, prefix_len: int = 10, gpt2_name: str = "gpt2"):
        from src.models.clipcap import ClipCapM2
        from src.models.encoders import get_clip
        self.model = ClipCapM2(prefix_len=prefix_len, gpt2_name=gpt2_name)
        sd = torch.load(ckpt_path, map_location='cpu')['model']
        self.model.load_state_dict(sd)
        self.clip_model, self.clip_proc, _, self.device = get_clip()
        self.model.to(self.device).eval()

    @property
    def name(self) -> str:
        return "clipcap"

    @torch.no_grad()
    def generate(self, thumbnail, k=1, top_p=0.9, temperature=1.1, max_new_tokens=25):
        from PIL import Image
        img = Image.open(thumbnail).convert("RGB")
        inp = self.clip_proc(images=img, return_tensors="pt").to(self.device)
        # bypass get_image_features (HF-version safety)
        out = self.clip_model.vision_model(pixel_values=inp["pixel_values"])
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        if pooled.shape[-1] != self.clip_model.config.projection_dim:
            pooled = self.clip_model.visual_projection(pooled)
        e_T = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        generations = self.model.generate(
            e_T, num_return_sequences=k, do_sample=k > 1,
            top_p=top_p, temperature=temperature,
            max_new_tokens=max_new_tokens, no_repeat_ngram_size=2,
        )
        return [s.strip() for s in generations[0]]
```

```python
# src/m2_generators/qwen_vl_generator.py
import torch
from pathlib import Path
from typing import List, Dict
from .base import M2Generator
from . import register

@register("qwen2vl_simple")
class QwenVLSimpleGenerator(M2Generator):
    PROMPT = (
        "Describe what is happening in this image in one short sentence. "
        "Start with a subject (a person, people, an object). "
        "Only describe what is visually present. "
        "Do not ask a question. Do not add opinions or assumptions."
    )

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct", load_in_4bit: bool = True):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    @property
    def name(self) -> str:
        return "qwen2vl_simple"

    @torch.no_grad()
    def generate(self, thumbnail, k=1, max_new_tokens=60, do_sample=False, temperature=1.0):
        from qwen_vl_utils import process_vision_info
        from PIL import Image
        img = Image.open(thumbnail).convert("RGB")
        queries = []
        for _ in range(k):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": self.PROMPT},
            ]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=(k > 1) or do_sample,
                                       temperature=temperature)
            trimmed = out[0][inputs.input_ids.shape[1]:]
            queries.append(self.processor.decode(trimmed, skip_special_tokens=True).strip())
        return queries


@register("qwen2vl_aspect")
class QwenVLAspectGenerator(M2Generator):
    """Generates structured aspects then re-emits the query field."""
    PROMPT = """Look at this image carefully.
Extract ONLY what is visually present, then write ONE concise query sentence.
Respond ONLY in this exact format:
<aspects>{"living_being": [...], "action": [...], "object": [...], "location": [...], "other": [...]}</aspects>
<query>YOUR QUERY HERE</query>"""

    def __init__(self, **kw):
        # reuse the simple generator's model loader
        self._delegate = QwenVLSimpleGenerator(**kw)
        self.model = self._delegate.model
        self.processor = self._delegate.processor

    @property
    def name(self) -> str:
        return "qwen2vl_aspect"

    @torch.no_grad()
    def generate(self, thumbnail, k=1, **kw) -> List[str]:
        import re
        raw_outs = []
        for _ in range(k):
            self._delegate.PROMPT = self.PROMPT
            text = self._delegate.generate(thumbnail, k=1, max_new_tokens=512)[0]
            raw_outs.append(text)
        # parse <query>...</query>
        results = []
        for raw in raw_outs:
            m = re.search(r'<query>\s*(.+?)\s*</query>', raw, re.DOTALL)
            results.append(m.group(1).strip() if m else raw[:200])
        return results

    def generate_with_metadata(self, thumbnail, k=1, **kw):
        # also expose aspects in metadata for the output JSON
        import re, json
        outs = []
        raw_outputs = []
        self._delegate.PROMPT = self.PROMPT
        for _ in range(k):
            raw = self._delegate.generate(thumbnail, k=1, max_new_tokens=512)[0]
            raw_outputs.append(raw)
        queries, aspects_list = [], []
        for raw in raw_outputs:
            q = re.search(r'<query>\s*(.+?)\s*</query>', raw, re.DOTALL)
            a = re.search(r'<aspects>\s*(\{.*?\})\s*</aspects>', raw, re.DOTALL)
            queries.append(q.group(1).strip() if q else raw[:200])
            try:
                aspects_list.append(json.loads(a.group(1)) if a else None)
            except Exception:
                aspects_list.append(None)
        return {'queries': queries, 'm2_name': self.name,
                'metadata': {'aspects': aspects_list, 'raw_outputs': raw_outputs}}
```

```python
# src/m2_generators/llava_generator.py
@register("llava16_aspect")
class LLaVAAspectGenerator(M2Generator):
    PROMPT = QwenVLAspectGenerator.PROMPT  # same prompt, same parser

    def __init__(self, model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf", load_in_4bit: bool = True):
        # ... same 4-bit loading pattern ...
        pass
    # ... generate / generate_with_metadata mirror Qwen aspect version
```

```python
# src/m2_generators/ground_truth_generator.py
@register("gt")
class GTQueryGenerator(M2Generator):
    """Looks up the QVH ground-truth query by qid."""
    def __init__(self, lookup_jsonl: str):
        import json
        self._by_qid = {}
        with open(lookup_jsonl, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    self._by_qid[r['qid']] = r.get('query') or r.get('thumb_query')

    @property
    def name(self) -> str:
        return "gt"

    def generate(self, thumbnail, k=1, qid=None, **kw) -> list:
        if qid is None or qid not in self._by_qid:
            raise ValueError(f"GT generator needs qid; got {qid}")
        return [self._by_qid[qid]] * k

    @property
    def gpu_required(self) -> bool:
        return False
```

```python
# src/m2_generators/random_generator.py
@register("random")
class RandomQueryGenerator(M2Generator):
    """Control baseline — random query from a pool."""
    def __init__(self, pool_jsonl: str, seed: int = 2026):
        import json, random
        self._pool = []
        with open(pool_jsonl, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    q = r.get('query') or r.get('thumb_query')
                    if q: self._pool.append(q)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    @property
    def gpu_required(self) -> bool:
        return False

    def generate(self, thumbnail, k=1, **kw):
        return [self._rng.choice(self._pool) for _ in range(k)]
```

```python
# src/m2_generators/aspect_rewriter.py
@register("aspect_rewriter")
class AspectRewriter(M2Generator):
    """Wraps another generator: runs aspect extraction on its output, then
    reconstructs a normalized query. Useful for ablations."""
    def __init__(self, upstream: str | M2Generator, llm_id: str = "Qwen/Qwen2.5-3B-Instruct",
                 **upstream_kw):
        from . import build_m2
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.upstream = upstream if isinstance(upstream, M2Generator) else build_m2(upstream, **upstream_kw)
        self.tok = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.llm.eval()

    @property
    def name(self) -> str:
        return f"aspect_rewriter:{self.upstream.name}"

    @torch.no_grad()
    def generate(self, thumbnail, k=1, **kw):
        from .aspect_utils import extract_aspects, aspects_to_query  # see §3.5
        raw_queries = self.upstream.generate(thumbnail, k=k, **kw)
        return [aspects_to_query(extract_aspects(q, self.tok, self.llm)) for q in raw_queries]
```

### 3.4 The ConsistencyScorer becomes generator-agnostic

```python
# src/scoring/consistency.py — UPDATED to accept any M2Generator
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any
from src.m2_generators.base import M2Generator
from src.models.moment_detr import MomentDETRFeaturePredictor

@dataclass
class ScoringConfig:
    alpha: float = 0.5
    tau: float = 0.1
    top_p: float = 0.25
    chunk_window: int = 75
    chunk_stride: int = 60
    damp_with_fg: bool = False
    k_queries: int = 3


class ConsistencyScorer:
    def __init__(self, m1: MomentDETRFeaturePredictor, m2: M2Generator,
                 feature_dir, cfg: ScoringConfig = None):
        self.m1 = m1
        self.m2 = m2
        self.feature_dir = feature_dir
        self.cfg = cfg or ScoringConfig()

    def score(self, thumbnail_path: str, vid: str,
              qid: Optional[int] = None,
              precomputed_queries: Optional[Sequence[str]] = None,
              precomputed_e_T = None) -> Dict[str, Any]:
        # 1. Generate queries (or use precomputed)
        if precomputed_queries is not None:
            queries = list(precomputed_queries)
            m2_meta = {'source': 'precomputed'}
        else:
            m2_out = self.m2.generate_with_metadata(thumbnail_path, k=self.cfg.k_queries, qid=qid)
            queries = m2_out['queries']
            m2_meta = m2_out

        # 2. ... same as today: encode_text(queries), load video features,
        #    chunk, run M1 per chunk, saliency-pool, compute C
        # (omitted for brevity — see existing src/scoring/consistency.py)
        ...
        return {
            'consistency_score': C,
            'queries': queries,
            'm2_meta': m2_meta,
            # ... full schema in §4.3 below
        }
```

### 3.5 Aspect utilities (shared)

```python
# src/m2_generators/aspect_utils.py
# Shared aspect extraction + aspects→query reconstruction
# (drop-in from aspect_extraction_pipeline.ipynb, refactored into pure functions)

SYSTEM_PROMPT = """...(from notebook cell 02)..."""
EMPTY = {"living_being": [], "action": [], "object": [], "location": [], "other": []}

def extract_aspects(query: str, tokenizer, model) -> dict: ...
def aspects_to_query(aspects: dict) -> str: ...
def parse_aspect_output(text: str) -> dict: ...
```

---

## Phase 4 — Final Project Structure

```
thumbnail-consistency/
├── README.md
├── LICENSE
├── .gitignore
│
├── configs/                              ← per-experiment configs
│   ├── m2/
│   │   ├── clipcap.yaml
│   │   ├── qwen2vl_simple.yaml
│   │   ├── qwen2vl_aspect.yaml
│   │   ├── llava16_aspect.yaml
│   │   ├── aspect_rewriter.yaml
│   │   ├── gt.yaml
│   │   └── random.yaml
│   ├── consistency.yaml                  ← α / τ / top_p / chunking
│   ├── eval.yaml                          ← which eval set, which sources to ablate
│   └── drive.yaml                         ← Drive root paths (Colab-aware)
│
├── src/
│   ├── m2_generators/                    ★ NEW MODULE
│   │   ├── __init__.py                   (registry + build_m2)
│   │   ├── base.py                       (M2Generator ABC)
│   │   ├── clipcap_generator.py
│   │   ├── qwen_vl_generator.py
│   │   ├── llava_generator.py
│   │   ├── aspect_rewriter.py
│   │   ├── ground_truth_generator.py
│   │   ├── random_generator.py
│   │   └── aspect_utils.py
│   │
│   ├── models/
│   │   ├── clipcap.py                    ← ClipCap module (used by ClipCapGenerator)
│   │   ├── encoders.py                   (CLIP / GPT-2 loaders, frozen)
│   │   └── moment_detr.py                (M1 wrapper)
│   │
│   ├── scoring/
│   │   ├── aggregate.py                  pure math
│   │   ├── chunking.py
│   │   └── consistency.py                ConsistencyScorer (now takes any M2)
│   │
│   ├── caching/                          ★ NEW
│   │   ├── __init__.py
│   │   ├── query_cache.py                cache m2 outputs per (m2_name, frame_hash)
│   │   ├── text_emb_cache.py             cache CLIP-text embeddings keyed by query hash
│   │   └── saliency_cache.py             cache M1 outputs per (vid, query_hash)
│   │
│   ├── eval/
│   │   ├── metrics.py                    (AUC, Cohen's d, etc.)
│   │   ├── runner.py                     ★ NEW: orchestrates eval over a JSONL
│   │   └── ablation.py                   ★ NEW: multi-m2 ablation runner
│   │
│   ├── cli/                              ★ NEW
│   │   ├── __init__.py
│   │   ├── score.py                      single-pair entrypoint
│   │   ├── evaluate.py                   batch eval entrypoint
│   │   ├── ablate.py                     multi-m2 ablation entrypoint
│   │   └── generate_queries.py           pre-compute m2 outputs only (no scoring)
│   │
│   ├── api/                              ★ NEW (lightweight)
│   │   ├── __init__.py
│   │   ├── server.py                     FastAPI: POST /score
│   │   └── schemas.py                    Pydantic models for I/O
│   │
│   ├── data/
│   │   ├── m2_dataset.py
│   │   └── eval_loader.py                load + classify negatives
│   │
│   └── utils/
│       ├── io.py
│       ├── seed.py
│       └── hashing.py                    stable hashes for cache keys
│
├── data/
│   ├── DATA_README.md
│   ├── processed/                        ← jsonl contracts (committed)
│   ├── qvh_videos/, qvh_frames/, ...     ← (gitignored, on Drive)
│   └── processed/README.md
│
├── notebooks/
│   ├── final_data_collection_preparation.ipynb
│   ├── train_clipcap.ipynb
│   ├── run_evaluation.ipynb              ← now iterates configs/m2/*.yaml
│   ├── sanity_consistency.ipynb
│   ├── compare_m2_generators.ipynb       ★ NEW: side-by-side qualitative + AUC
│   └── demo_single_video.ipynb           ★ NEW: notebook mode
│
├── outputs/                              (gitignored)
│   ├── checkpoints/clipcap_best.pt
│   ├── caches/
│   │   ├── m2_queries/{m2_name}/{eval_id}.jsonl
│   │   ├── text_embs.npz                 (qhash -> 512d)
│   │   └── saliency/{vid}_{qhash}.npz
│   ├── predictions/
│   │   └── scores_{m2_name}_{eval_id}.jsonl
│   └── figures/
│       └── ablation_table.csv
│
├── scripts/                              (gitignored)
│   └── patch_data_notebook.py, build_*.py, ...
│
├── docs/
│   ├── 1-FINAL_DESIGN.md
│   ├── 2-GUIDE.md
│   ├── 3-EVALUATION.md
│   └── Markdowns/
│       ├── INTEGRATION_PLAN.md           ← (this file)
│       ├── FINAL_REPORT.md
│       └── ...
│
└── tests/
    ├── test_aggregate.py                 (existing 18 tests pass)
    ├── test_m2_registry.py               ★ NEW: registry contract
    └── test_score_schema.py              ★ NEW: output JSON schema
```

---

## Phase 5 — Inference Modes

### 5.1 Single-video / single-pair (CLI)

```bash
python -m src.cli.score \
    --thumbnail /path/to/thumb.jpg \
    --video-vid Exr8FNpynKo_210.0_360.0 \
    --m2 qwen2vl_simple \
    --m2-config configs/m2/qwen2vl_simple.yaml \
    --features-dir /content/features_local/features/clip_features \
    --m1-ckpt /content/.../model_best.ckpt \
    --output result.json
```

Input: file paths.
Output: JSON file matching §6 schema.

### 5.2 Batch folder inference (CLI)

```bash
python -m src.cli.evaluate \
    --eval-jsonl data/processed/consistency_eval_hard.jsonl \
    --m2 clipcap \
    --m2-config configs/m2/clipcap.yaml \
    --output outputs/predictions/scores_clipcap_hard.jsonl
```

Input: JSONL of pairs.
Output: JSONL with one score per pair + summary AUC at the end.

### 5.3 Multi-M2 ablation (CLI)

```bash
python -m src.cli.ablate \
    --eval-jsonl data/processed/consistency_eval_hard.jsonl \
    --m2-list clipcap qwen2vl_simple qwen2vl_aspect llava16_aspect gt random \
    --output-dir outputs/predictions/ \
    --table outputs/figures/ablation_table.csv
```

Runs all listed M2s on the same eval set, writes one JSONL per M2, prints
a comparison table.

### 5.4 API mode (FastAPI)

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8080
```

Then:

```bash
curl -X POST http://localhost:8080/score \
  -F "thumbnail=@thumb.jpg" \
  -F "vid=Exr8FNpynKo_210.0_360.0" \
  -F "m2=qwen2vl_simple"
```

### 5.5 Notebook / demo

```python
# notebooks/demo_single_video.ipynb
from src.m2_generators import build_m2
from src.models.moment_detr import MomentDETRFeaturePredictor
from src.scoring.consistency import ConsistencyScorer, ScoringConfig

m1 = MomentDETRFeaturePredictor(ckpt_path=..., repo_root=...)
m2 = build_m2("qwen2vl_simple")
scorer = ConsistencyScorer(m1=m1, m2=m2, feature_dir=..., cfg=ScoringConfig())

result = scorer.score(thumbnail_path="thumb.jpg",
                      vid="Exr8FNpynKo_210.0_360.0")
print(result["consistency_score"])
```

---

## Phase 6 — Output JSON Schema (canonical)

```json
{
  "schema_version": "1.0",
  "consistency_score": 0.642,
  "overall_score": 0.642,
  "m2_generator": "qwen2vl_simple",

  "generated_queries": [
    "A man in a red hoodie stretches on a sidewalk.",
    "A person in a red top exercises outdoors.",
    "Man stretching at the edge of the road."
  ],

  "per_query_scores": [
    {"query": "...", "score": 0.71, "saliency_peak": 0.87, "chunk_idx": 0},
    {"query": "...", "score": 0.61, "saliency_peak": 0.74, "chunk_idx": 0},
    {"query": "...", "score": 0.55, "saliency_peak": 0.66, "chunk_idx": 0}
  ],

  "temporal": {
    "predicted_spans": [[12.0, 28.0, 0.84], [55.0, 78.0, 0.41]],
    "max_fg_prob": 0.84,
    "n_chunks": 1,
    "best_chunk": 0
  },

  "saliency": {
    "scores_per_clip": [0.10, 0.20, 0.31, 0.55, 0.87, 0.78, 0.60, 0.40, 0.22, 0.15],
    "top_p_clips": [22, 23, 24, 25],
    "weighted_pool_l2": 1.0
  },

  "confidence": {
    "score_std_across_queries": 0.07,
    "fg_prob": 0.84,
    "saliency_entropy": 0.32
  },

  "thumbnail": {
    "path": "/path/to/thumb.jpg",
    "clip_embedding_l2": 1.0
  },

  "video": {
    "vid": "Exr8FNpynKo_210.0_360.0",
    "duration_s": 150.0,
    "n_clips": 75
  },

  "config": {
    "alpha": 0.5, "tau": 0.1, "top_p": 0.25,
    "k_queries": 3, "chunk_window": 75, "chunk_stride": 60
  },

  "explanation": {
    "score_breakdown": {
      "cos_q_clip_mean": 0.68,
      "cos_T_clip": 0.61,
      "alpha_weighted": 0.642
    },
    "m2_metadata": {
      "aspects": null,
      "raw_outputs": null
    },
    "elapsed_ms": {
      "m2_generation": 4200,
      "clip_text_encode": 12,
      "m1_inference": 95,
      "scoring": 3,
      "total": 4310
    },
    "cache_hits": {"m2_query": false, "text_emb": true, "saliency": false}
  }
}
```

---

## Phase 7 — Caching Strategy

Three caches keyed by stable hashes (see `src/utils/hashing.py`):

### 7.1 M2 query cache

```
outputs/caches/m2_queries/{m2_name}/{frame_hash}.json
```

Key: SHA-1 of `(m2_name, frame_path, k, model_version)`.
Value: list of K query strings + metadata + timestamp.

**Why**: VLMs are expensive (~5s/image). Cache lets you re-run scoring with
different α / τ / top_p without re-invoking the VLM.

### 7.2 CLIP-text embedding cache

```
outputs/caches/text_embs.npz
  arrays: {qhash: np.array (512,), ...}
```

Key: SHA-1 of the query string.
Value: 512-d L2-normalized embedding.

**Why**: Many M2 generators produce overlapping queries. Dedup before embedding.

### 7.3 M1 saliency cache

```
outputs/caches/saliency/{vid}_{qhash}.npz
  arrays: saliency (n_clips,), pred_spans (10, 3), max_fg_prob (float)
```

Key: SHA-1 of `(vid, query, m1_ckpt_hash)`.
Value: per-clip saliency + top-10 spans + max foreground prob.

**Why**: Same query + same video = same M1 output. Saves ~90 ms per call.

### 7.4 Invalidation policy

- M2 cache invalidates on `model_version` bump (track in cfg).
- Text-emb cache invalidates only if CLIP weights change (rare).
- Saliency cache invalidates on M1 ckpt change (track ckpt SHA in key).

---

## Phase 8 — Migration / Integration Order

| Step | Action | Owner | Time |
|---|---|---|---|
| 1 | Create `src/m2_generators/base.py` + `__init__.py` (registry) | A | 30 min |
| 2 | Wrap existing `ClipCapM2` → `ClipCapGenerator(M2Generator)` | A | 30 min |
| 3 | Add `GTQueryGenerator`, `RandomQueryGenerator` | A | 20 min |
| 4 | Port `vlm_query_generation.ipynb` → `QwenVLSimpleGenerator` | B | 1 h |
| 5 | Port `vlm_aspect_extraction_test.ipynb` Qwen branch → `QwenVLAspectGenerator` | B | 1 h |
| 6 | Port LLaVA branch → `LLaVAAspectGenerator` | B | 30 min |
| 7 | Port `aspect_extraction_pipeline.ipynb` → `aspect_utils.py` + `AspectRewriter` | B | 1 h |
| 8 | Refactor `ConsistencyScorer.__init__` to accept `m2: M2Generator` | A | 30 min |
| 9 | Build `src/cli/score.py`, `evaluate.py`, `ablate.py` | C | 2 h |
| 10 | Build `src/caching/*.py` (3 simple key-value stores on disk) | C | 1 h |
| 11 | Build FastAPI `src/api/server.py` | C | 1 h |
| 12 | Add `tests/test_m2_registry.py` (contract test) | A | 30 min |
| 13 | Update `notebooks/run_evaluation.ipynb` to loop over configs | D | 30 min |
| 14 | Write `notebooks/compare_m2_generators.ipynb` (qualitative) | D | 1 h |
| 15 | Update `README.md` with new structure | D | 30 min |

Total: ~12 person-hours. With 3 people in parallel: ~4-5 hours wall time.

---

## Phase 9 — Three Deployment Options

### 9.1 Lightweight (CPU-only, demo)

- M2: `gt` (no model) OR `clipcap` (small, runs on CPU in ~1s)
- M1: feature path only (loads ckpt, no raw-video extraction)
- Use case: hosted demo with pre-extracted features

```yaml
# configs/m2/clipcap_cpu.yaml
m2: clipcap
m2_kwargs: { ckpt_path: outputs/checkpoints/clipcap_best.pt }
device: cpu
k_queries: 1
```

### 9.2 Production default (T4-class GPU)

- M2: `qwen2vl_simple` (4-bit, ~5 GB VRAM, ~4 s/image)
- M1: feature path, GPU
- K=3 queries (use sampling for diversity)
- Caching enabled

```yaml
# configs/m2/qwen2vl_simple.yaml
m2: qwen2vl_simple
m2_kwargs: { model_id: Qwen/Qwen2-VL-7B-Instruct, load_in_4bit: true }
k_queries: 3
```

### 9.3 High-accuracy research (A100-class)

- M2: ensemble — `aspect_rewriter` wrapping `qwen2vl_aspect`
- K=5 nucleus samples
- Use both `consistency_eval.jsonl` AND `consistency_eval_hard.jsonl` simultaneously
- Cache aggressively

```yaml
# configs/m2/ensemble.yaml
m2: aspect_rewriter
m2_kwargs:
  upstream: qwen2vl_aspect
  llm_id: Qwen/Qwen2.5-3B-Instruct
k_queries: 5
```

---

## Phase 10 — Recommended Default Pipeline

For V1 submission, the recommended evaluation pipeline:

```
Inputs:  thumbnail.jpg + vid + (optional qid for GT lookup)
         |
         v
   M2: build_m2("qwen2vl_simple")    (production default)
         |
         v
   K=3 nucleus-sampled queries
         |
         v
   CLIP-text encode (cached if dup queries)
         |
         v
   M1 inference per (chunk, query) pair (cached on disk)
         |
         v
   Saliency-softmax pool (α=0.5, τ=0.1, top_p=0.25)
         |
         v
   C(T, V) = mean_k max_w [α·cos(e_q, e_clip) + (1-α)·cos(e_T, e_clip)]
         |
         v
   JSON response (Phase 6 schema)
```

**Ablation report**: same pipeline run with `m2 ∈ {gt, clipcap, qwen2vl_simple,
qwen2vl_aspect, llava16_aspect, random}` against the three eval sets
(cross-video, within-video, query-mismatch). One ablation table per eval set
× 6 M2 generators = 18-cell headline result.

---

## What this integration buys you vs the current state

| | Before | After |
|---|---|---|
| Compare ClipCap vs VLMs | Not possible — VLM code lives in 3 notebooks | One CLI command: `python -m src.cli.ablate --m2-list clipcap qwen2vl_simple llava16_aspect` |
| Add a new M2 generator | Edit notebook + duplicate scoring code | One file in `src/m2_generators/` + register decorator |
| Hosted demo | Reproduce a notebook | Single FastAPI endpoint |
| Cost per re-evaluation | Re-run VLMs every time | Cache hits make iteration ~50 ms/pair |
| Reproducibility | Notebook state varies | Stable hashes → identical results across runs |
| Output schema | Ad-hoc per notebook | Canonical JSON (Phase 6) |

---

## Migration risks

| Risk | Mitigation |
|---|---|
| VLMs OOM on T4 | All 4-bit quantized; document VRAM requirements per generator |
| Different processors per VLM break shared code | Hide behind `M2Generator.generate()` — no leakage |
| Parsing failures on aspect-style outputs | `generate_with_metadata` returns `aspects=None` on failure; scorer treats failure as "use raw output" |
| Cache invalidation on model version change | Track `model_version` field in cache keys; bump on ckpt update |
| Notebook integration breaks | Each notebook gets a 5-line bootstrap that imports from `src.*` |

---

## Acceptance — when is this integration done?

1. ☐ `tests/test_m2_registry.py` passes: every generator implements `generate()`, returns `List[str]`
2. ☐ `python -m src.cli.score --m2 clipcap` produces a JSON matching Phase 6 schema
3. ☐ Same command with `--m2 qwen2vl_simple` produces a JSON with `m2_generator: "qwen2vl_simple"` and longer `elapsed_ms.m2_generation`
4. ☐ `python -m src.cli.ablate` produces `outputs/figures/ablation_table.csv` with 6 rows
5. ☐ FastAPI demo at `/score` returns 200 with the schema on a valid input
6. ☐ Same `consistency_eval.jsonl` AUC numbers as before for ClipCap (no regression)
7. ☐ All 18 existing math tests still pass

When all 7 boxes check, the integration is shipped.
