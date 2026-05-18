"""Abstract base class for M2 (thumbnail -> query) generators.

Every concrete generator must subclass M2Generator and implement
`generate(thumbnail, k=1)` returning a List[str] of K query strings.

The optional `generate_with_metadata(...)` returns the same queries plus
generator-specific metadata (e.g. structured aspects, raw model output,
latency breakdown) for inclusion in the canonical output JSON schema.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


class M2Generator(ABC):
    """Interchangeable thumbnail -> query generator.

    Contract:
      - generate(thumbnail, k=1) MUST return a list of K strings.
      - name MUST be a short stable identifier (used in config + cache paths).
      - gpu_required is informational; downstream code may use it to decide
        whether the generator can run on a CPU-only runtime.
    """

    @abstractmethod
    def generate(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """Return K query strings for the given thumbnail image.

        Parameters
        ----------
        thumbnail : path to the thumbnail image (.jpg/.png/.webp)
        k         : number of queries to return; deterministic generators may
                    return [q] * k.
        kwargs    : generator-specific hints (e.g. qid for GTQueryGenerator,
                    temperature for nucleus sampling).
        """

    def generate_with_metadata(
        self,
        thumbnail: Union[str, Path],
        k: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return {queries, m2_name, elapsed_ms, metadata} for the JSON schema.

        Default implementation wraps generate() and adds timing. Subclasses
        may override to include richer metadata (e.g. aspect dicts, raw text).
        """
        import time
        t0 = time.time()
        queries = self.generate(thumbnail, k=k, **kwargs)
        return {
            "queries": queries,
            "m2_name": self.name,
            "elapsed_ms": (time.time() - t0) * 1000.0,
            "metadata": {},
        }

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in configs, cache paths, output filenames."""

    @property
    def gpu_required(self) -> bool:
        """True if this generator requires CUDA (overridden by lightweight ones)."""
        return True

    @property
    def model_version(self) -> str:
        """Used as a cache-key component. Bump when the model changes."""
        return "v1"

    def __repr__(self) -> str:
        return f"M2Generator({self.name})"
