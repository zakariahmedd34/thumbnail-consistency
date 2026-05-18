"""Execute every code cell in pipeline_walkthrough.ipynb in order, sharing
a single global namespace (as Jupyter does). Catches import errors, runtime
errors, and shape bugs without needing nbconvert.

Skips the GPT-2 / ClipCap cells if `transformers` weights can't be downloaded
(no network) — those still PARSE cleanly, which is the main risk."""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "pipeline_walkthrough.ipynb"

# Pretend we're running from the project root (matches what users do).
sys.path.insert(0, str(NB.parents[1]))


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"Found {len(code_cells)} code cells in {NB.name}\n")

    ns: dict = {"__name__": "__main__"}
    failures = []

    for i, c in enumerate(code_cells):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        first_line = src.strip().split("\n", 1)[0][:80]
        print(f"--- cell {i:02d} --- {first_line}")
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception as e:
            failures.append((i, first_line, e))
            print(f"  [FAIL] {type(e).__name__}: {e}")
            if len(failures) == 1:
                print()
                traceback.print_exc()
                print()
        else:
            print("  [OK]")
    print()
    print(f"Passed: {len(code_cells) - len(failures)} / {len(code_cells)}")
    if failures:
        print("Failed cells:")
        for i, head, e in failures:
            print(f"  [{i:02d}] {head} -> {type(e).__name__}: {e}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
