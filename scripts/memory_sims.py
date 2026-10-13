"""For each eval question: similarity of the best correct fact vs the best wrong one.

Shows where a recall threshold can separate them. Reads the store memory_eval.py left behind.

    uv run python scripts/memory_sims.py
"""

import re
import sys

from memory_eval import FACTS

from murmur.config import get_settings
from murmur.memory import open_memory


def main() -> int:
    settings = get_settings()
    settings = settings.model_copy(
        update={"memory_db": settings.results_dir / "memory_eval" / "memory.db"}
    )
    memory = open_memory(settings)
    print(f"{'question':45s} {'correct':>8s} {'wrong':>8s}  best correct fact")
    for fact in FACTS:
        pairs = memory.similar(fact.question, threshold=-1.0, include_superseded=False)
        right = [(m, s) for m, s in pairs if re.search(fact.key, m.text, re.IGNORECASE)]
        wrong = [(m, s) for m, s in pairs if not re.search(fact.key, m.text, re.IGNORECASE)]
        r = f"{right[0][1]:.3f}" if right else "absent"
        w = f"{wrong[0][1]:.3f}" if wrong else "-"
        print(f"{fact.question:45s} {r:>8s} {w:>8s}  {right[0][0].text if right else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
