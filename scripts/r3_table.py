"""Prints the R3 before/after table from results/r3/*/latency.json, ready to paste.

uv run python scripts/r3_table.py
"""

import json
import sys
from pathlib import Path

from murmur.config import get_settings

CONFIGS = ["baseline", "clause", "gpu", "final"]
STAGES = [
    ("endpoint", "Endpointing"),
    ("stt_final", "Transcription (final)"),
    ("memory_recall", "Memory recall"),
    ("llm_first_token", "Model, first token"),
    ("llm_first_sentence", "Model, first chunk"),
    ("tts_first_audio", "Speech, first audio"),
    ("time_to_first_audio", "**Time to first audio**"),
]


def load(path: Path) -> dict[str, dict[str, float | None]] | None:
    if not path.exists():
        return None
    backends = json.loads(path.read_text())["backends"]
    return backends.get("local")


def cell(stats: dict[str, dict[str, float | None]] | None, stage: str) -> str:
    if stats is None or stats.get(stage, {}).get("p50") is None:
        return "TBD"
    s = stats[stage]
    return f"{s['p50']:.0f} / {s['p90']:.0f}"


def main() -> int:
    root = get_settings().results_dir / "r3"
    runs = {name: load(root / name / "latency.json") for name in CONFIGS}
    print("p50 / p90 ms\n")
    print("| Stage | " + " | ".join(CONFIGS) + " |")
    print("| --- |" + " --- |" * len(CONFIGS))
    for stage, label in STAGES:
        print(f"| {label} | " + " | ".join(cell(runs[c], stage) for c in CONFIGS) + " |")

    barge = load(root / "barge-in" / "latency.json")
    print(f"\nBarge-in (onset to cut), p50 / p90: {cell(barge, 'barge_in')} ms")
    for name in [*CONFIGS, "barge-in"]:
        vram = root / name / "vram.txt"
        if vram.exists():
            print(f"VRAM {name}: {vram.read_text().strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
