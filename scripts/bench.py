"""Aggregates results/turns.jsonl into results/latency.json: p50/p90 per stage plus the stack."""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from murmur.config import get_settings

STAGES = [
    "endpoint",
    "stt_final",
    "memory_recall",
    "llm_first_token",
    "llm_first_sentence",
    "tts_first_audio",
    "time_to_first_audio",
    "barge_in",
]
PACKAGES = [
    "faster-whisper",
    "ctranslate2",
    "kokoro-onnx",
    "onnxruntime-gpu",
    "fastembed",
    "openai",
    "fastapi",
]


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "n": 0}
    return {
        "p50": round(float(np.percentile(values, 50))),
        "p90": round(float(np.percentile(values, 90))),
        "n": len(values),
    }


def main() -> int:
    settings = get_settings()
    turns_path = settings.results_dir / "turns.jsonl"
    if not turns_path.exists():
        print(f"no turns recorded yet at {turns_path}; talk to it first")
        return 1

    turns = [json.loads(line) for line in turns_path.read_text().splitlines() if line.strip()]
    by_backend: dict[str, dict[str, dict[str, float | None]]] = {}
    for backend in sorted({t["backend"] for t in turns}):
        mine = [t for t in turns if t["backend"] == backend]
        by_backend[backend] = {}
        for stage in STAGES:
            # Barge-in turns stop early and would skew the other stages; the barge-in stage
            # exists only on them.
            rows = [t for t in mine if t["barge_in"] == (stage == "barge_in")]
            by_backend[backend][stage] = percentiles(
                [t["ms"][stage] for t in rows if t["ms"].get(stage) is not None]
            )

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "turns": len(turns),
        "backends": by_backend,
        "stack": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gpu": gpu_name(),
            "llm_model": settings.llm_model,
            "stt_model": settings.stt_model,
            "stt_device": settings.stt_device,
            "tts_voice": settings.tts_voice,
            "tts_handoff": settings.tts_handoff,
            "tts_first_chunk": settings.tts_first_chunk,
            "tts_device": settings.tts_device,
            "endpoint_commit_ms": settings.endpoint_commit_ms,
            "llm_max_tokens": settings.llm_max_tokens,
            "endpoint_silence_ms": settings.endpoint_silence_ms,
            "packages": {p: version(p) for p in PACKAGES},
        },
    }
    out = Path(settings.results_dir) / "latency.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(by_backend, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
