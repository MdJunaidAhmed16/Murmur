"""Endpointing's two failure modes, measured with the VAD alone (CPU, seconds).

Every clip is one utterance, so every endpoint before its end is a premature cut — the "too
aggressive" failure. The "too patient" failure is the silence threshold itself: it is added
to every turn's time to first audio. Also prints the longest pause inside each utterance.

    uv run python scripts/endpoint_sweep.py clips/*.wav results/gate2-clips/*.wav
"""

import json
import sys
from pathlib import Path

import numpy as np
from replay import load_clip

from murmur.audio.frames import FRAME_MS, FRAME_SAMPLES, pcm16_to_float
from murmur.audio.vad import EndpointDetector, SileroVAD, VadEvent
from murmur.config import get_settings

THRESHOLDS_MS = [200, 300, 400, 500, 600, 700, 800]


def speech_flags(vad: SileroVAD, pcm: bytes, threshold: float) -> list[bool]:
    vad.reset()
    audio = pcm16_to_float(pcm)
    frames = audio[: len(audio) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    return [vad.speech_probability(f) >= threshold for f in frames]


def endpoints(flags: list[bool], silence_ms: int, min_speech_ms: int) -> int:
    detector = EndpointDetector(silence_ms, min_speech_ms, onset_ms=min_speech_ms)
    return sum(detector.update(f) is VadEvent.ENDPOINT for f in flags)


def longest_inner_pause_ms(flags: list[bool]) -> int:
    speech = [i for i, f in enumerate(flags) if f]
    if not speech:
        return 0
    longest = run = 0
    for f in flags[speech[0] : speech[-1] + 1]:
        run = 0 if f else run + 1
        longest = max(longest, run)
    return longest * FRAME_MS


def main(paths: list[Path]) -> int:
    settings = get_settings()
    vad = SileroVAD(settings.models_dir / "silero_vad.onnx")
    flags = {p.name: speech_flags(vad, load_clip(p), settings.vad_threshold) for p in paths}

    print(f"{len(paths)} single-utterance clips, min_speech_ms={settings.min_speech_ms}")
    print("\nlongest pause inside each utterance (ms):")
    pauses = {name: longest_inner_pause_ms(f) for name, f in flags.items()}
    for name, ms in sorted(pauses.items(), key=lambda kv: -kv[1]):
        print(f"  {ms:5d}  {name}")

    print(f"\n{'silence_ms':>10} {'clips cut early':>16} {'extra endpoints':>16} {'missed':>7}")
    rows = []
    for threshold in THRESHOLDS_MS:
        counts = [endpoints(f, threshold, settings.min_speech_ms) for f in flags.values()]
        cut = sum(c > 1 for c in counts)
        extra = sum(max(0, c - 1) for c in counts)
        missed = sum(c == 0 for c in counts)  # never detected as speech at all
        rows.append({"silence_ms": threshold, "clips_cut": cut, "extra": extra, "missed": missed})
        print(f"{threshold:>10} {cut:>10} / {len(counts):<4} {extra:>16} {missed:>7}")

    out = settings.results_dir / "endpoint_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "clips": [p.name for p in paths],
                "vad_threshold": settings.vad_threshold,
                "min_speech_ms": settings.min_speech_ms,
                "inner_pause_ms": pauses,
                "sweep": rows,
                "p90_inner_pause_ms": float(np.percentile(list(pauses.values()), 90)),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main([Path(p) for p in sys.argv[1:]]))
