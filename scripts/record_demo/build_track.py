"""Lays the demo lines on one timeline: the fake microphone for scripts/record_demo.

Each line starts at a fixed second. The gaps leave room for the agent's reply, except before
the interruption, which starts while the agent is still telling its story.

    uv run python scripts/record_demo/build_track.py   # -> results/demo/mic.wav
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np

CLIPS = Path("results/gate3-clips")
OUT = Path("results/demo")
RATE = 48_000
# (start second, clip)
SCHEDULE = [
    (2.0, "d1-intro.wav"),
    (15.0, "d2-calc.wav"),
    (25.0, "d3-story.wav"),
    (32.5, "d4-cut.wav"),  # barge-in: lands ~3 s into the story
    (41.0, "d5-recall.wav"),
    (53.0, "d6-forget.wav"),
]
TOTAL_S = 68.0


def read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        rate = w.getframerate()
    n = int(len(audio) * RATE / rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    track = np.zeros(int(TOTAL_S * RATE), dtype=np.float32)
    for start, name in SCHEDULE:
        clip = read(CLIPS / name)
        i = int(start * RATE)
        track[i : i + len(clip)] += clip[: len(track) - i]
    with wave.open(str(OUT / "mic.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())
    (OUT / "schedule.json").write_text(json.dumps({"total_s": TOTAL_S, "lines": SCHEDULE}))
    print(f"wrote {OUT / 'mic.wav'} ({TOTAL_S:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
