"""Synthesises the scripted clips Gate 3 replays: barge-in pairs and tool questions.

A different Kokoro voice from the agent's, so the two are never confused in recordings.

    uv run python scripts/make_test_clips.py          # writes results/gate3-clips/*.wav
"""

import sys
import wave

import numpy as np
from kokoro_onnx import Kokoro

from murmur.config import get_settings

CLIPS = {
    # Barge-in pairs: a question with a long answer, then an interruption.
    "b1-ask.wav": "Tell me a long story about a fox who lives in a forest.",
    "b1-cut.wav": "Okay, stop, that's enough.",
    "b2-ask.wav": "Explain how a car engine works, in detail.",
    "b2-cut.wav": "Wait, actually, never mind.",
    "b3-ask.wav": "Can you describe the history of the Roman empire?",
    "b3-cut.wav": "Hold on, let me ask something else.",
    # Tools.
    "t1-time.wav": "What time is it right now?",
    "t2-calc.wav": "What is eighteen times forty two?",
    "t3-calc.wav": "What's fifteen percent of two thousand four hundred?",
}


def main() -> int:
    settings = get_settings()
    out = settings.results_dir / "gate3-clips"
    out.mkdir(parents=True, exist_ok=True)
    kokoro = Kokoro(
        str(settings.models_dir / "kokoro-v1.0.onnx"), str(settings.models_dir / "voices-v1.0.bin")
    )
    for name, text in CLIPS.items():
        audio, rate = kokoro.create(text, voice="bm_george", speed=1.0, lang="en-gb")
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(out / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        print(f"{name:14s} {len(pcm) / rate:4.1f}s  {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
