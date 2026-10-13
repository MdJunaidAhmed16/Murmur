"""Plays WAV clips into the running server as if spoken into the browser, one turn per clip.

uv run python scripts/replay.py clips/*.wav
"""

import asyncio
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import websockets

from murmur.audio.frames import FRAME_MS, FRAME_SAMPLES, INPUT_SAMPLE_RATE
from murmur.config import get_settings

# Long enough for endpointing to fire and the reply to finish before the next clip.
TRAILING_SILENCE_S = 1.0
REPLY_TIMEOUT_S = 30.0
IDLE_S = 2.0


def load_clip(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != INPUT_SAMPLE_RATE:
        n = int(len(audio) * INPUT_SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        audio = audio.astype(np.int16)
    silence = np.zeros(int(TRAILING_SILENCE_S * INPUT_SAMPLE_RATE), dtype=np.int16)
    audio = np.concatenate([audio, silence])
    usable = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
    return audio[:usable].tobytes()


async def play(ws: Any, pcm: bytes) -> None:
    frame_bytes = FRAME_SAMPLES * 2
    for i in range(0, len(pcm), frame_bytes):
        await ws.send(pcm[i : i + frame_bytes])
        # Real-time pacing: endpointing counts frames, but the server's clock should see real gaps.
        await asyncio.sleep(FRAME_MS / 1000)


async def run_clip(ws: Any, pcm: bytes) -> list[dict[str, Any]]:
    """Returns every turn record the clip produced; pauses can split one clip into several."""
    sender = asyncio.create_task(play(ws, pcm))
    turns: list[dict[str, Any]] = []
    reply: list[str] = []
    audio_bytes = 0
    in_flight = 0
    while True:
        # Once the clip is fully sent and no turn is running, a short quiet spell means done.
        idle = sender.done() and in_flight == 0
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=IDLE_S if idle else REPLY_TIMEOUT_S)
        except TimeoutError:
            if idle:
                return turns
            raise
        if isinstance(message, bytes):
            audio_bytes += len(message)
            continue
        event = json.loads(message)
        if event["type"] == "user":
            in_flight += 1
            print(f"  heard:   {event['text']}")
        elif event["type"] == "assistant_delta":
            reply.append(event["text"])
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
        elif event["type"] == "metrics":
            in_flight -= 1
            print(f"  replied: {''.join(reply).strip()}")
            print(f"  audio:   {audio_bytes / 2 / 24_000:.1f}s")
            print(f"  ms:      {event['ms']}")
            if event["barge_in"]:
                print("  barge-in: cut off (did endpointing split one utterance?)")
            turns.append(event)
            reply, audio_bytes = [], 0


async def main(paths: list[Path]) -> int:
    settings = get_settings()
    async with websockets.connect(f"ws://{settings.host}:{settings.port}/ws") as ws:
        config = json.loads(await ws.recv())
        assert config["type"] == "config", config
        for path in paths:
            print(path.name)
            turns = await run_clip(ws, load_clip(path))
            if len(turns) > 1:
                print(f"  split into {len(turns)} turns")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(asyncio.run(main([Path(p) for p in sys.argv[1:]])))
