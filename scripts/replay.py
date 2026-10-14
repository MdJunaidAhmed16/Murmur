"""Plays WAV clips into the running server as if spoken into the browser.

    uv run python scripts/replay.py clips/*.wav                      # one turn per clip
    uv run python scripts/replay.py --interrupt-after 1.5 ask.wav cut.wav ...

With --interrupt-after, clips go in pairs: the second starts that many seconds after the
agent's reply to the first starts playing — a scripted barge-in. A virtual player stands in
for the browser's: it plays received audio in real time, honours pause/resume, and answers a
clear with the number of samples it dropped, exactly as the page does.
"""

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import websockets

from murmur.audio.frames import FRAME_MS, FRAME_SAMPLES, INPUT_SAMPLE_RATE
from murmur.config import get_settings

# Long enough for endpointing to fire and the reply to finish before the next clip.
TRAILING_SILENCE_S = 1.0
# A turn's metrics arrive after its audio has played, and a long reply can be a minute of
# audio with nothing on the wire while the browser plays out the buffer.
REPLY_TIMEOUT_S = 120.0
IDLE_S = 2.0


class VirtualPlayer:
    """The browser playback worklet, as arithmetic: a queue drained at the sample rate."""

    def __init__(self, rate: int) -> None:
        self._rate = rate
        self._queued = 0.0
        self._paused = False
        self._last = time.monotonic()

    def _advance(self) -> None:
        now = time.monotonic()
        if not self._paused:
            self._queued = max(0.0, self._queued - (now - self._last) * self._rate)
        self._last = now

    @property
    def remaining_s(self) -> float:
        self._advance()
        return self._queued / self._rate

    def receive(self, samples: int) -> None:
        self._advance()
        self._queued += samples

    def pause(self) -> None:
        self._advance()
        self._paused = True

    def resume(self) -> None:
        self._advance()
        self._paused = False

    def clear(self) -> int:
        self._advance()
        dropped, self._queued, self._paused = int(self._queued), 0.0, False
        return dropped


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


async def play(ws: Any, pcm: bytes, delay_s: float = 0.0) -> None:
    await asyncio.sleep(delay_s)
    frame_bytes = FRAME_SAMPLES * 2
    for i in range(0, len(pcm), frame_bytes):
        await ws.send(pcm[i : i + frame_bytes])
        # Real-time pacing: endpointing counts frames, but the server's clock should see real gaps.
        await asyncio.sleep(FRAME_MS / 1000)


async def run_clip(
    ws: Any,
    player: VirtualPlayer,
    pcm: bytes,
    interrupt: tuple[bytes, float] | None = None,
) -> list[dict[str, Any]]:
    """Returns every turn record the clip produced; pauses can split one clip into several."""
    senders = [asyncio.create_task(play(ws, pcm))]
    turns: list[dict[str, Any]] = []
    reply: list[str] = []
    audio_bytes = 0
    in_flight = 0
    while True:
        # Once everything is sent and no turn is running, quiet for the rest of the playback plus
        # a margin means done.
        idle = all(s.done() for s in senders) and in_flight == 0
        wait_s = IDLE_S + player.remaining_s if idle else REPLY_TIMEOUT_S
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=wait_s)
        except TimeoutError:
            if idle:
                return turns
            raise
        if isinstance(message, bytes):
            if audio_bytes == 0 and interrupt is not None:
                cut_pcm, after_s = interrupt
                senders.append(asyncio.create_task(play(ws, cut_pcm, after_s)))
                print(f"  (interrupting {after_s}s into the reply)")
                interrupt = None
            audio_bytes += len(message)
            player.receive(len(message) // 2)
            continue
        event = json.loads(message)
        kind = event["type"]
        if kind == "user":
            in_flight += 1
            print(f"  heard:   {event['text']}")
        elif kind == "assistant_delta":
            reply.append(event["text"])
        elif kind == "pause":
            player.pause()
        elif kind == "resume":
            player.resume()
        elif kind == "clear":
            dropped = player.clear()
            await ws.send(json.dumps({"type": "playback", "discarded_samples": dropped}))
        elif kind == "error":
            raise RuntimeError(event["message"])
        elif kind == "metrics":
            in_flight -= 1
            print(f"  replied: {''.join(reply).strip()}")
            print(f"  audio:   {audio_bytes / 2 / 24_000:.1f}s sent")
            print(f"  ms:      {event['ms']}")
            if event["barge_in"]:
                print(f"  barge-in: heard {event['spoken_chars']} chars before the cut")
            turns.append(event)
            reply, audio_bytes = [], 0


async def main(paths: list[Path], interrupt_after: float | None) -> int:
    settings = get_settings()
    async with websockets.connect(f"ws://{settings.host}:{settings.port}/ws") as ws:
        config = json.loads(await ws.recv())
        assert config["type"] == "config", config
        player = VirtualPlayer(config["output_sample_rate"])
        if interrupt_after is None:
            for path in paths:
                print(path.name)
                turns = await run_clip(ws, player, load_clip(path))
                if len(turns) > 1:
                    print(f"  split into {len(turns)} turns")
            return 0
        if len(paths) % 2:
            raise SystemExit("--interrupt-after takes clips in pairs: ask, interruption")
        for ask, cut in zip(paths[::2], paths[1::2], strict=True):
            print(f"{ask.name} -> {cut.name}")
            await run_clip(ws, player, load_clip(ask), (load_clip(cut), interrupt_after))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--interrupt-after", type=float, default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.clips, args.interrupt_after)))
