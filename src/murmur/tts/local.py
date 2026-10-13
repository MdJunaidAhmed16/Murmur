"""Kokoro (82M) over ONNX. Chosen over Piper for voice quality; see docs/03 decisions."""

from collections.abc import AsyncIterator
from pathlib import Path

from kokoro_onnx import Kokoro

from murmur.audio.frames import float_to_pcm16
from murmur.tts.base import TextToSpeech


class KokoroTTS(TextToSpeech):
    sample_rate = 24_000

    def __init__(self, models_dir: Path, voice: str, speed: float) -> None:
        self._kokoro = Kokoro(
            str(models_dir / "kokoro-v1.0.onnx"), str(models_dir / "voices-v1.0.bin")
        )
        self._voice = voice
        self._speed = speed

    def warm_up(self) -> None:
        self._kokoro.create("Warming up.", voice=self._voice, speed=self._speed, lang="en-us")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        stream = self._kokoro.create_stream(
            text, voice=self._voice, speed=self._speed, lang="en-us"
        )
        async for samples, sample_rate in stream:
            if sample_rate != self.sample_rate:
                raise RuntimeError(f"Kokoro returned {sample_rate} Hz, expected {self.sample_rate}")
            yield float_to_pcm16(samples)
