"""Kokoro (82M) over ONNX. Chosen over Piper for voice quality; see docs/03 decisions."""

from collections.abc import AsyncIterator
from pathlib import Path

import onnxruntime as ort
from kokoro_onnx import Kokoro

from murmur.audio.frames import float_to_pcm16
from murmur.tts.base import TextToSpeech

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def _session(model_path: Path, device: str) -> ort.InferenceSession:
    if device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            # Fail loudly: silently falling back to CPU costs ~1.5 s per turn (R1).
            raise RuntimeError(
                "MURMUR_TTS_DEVICE=cuda but this onnxruntime has no CUDA provider; "
                "run `make setup` to install onnxruntime-gpu"
            )
        # onnxruntime-gpu loads CUDA/cuDNN from the nvidia-* pip wheels when asked to.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    return ort.InferenceSession(str(model_path), providers=_PROVIDERS[device])


class KokoroTTS(TextToSpeech):
    sample_rate = 24_000

    def __init__(self, models_dir: Path, voice: str, speed: float, device: str) -> None:
        session = _session(models_dir / "kokoro-v1.0.onnx", device)
        self._kokoro = Kokoro.from_session(session, str(models_dir / "voices-v1.0.bin"))
        self.providers: list[str] = session.get_providers()
        self._voice = voice
        self._speed = speed

    def warm_up(self) -> None:
        # Twice: the first CUDA run also pays for kernel selection.
        for _ in range(2):
            self._kokoro.create("Warming up.", voice=self._voice, speed=self._speed, lang="en-us")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        stream = self._kokoro.create_stream(
            text, voice=self._voice, speed=self._speed, lang="en-us"
        )
        async for samples, sample_rate in stream:
            if sample_rate != self.sample_rate:
                raise RuntimeError(f"Kokoro returned {sample_rate} Hz, expected {self.sample_rate}")
            yield float_to_pcm16(samples)
