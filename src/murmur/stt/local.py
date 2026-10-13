"""faster-whisper. Not natively streaming: partials re-transcribe the utterance so far."""

import asyncio
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from murmur.audio.frames import Float32Audio
from murmur.stt.base import SpeechToText, TranscriptionStream


class WhisperSTT(SpeechToText):
    def __init__(self, model: str, device: str, compute_type: str, download_root: Path) -> None:
        self._model = WhisperModel(
            model, device=device, compute_type=compute_type, download_root=str(download_root)
        )

    def transcribe(self, audio: Float32Audio) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            # Each call sees one utterance; the VAD upstream already trimmed silence.
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def warm_up(self) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def stream(self) -> TranscriptionStream:
        return _WhisperStream(self)


class _WhisperStream(TranscriptionStream):
    def __init__(self, stt: WhisperSTT) -> None:
        self._stt = stt
        self._frames: list[Float32Audio] = []

    def feed(self, frame: Float32Audio) -> None:
        self._frames.append(frame)

    def _audio(self) -> Float32Audio:
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)

    async def partial(self) -> str:
        return await asyncio.to_thread(self._stt.transcribe, self._audio())

    async def finalize(self) -> str:
        # A partial still running in its thread delays this one; that cost lands in stt_final.
        return await asyncio.to_thread(self._stt.transcribe, self._audio())
