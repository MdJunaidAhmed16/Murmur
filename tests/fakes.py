"""Stand-ins for the model-backed stages so the pipeline runs without a GPU or microphone."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from murmur.audio.frames import FRAME_SAMPLES, Float32Audio
from murmur.llm.base import LLM, ChatMessage
from murmur.stt.base import SpeechToText, TranscriptionStream
from murmur.tts.base import TextToSpeech

SPEECH_FRAME = (np.full(FRAME_SAMPLES, 8000, dtype=np.int16)).tobytes()
SILENCE_FRAME = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()


class LoudnessVAD:
    def speech_probability(self, frame: Float32Audio) -> float:
        return 1.0 if float(np.abs(frame).max()) > 0.1 else 0.0


class FakeStream(TranscriptionStream):
    def __init__(self, text: str) -> None:
        self.text = text
        self.frames = 0

    def feed(self, frame: Float32Audio) -> None:
        self.frames += 1

    async def partial(self) -> str:
        return ""

    async def finalize(self) -> str:
        return self.text


class FakeSTT(SpeechToText):
    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)
        self.streams: list[FakeStream] = []

    def stream(self) -> TranscriptionStream:
        stream = FakeStream(next(self._texts))
        self.streams.append(stream)
        return stream


class FakeLLM(LLM):
    backend_name = "fake"

    def __init__(self, reply: str, delay: float = 0.0) -> None:
        self._reply = reply
        self._delay = delay
        self.prompts: list[list[ChatMessage]] = []

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.prompts.append(list(messages))
        for word in self._reply.split(" "):
            await asyncio.sleep(self._delay)
            yield word + " "


class FakeTTS(TextToSpeech):
    sample_rate = 24_000

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay
        self.requests: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.requests.append(text)
        for _ in range(3):
            await asyncio.sleep(self._delay)
            yield b"\x00\x00" * 240


class RecordingOutbound:
    def __init__(self) -> None:
        self.audio_chunks = 0
        self.events: list[dict[str, Any]] = []
        self._audio_sent = asyncio.Condition()

    async def audio(self, pcm: bytes) -> None:
        async with self._audio_sent:
            self.audio_chunks += 1
            self._audio_sent.notify_all()

    async def wait_for_audio(self, chunks: int) -> None:
        async with self._audio_sent:
            await self._audio_sent.wait_for(lambda: self.audio_chunks >= chunks)

    async def event(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == kind]
