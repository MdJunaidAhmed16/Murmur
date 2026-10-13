"""Stand-ins for the model-backed stages so the pipeline runs without a GPU or microphone."""

import asyncio
import hashlib
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from murmur.audio.frames import FRAME_SAMPLES, Float32Audio
from murmur.llm.base import LLM, ChatMessage, ToolCall, ToolSpec
from murmur.memory import MemoryService
from murmur.memory.embed import Vector
from murmur.memory.store import MemoryStore
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
    """Streams `reply` word by word. `scripted` overrides it per call, in order."""

    backend_name = "fake"

    def __init__(
        self,
        reply: str = "ok",
        delay: float = 0.0,
        scripted: list[list[str | ToolCall]] | None = None,
        completions: list[str] | None = None,
    ) -> None:
        self._reply = reply
        self._delay = delay
        self._scripted = list(scripted or [])
        self._completions = list(completions or [])
        self.prompts: list[list[ChatMessage]] = []
        self.tools_offered: list[list[ToolSpec] | None] = []
        self.completion_prompts: list[list[ChatMessage]] = []

    async def stream(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[str | ToolCall]:
        self.prompts.append(list(messages))
        self.tools_offered.append(tools)
        events: list[str | ToolCall] = (
            self._scripted.pop(0) if self._scripted else [w + " " for w in self._reply.split(" ")]
        )
        for event in events:
            await asyncio.sleep(self._delay)
            yield event

    async def complete(self, messages: list[ChatMessage], max_tokens: int) -> str:
        self.completion_prompts.append(list(messages))
        return self._completions.pop(0) if self._completions else "NONE"


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


class WordEmbedder:
    """Bag-of-words hashed into a vector: shared words mean high cosine, deterministically."""

    dim = 256

    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s

    def _embed(self, text: str) -> Vector:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def embed_query(self, text: str) -> Vector:
        if self._delay_s:
            time.sleep(self._delay_s)
        return self._embed(text)

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        return [self._embed(t) for t in texts]


def make_memory(path: Path, embedder: WordEmbedder | None = None) -> MemoryService:
    return MemoryService(MemoryStore(path), embedder or WordEmbedder(), min_similarity=0.3)


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
