"""Wires VAD → STT → agent → TTS for one session and owns cancellation.

Each turn is one task holding one TaskGroup (generation + synthesis). Barge-in cancels that
task, which cancels everything in it as a unit.
"""

import asyncio
import contextlib
from collections import deque
from typing import Any, Protocol

import structlog

from murmur.agent.loop import Agent
from murmur.audio.frames import FRAME_MS, Float32Audio, pcm16_to_float
from murmur.audio.vad import EndpointDetector, VadEvent
from murmur.config import Settings
from murmur.pipeline.metrics import TurnLog, TurnTimer, now_ms
from murmur.stt.base import SpeechToText, TranscriptionStream
from murmur.tts.base import TextToSpeech
from murmur.tts.text import SentenceChunker

log = structlog.get_logger()

# Extra audio kept before SPEECH_START fires, so the first syllable reaches STT.
_PREROLL_EXTRA_MS = 200


class SpeechDetector(Protocol):
    def speech_probability(self, frame: Float32Audio) -> float: ...


class Outbound(Protocol):
    async def audio(self, pcm: bytes) -> None: ...

    async def event(self, payload: dict[str, Any]) -> None: ...


class Session:
    def __init__(
        self,
        *,
        settings: Settings,
        vad: SpeechDetector,
        stt: SpeechToText,
        tts: TextToSpeech,
        agent: Agent,
        backend_name: str,
        outbound: Outbound,
        turn_log: TurnLog,
    ) -> None:
        self._settings = settings
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._agent = agent
        self._backend_name = backend_name
        self._out = outbound
        self._turn_log = turn_log
        self._detector = EndpointDetector(settings.endpoint_silence_ms, settings.min_speech_ms)
        self._preroll: deque[Float32Audio] = deque(
            maxlen=(settings.min_speech_ms + _PREROLL_EXTRA_MS) // FRAME_MS
        )
        self._stream: TranscriptionStream | None = None
        self._partials: asyncio.Task[None] | None = None
        self._turn: asyncio.Task[None] | None = None
        self._last_speech_at = 0.0
        self._audio_since_clear = False

    async def handle_frame(self, data: bytes) -> None:
        frame = pcm16_to_float(data)
        is_speech = self._vad.speech_probability(frame) >= self._settings.vad_threshold
        if is_speech:
            self._last_speech_at = now_ms()
        event = self._detector.update(is_speech)

        if self._stream is not None:
            self._stream.feed(frame)
        else:
            self._preroll.append(frame)

        if event is VadEvent.SPEECH_START:
            await self._on_speech_start()
        elif event is VadEvent.ENDPOINT:
            self._on_endpoint()

    @property
    def agent_speaking(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def _on_speech_start(self) -> None:
        if self._settings.barge_in:
            await self.interrupt()
        elif self.agent_speaking:
            return
        stream = self._stt.stream()
        for frame in self._preroll:
            stream.feed(frame)
        self._preroll.clear()
        self._stream = stream
        self._partials = asyncio.create_task(self._emit_partials(stream))
        await self._out.event({"type": "speech_start"})

    def _on_endpoint(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        if self._partials is not None:
            self._partials.cancel()
        timer = TurnTimer(backend=self._backend_name)
        timer.mark("speech_end", self._last_speech_at)
        timer.mark("endpoint")
        self._turn = asyncio.create_task(self._run_turn(stream, timer))

    async def interrupt(self) -> None:
        # Clear even when no turn is running: audio is sent faster than real time, so the browser
        # can still be playing a reply whose turn finished long ago.
        turn = self._turn
        if turn is not None and not turn.done():
            turn.cancel()
        if self._audio_since_clear:
            self._audio_since_clear = False
            await self._out.event({"type": "clear"})
        if turn is not None:
            await asyncio.wait([turn])

    async def _emit_partials(self, stream: TranscriptionStream) -> None:
        interval = self._settings.stt_partial_interval_ms / 1000
        if interval <= 0:
            return
        while True:
            await asyncio.sleep(interval)
            if text := await stream.partial():
                await self._out.event({"type": "partial", "text": text})

    async def _run_turn(self, stream: TranscriptionStream, timer: TurnTimer) -> None:
        spoken: list[str] = []
        text = ""
        try:
            text = await stream.finalize()
            timer.mark("stt_final")
            if not text:
                return
            await self._out.event({"type": "user", "text": text})
            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._generate(text, timer, sentences))
                tg.create_task(self._speak(timer, sentences, spoken))
        except asyncio.CancelledError:
            timer.barge_in = True
            raise
        except Exception as exc:
            # A failed turn (vLLM down, TTS error) must not kill the session.
            log.exception("turn_failed")
            await self._out.event({"type": "error", "message": str(exc)})
        finally:
            if text:
                self._finish_turn(timer, " ".join(spoken))

    def _finish_turn(self, timer: TurnTimer, spoken_text: str) -> None:
        self._agent.commit_reply(spoken_text)
        timer.spoken_chars = len(spoken_text)
        record = self._turn_log.write(timer)
        log.info("turn", **record)
        # Fire-and-forget: this runs during cancellation, where awaiting is unreliable.
        task = asyncio.get_running_loop().create_task(
            self._out.event({"type": "metrics", **record})
        )
        task.add_done_callback(_ignore_result)

    async def _generate(
        self, text: str, timer: TurnTimer, sentences: asyncio.Queue[str | None]
    ) -> None:
        chunker = SentenceChunker()
        reply: list[str] = []
        timer.mark("llm_request")
        async for delta in self._agent.respond(text):
            timer.mark("llm_first_token")
            reply.append(delta)
            await self._out.event({"type": "assistant_delta", "text": delta})
            if self._settings.tts_handoff == "sentence":
                for sentence in chunker.feed(delta):
                    await sentences.put(sentence)
        rest = chunker.flush() if self._settings.tts_handoff == "sentence" else "".join(reply)
        if rest.strip():
            await sentences.put(rest.strip())
        await sentences.put(None)

    async def _speak(
        self, timer: TurnTimer, sentences: asyncio.Queue[str | None], spoken: list[str]
    ) -> None:
        while (sentence := await sentences.get()) is not None:
            timer.mark("tts_request")
            async for pcm in self._tts.synthesize(sentence):
                timer.mark("tts_first_audio")
                self._audio_since_clear = True
                await self._out.audio(pcm)
            # Sent, not yet heard: the browser may still be buffering it. Tightened on day 7.
            spoken.append(sentence)

    async def close(self) -> None:
        for task in (self._partials, self._turn):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


def _ignore_result(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
