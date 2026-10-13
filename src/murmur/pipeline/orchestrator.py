"""Wires VAD → STT → recall → agent → TTS for one session and owns cancellation.

Each turn is one task holding one TaskGroup (generation + synthesis). Barge-in cancels that
task, which cancels everything in it as a unit. Memory writes are handed to the background
worker after the turn; nothing here waits on them.
"""

import asyncio
import contextlib
from collections import deque
from typing import Any, Protocol

import structlog

from murmur.agent.loop import Agent
from murmur.agent.memory_jobs import MemoryJobs, TurnForMemory
from murmur.audio.frames import FRAME_MS, Float32Audio, pcm16_to_float
from murmur.audio.vad import EndpointDetector, VadEvent
from murmur.config import Settings
from murmur.memory import Memory, MemoryService
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
        memory: MemoryService,
        memory_jobs: MemoryJobs,
        backend_name: str,
        outbound: Outbound,
        turn_log: TurnLog,
    ) -> None:
        self._settings = settings
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._agent = agent
        self._memory = memory
        self._jobs = memory_jobs
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
        # The loop holds tasks weakly; an unreferenced fire-and-forget task can be collected
        # before it runs. This set keeps them alive until done.
        self._background: set[asyncio.Task[None]] = set()
        # User words from a turn cut off before the agent said anything (see _finish_turn).
        self._unanswered = ""

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
        self._update_voice_idle()

    @property
    def agent_speaking(self) -> bool:
        return self._turn is not None and not self._turn.done()

    def _update_voice_idle(self, _: object = None) -> None:
        # Background memory work holds off while the user talks or a turn runs.
        if self._detector.in_speech or self.agent_speaking:
            self._jobs.voice_idle.clear()
        else:
            self._jobs.voice_idle.set()

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
        self._turn.add_done_callback(self._update_voice_idle)

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
        recalled: list[Memory] = []
        previous_reply = self._agent.last_reply()
        text = ""
        try:
            heard = await stream.finalize()
            timer.mark("stt_final")
            if not heard:
                return
            text, self._unanswered = f"{self._unanswered} {heard}".strip(), ""
            await self._out.event({"type": "user", "text": text})
            recalled = await self._recall(text, timer)
            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._generate(text, recalled, timer, sentences))
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
                self._finish_turn(timer, text, previous_reply, recalled, " ".join(spoken))

    async def _recall(self, text: str, timer: TurnTimer) -> list[Memory]:
        timer.mark("memory_recall_start")
        try:
            recalled = await asyncio.wait_for(
                asyncio.to_thread(self._memory.recall, text, self._settings.recall_k),
                timeout=self._settings.recall_budget_ms / 1000,
            )
        except TimeoutError:
            # Over budget: answer without memories rather than make the user wait.
            log.warning("recall_over_budget", budget_ms=self._settings.recall_budget_ms)
            recalled = []
        timer.mark("memory_recall")
        timer.memories_retrieved = len(recalled)
        if recalled:
            await self._out.event({"type": "recalled", "memories": [m.text for m in recalled]})
        return recalled

    def _finish_turn(
        self,
        timer: TurnTimer,
        user_text: str,
        previous_reply: str,
        recalled: list[Memory],
        spoken_text: str,
    ) -> None:
        timer.spoken_chars = len(spoken_text)
        timer.tool_calls = list(self._agent.last_tool_calls)
        record = self._turn_log.write(timer)
        log.info("turn", **record)
        if timer.barge_in and not spoken_text and not timer.tool_calls:
            # Cut off before saying anything: the user was not interrupting, they were still
            # talking and endpointing fired early. Carry their words into the next turn, so
            # "please forget what I told you, ... about my sister" stays one request.
            self._agent.retract_unanswered()
            self._unanswered = user_text
        else:
            self._agent.commit_reply(spoken_text)
            self._jobs.submit(
                TurnForMemory(
                    turn_id=timer.turn_id,
                    user_text=user_text,
                    previous_reply=previous_reply,
                    recalled_ids=tuple(m.id for m in recalled),
                    skip_extraction="forget" in timer.tool_calls,
                )
            )
        # Fire-and-forget: this runs during cancellation, where awaiting is unreliable.
        task = asyncio.get_running_loop().create_task(
            self._out.event({"type": "metrics", **record})
        )
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_ignore_result)

    async def _generate(
        self,
        text: str,
        recalled: list[Memory],
        timer: TurnTimer,
        sentences: asyncio.Queue[str | None],
    ) -> None:
        chunker = SentenceChunker()
        reply: list[str] = []
        timer.mark("llm_request")
        async for delta in self._agent.respond(text, [m.text for m in recalled]):
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
        self._jobs.voice_idle.set()


def _ignore_result(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
