"""Wires VAD → STT → recall → agent → TTS for one session and owns cancellation.

Each turn is one task holding one TaskGroup (generation + synthesis), and it stays alive until
its audio has finished playing: a reply the user can still hear is a reply they can cut off.
Barge-in cancels that task, which cancels everything in it as a unit. A separate settle step
then works out what the user actually heard and only that goes into history and memory.
"""

import asyncio
import contextlib
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from murmur.agent.loop import Agent
from murmur.agent.memory_jobs import MemoryJobs, TurnForMemory
from murmur.audio.frames import FRAME_MS, Float32Audio, pcm16_to_float
from murmur.audio.vad import EndpointDetector, VadEvent
from murmur.config import Settings
from murmur.memory import Memory, MemoryService
from murmur.pipeline.metrics import TurnLog, TurnTimer, now_ms
from murmur.pipeline.playback import PlaybackLedger
from murmur.stt.base import SpeechToText, TranscriptionStream
from murmur.tts.base import TextToSpeech
from murmur.tts.text import SentenceChunker

log = structlog.get_logger()

# Extra audio kept before SPEECH_START fires, so the first syllable reaches STT.
_PREROLL_EXTRA_MS = 200
# Network and browser buffering between "sent" and "heard", added to the playback estimate.
_PLAYBACK_MARGIN_MS = 150
# How long a cut-off turn waits for the browser to say how much audio it dropped.
_PLAYBACK_REPORT_TIMEOUT_S = 0.3


class SpeechDetector(Protocol):
    def speech_probability(self, frame: Float32Audio) -> float: ...


class Outbound(Protocol):
    async def audio(self, pcm: bytes) -> None: ...

    async def event(self, payload: dict[str, Any]) -> None: ...


@dataclass
class _Cut:
    """Where playback stood when the user barged in."""

    sent_until: int  # session samples sent when the clear went out
    at_ms: float
    report: asyncio.Future[int] | None  # samples the browser discarded, if it says


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
        self._detector = EndpointDetector(
            settings.endpoint_silence_ms, settings.min_speech_ms, settings.barge_in_onset_ms
        )
        self._preroll: deque[Float32Audio] = deque(
            maxlen=(settings.min_speech_ms + _PREROLL_EXTRA_MS) // FRAME_MS
        )
        self._stream: TranscriptionStream | None = None
        self._partials: asyncio.Task[None] | None = None
        self._turn: asyncio.Task[None] | None = None
        self._turn_timer: TurnTimer | None = None
        self._last_speech_at = 0.0
        self._was_speech = False
        self._speech_run_started_at = 0.0
        self._audio_since_clear = False
        self._paused = False
        self._samples_sent = 0
        self._cut: _Cut | None = None
        # The browser's answer to the latest clear; outlives _cut, which the turn takes over.
        self._report: asyncio.Future[int] | None = None
        # The loop holds tasks weakly; an unreferenced fire-and-forget task can be collected
        # before it runs. This set keeps them alive until done.
        self._background: set[asyncio.Task[None]] = set()
        self._settling: asyncio.Task[None] | None = None
        # User words from a turn cut off before the agent said anything (see _settle).
        self._unanswered = ""

    async def handle_frame(self, data: bytes) -> None:
        frame = pcm16_to_float(data)
        is_speech = self._vad.speech_probability(frame) >= self._settings.vad_threshold
        if is_speech:
            self._last_speech_at = now_ms()
            if not self._was_speech:
                self._speech_run_started_at = self._last_speech_at
        self._was_speech = is_speech
        event = self._detector.update(is_speech)

        if self._stream is not None:
            self._stream.feed(frame)
        else:
            self._preroll.append(frame)

        if event is VadEvent.ONSET:
            await self._on_onset()
        elif event is VadEvent.ABORT:
            await self._on_abort()
        elif event is VadEvent.SPEECH_START:
            await self._on_speech_start()
        elif event is VadEvent.ENDPOINT:
            self._on_endpoint()
        self._update_voice_idle()

    async def handle_control(self, payload: dict[str, Any]) -> None:
        report = self._report
        if payload.get("type") == "playback" and report is not None and not report.done():
            report.set_result(int(payload.get("discarded_samples", 0)))

    @property
    def agent_speaking(self) -> bool:
        return self._turn is not None and not self._turn.done()

    def _update_voice_idle(self, _: object = None) -> None:
        # Background memory work holds off while the user talks or a turn runs.
        if self._detector.in_speech or self.agent_speaking:
            self._jobs.voice_idle.clear()
        else:
            self._jobs.voice_idle.set()

    async def _on_onset(self) -> None:
        # Stop the agent talking over the user now; whether it is a real interruption is decided
        # at SPEECH_START. Pausing keeps the queue, so a cough costs a short gap, not the reply.
        if not (self._settings.barge_in and self.agent_speaking and self._audio_since_clear):
            return
        self._paused = True
        self._mark_barge_in()
        await self._out.event({"type": "pause"})

    async def _on_abort(self) -> None:
        if not self._paused:
            return
        self._paused = False
        if self._turn_timer is not None:
            self._turn_timer.unmark("barge_in_onset", "barge_in_cut")
        await self._out.event({"type": "resume"})

    def _mark_barge_in(self) -> None:
        if self._turn_timer is not None and self.agent_speaking:
            self._turn_timer.mark("barge_in_onset", self._speech_run_started_at)
            self._turn_timer.mark("barge_in_cut")

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
        self._turn_timer = timer
        self._turn = asyncio.create_task(self._run_turn(stream, timer))
        self._turn.add_done_callback(self._update_voice_idle)

    async def interrupt(self) -> None:
        turn = self._turn
        live = turn is not None and not turn.done()
        if live:
            self._mark_barge_in()
        clearing = self._audio_since_clear
        self._report = asyncio.get_running_loop().create_future() if clearing else None
        # Recorded before cancelling: the turn's settle step reads it.
        self._cut = _Cut(sent_until=self._samples_sent, at_ms=now_ms(), report=self._report)
        if live and turn is not None:
            turn.cancel()
        self._paused = False
        # Clear even when no turn is running: the browser can still be playing audio past the
        # estimate (underruns stretch playback).
        if clearing:
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
        ledger = PlaybackLedger(self._tts.sample_rate, self._samples_sent)
        recalled: list[Memory] = []
        previous_reply = ""
        text = ""
        try:
            heard = await stream.finalize()
            timer.mark("stt_final")
            if not heard:
                return
            # The previous turn may still be working out what was heard; its outcome decides
            # the history this turn sees and whether its words carry over into this one.
            await self.settled()
            previous_reply = self._agent.last_reply()
            text, self._unanswered = f"{self._unanswered} {heard}".strip(), ""
            await self._out.event({"type": "user", "text": text})
            recalled = await self._recall(text, timer)
            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._generate(text, recalled, timer, sentences))
                tg.create_task(self._speak(timer, sentences, ledger))
            await self._play_out(timer, ledger)
        except asyncio.CancelledError:
            timer.barge_in = True
            raise
        except Exception as exc:
            # A failed turn (vLLM down, TTS error) must not kill the session.
            log.exception("turn_failed")
            await self._out.event({"type": "error", "message": str(exc)})
        finally:
            if text:
                cut, self._cut = (self._cut if timer.barge_in else None), None
                self._settling = self._spawn(
                    self._settle(timer, text, previous_reply, recalled, ledger, cut)
                )

    async def _play_out(self, timer: TurnTimer, ledger: PlaybackLedger) -> None:
        """Wait until the browser has (by estimate) played everything this turn sent."""
        started = timer.marks.get("tts_first_audio")
        if started is None or not ledger.sent_samples:
            return
        ends_at = started + ledger.sent_samples * 1000 / ledger.sample_rate + _PLAYBACK_MARGIN_MS
        await asyncio.sleep(max(0.0, (ends_at - now_ms()) / 1000))

    async def settled(self) -> None:
        if self._settling is not None:
            # Shielded: cancelling the waiter must not cancel the settle it is waiting for.
            await asyncio.shield(self._settling)

    async def _played_until(self, timer: TurnTimer, ledger: PlaybackLedger, cut: _Cut) -> int:
        if cut.report is not None:
            try:
                discarded = await asyncio.wait_for(cut.report, _PLAYBACK_REPORT_TIMEOUT_S)
                return cut.sent_until - discarded
            except TimeoutError:
                log.info("playback_report_missing")  # a client that is not the browser
        started = timer.marks.get("tts_first_audio")
        if started is None:
            return ledger.session_offset
        elapsed = int((cut.at_ms - started) / 1000 * ledger.sample_rate)
        return ledger.session_offset + min(elapsed, ledger.sent_samples)

    async def _settle(
        self,
        timer: TurnTimer,
        user_text: str,
        previous_reply: str,
        recalled: list[Memory],
        ledger: PlaybackLedger,
        cut: _Cut | None,
    ) -> None:
        # Record what was heard, not what was generated: barge-in truncates mid-sentence.
        if cut is None:
            spoken_text = ledger.all_text()
        else:
            spoken_text = ledger.heard_text(await self._played_until(timer, ledger, cut))
        timer.spoken_chars = len(spoken_text)
        timer.tool_calls = list(self._agent.last_tool_calls)
        record = self._turn_log.write(timer)
        log.info("turn", **record)
        if timer.barge_in and not spoken_text and not timer.tool_calls:
            # Cut off before the user heard anything: they were not interrupting, they were
            # still talking and endpointing fired early. Carry their words into the next turn,
            # so "please forget what I told you, ... about my sister" stays one request.
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
        await self._out.event({"type": "metrics", **record})

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_log_failure)
        return task

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

    async def _generate(
        self,
        text: str,
        recalled: list[Memory],
        timer: TurnTimer,
        sentences: asyncio.Queue[str | None],
    ) -> None:
        chunker = SentenceChunker(first_clause=self._settings.tts_first_chunk == "clause")
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
        self, timer: TurnTimer, sentences: asyncio.Queue[str | None], ledger: PlaybackLedger
    ) -> None:
        while (sentence := await sentences.get()) is not None:
            timer.mark("tts_request")
            ledger.begin(sentence)
            async for pcm in self._tts.synthesize(sentence):
                if "tts_first_audio" not in timer.marks:
                    await self._hold_until_committed(timer)
                timer.mark("tts_first_audio")
                self._audio_since_clear = True
                samples = len(pcm) // 2
                ledger.add_audio(samples)
                self._samples_sent += samples
                await self._out.audio(pcm)

    async def _hold_until_committed(self, timer: TurnTimer) -> None:
        # Speaking before endpoint_commit_ms of silence risks talking over a user who only
        # paused. Holding costs nothing when the pipeline is slower than the commit point,
        # and the wait is inside time_to_first_audio, so it is never hidden.
        commit_at = timer.marks["speech_end"] + self._settings.endpoint_commit_ms
        await asyncio.sleep(max(0.0, (commit_at - now_ms()) / 1000))

    async def close(self) -> None:
        for task in (self._partials, self._turn):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        for task in list(self._background):
            task.cancel()
        self._jobs.voice_idle.set()


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        log.error("background_task_failed", error=repr(exc))
