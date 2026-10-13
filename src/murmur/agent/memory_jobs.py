"""Memory work that needs the model: extraction and consolidation. Never on the turn path.

One worker drains a queue in order. Before each model call it waits for the voice loop to go
quiet, so a background extraction does not share the GPU with a live turn.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import timedelta

import structlog

from murmur.agent.grounding import shares_content
from murmur.agent.prompts import (
    EXTRACT_EXAMPLES,
    EXTRACT_PROMPT,
    JUDGE_EXAMPLES,
    JUDGE_PROMPT,
    extraction_input,
    judge_input,
)
from murmur.config import Settings
from murmur.llm.base import LLM, ChatMessage
from murmur.memory import Memory, MemoryService

log = structlog.get_logger()

_MAX_FACTS_PER_TURN = 5
_VERDICT = re.compile(r"\b(same|replaced|keep)\b", re.IGNORECASE)
# Pairs judged per new fact, most similar first. Each is one short background call.
_MAX_JUDGED = 3
_PLACEHOLDER = re.compile(r"[\[\]<>{}]|\bunknown\b|\bnot specified\b", re.IGNORECASE)


@dataclass(frozen=True)
class TurnForMemory:
    turn_id: str
    user_text: str
    previous_reply: str
    recalled_ids: tuple[str, ...]
    # A turn that asked to forget must not write what it asked to forget.
    skip_extraction: bool


# Subject and possessive only: "me" mostly appears in requests ("tell me", "remind me").
_SELF = re.compile(r"\b(i|i'm|im|i've|i'd|i'll|my|mine|myself|we|our)\b", re.IGNORECASE)
_ASKED_ABOUT_USER = re.compile(r"\b(you|your|yours)\b[^.!]*\?", re.IGNORECASE)


def worth_extracting(user_text: str, previous_reply: str) -> bool:
    """Facts about the user come in the first person, or as answers to a question about them.

    "What's the capital of Australia?" is neither, and a small model asked anyway will store
    the answer as if it were about the user.
    """
    return bool(_SELF.search(user_text) or _ASKED_ABOUT_USER.search(previous_reply))


def is_grounded(fact: str, source: str) -> bool:
    """A fact must share a content word with what was said; small models invent the rest."""
    return shares_content(fact, source)


def with_examples(system: str, examples: list[tuple[str, str]], query: str) -> list[ChatMessage]:
    messages = [ChatMessage("system", system)]
    for user, assistant in examples:
        messages += [ChatMessage("user", user), ChatMessage("assistant", assistant)]
    return [*messages, ChatMessage("user", query)]


def parse_facts(text: str) -> list[str]:
    if text.strip().upper().startswith("NONE"):
        return []
    facts = [line.strip()[2:].strip() for line in text.splitlines() if line.strip()[:2] == "- "]
    # "Sibling named [Sister's Name]": a template slot the model filled from a question.
    return [f for f in facts if f and not _PLACEHOLDER.search(f)][:_MAX_FACTS_PER_TURN]


def parse_verdict(text: str) -> str:
    # Anything unparseable is "keep": a missed supersede is cheaper than deleting a true fact.
    match = _VERDICT.search(text)
    return match.group(1).lower() if match else "keep"


class MemoryJobs:
    def __init__(
        self,
        llm: LLM,
        memory: MemoryService,
        *,
        duplicate_threshold: float,
        related_threshold: float,
        stale_after: timedelta,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._duplicate_threshold = duplicate_threshold
        self._related_threshold = related_threshold
        self._stale_after = stale_after
        self._queue: asyncio.Queue[TurnForMemory] = asyncio.Queue()
        # Set by the orchestrator whenever nobody is speaking and no turn is running.
        self.voice_idle = asyncio.Event()
        self.voice_idle.set()
        self._urgent = asyncio.Event()

    @classmethod
    def from_settings(cls, llm: LLM, memory: MemoryService, settings: Settings) -> "MemoryJobs":
        return cls(
            llm,
            memory,
            duplicate_threshold=settings.memory_duplicate_threshold,
            related_threshold=settings.memory_related_threshold,
            stale_after=timedelta(days=settings.memory_stale_days),
        )

    def submit(self, turn: TurnForMemory) -> None:
        self._queue.put_nowait(turn)

    async def flush(self) -> None:
        """Process everything queued now, without waiting for quiet. Used by forget."""
        self._urgent.set()
        try:
            await self._queue.join()
        finally:
            self._urgent.clear()

    async def run(self) -> None:
        while True:
            turn = await self._queue.get()
            try:
                await self._process(turn)
            except Exception:
                # One bad extraction must not stop memory for the rest of the session.
                log.exception("memory_job_failed", turn_id=turn.turn_id)
            finally:
                self._queue.task_done()

    async def _quiet(self) -> None:
        if self.voice_idle.is_set() or self._urgent.is_set():
            return
        waiters = [
            asyncio.create_task(self.voice_idle.wait()),
            asyncio.create_task(self._urgent.wait()),
        ]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()

    async def _process(self, turn: TurnForMemory) -> None:
        if turn.recalled_ids:
            await asyncio.to_thread(self._memory.mark_used, list(turn.recalled_ids))
        if turn.skip_extraction:
            return
        facts = await self.extract(turn.user_text, turn.previous_reply)
        added = await asyncio.to_thread(self._memory.add, facts, turn.turn_id)
        for memory in added:
            await self.consolidate(memory)
        if added:
            log.info("memory_written", turn_id=turn.turn_id, facts=[m.text for m in added])

    async def extract(self, user_text: str, previous_reply: str) -> list[str]:
        if not worth_extracting(user_text, previous_reply):
            return []
        await self._quiet()
        reply = await self._llm.complete(
            with_examples(
                EXTRACT_PROMPT, EXTRACT_EXAMPLES, extraction_input(user_text, previous_reply)
            ),
            max_tokens=150,
        )
        facts = parse_facts(reply)
        # Grounded in the user's words, or in the question they were answering ("Afifa." to
        # "what is your sister called?").
        grounded = [f for f in facts if is_grounded(f, f"{user_text} {previous_reply}")]
        if len(grounded) < len(facts):
            log.info("memory_ungrounded", dropped=[f for f in facts if f not in grounded])
        return grounded

    async def consolidate(self, memory: Memory) -> None:
        """Merge into a near-duplicate, or supersede the older facts this one contradicts."""
        older = [
            (m, sim)
            for m, sim in self._memory.neighbours(memory.id, self._related_threshold)
            if m.created_at < memory.created_at
        ]
        if not older:
            return
        twin = next((m for m, sim in older if sim >= self._duplicate_threshold), None)
        if twin is not None:
            await self._merge(keep=twin, drop=memory)
            return

        verdicts = [
            (old, await self.judge(old.text, memory.text)) for old, _ in older[:_MAX_JUDGED]
        ]
        # Merge before superseding: deleting a duplicate also deletes what it superseded.
        same = next((old for old, verdict in verdicts if verdict == "same"), None)
        if same is not None:
            await self._merge(keep=same, drop=memory)
            return
        for old, verdict in verdicts:
            if verdict == "replaced":
                await asyncio.to_thread(self._memory.supersede, old.id, memory.id)
                log.info("memory_superseded", old=old.text, new=memory.text)

    async def judge(self, old_fact: str, new_fact: str) -> str:
        await self._quiet()
        reply = await self._llm.complete(
            with_examples(JUDGE_PROMPT, JUDGE_EXAMPLES, judge_input(old_fact, new_fact)),
            max_tokens=4,
        )
        return parse_verdict(reply)

    async def _merge(self, keep: Memory, drop: Memory) -> None:
        # The older record survives; saying it again counts as a use.
        await asyncio.to_thread(self._memory.delete, [drop.id])
        await asyncio.to_thread(self._memory.mark_used, [keep.id])
        log.info("memory_merged", kept=keep.text, dropped=drop.text)

    async def consolidate_all(self) -> None:
        """Batch pass over the whole store, oldest first, then decay."""
        for memory in sorted(
            self._memory.all(include_superseded=False), key=lambda m: m.created_at
        ):
            # Earlier iterations may have merged or superseded this one already.
            current = self._memory.get(memory.id)
            if current is not None and current.active:
                await self.consolidate(current)
        await self.decay()

    async def decay(self) -> int:
        return await asyncio.to_thread(self._memory.decay, self._stale_after)
