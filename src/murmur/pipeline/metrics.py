"""Per-stage turn timing. The only place in the codebase that records time."""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Mark = Literal[
    "speech_end",
    "endpoint",
    "stt_final",
    "memory_recall_start",
    "memory_recall",
    "llm_request",
    "llm_first_token",
    "tts_request",
    "tts_first_audio",
]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class TurnTimer:
    backend: str
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    marks: dict[str, float] = field(default_factory=dict)
    barge_in: bool = False
    tool_calls: list[str] = field(default_factory=list)
    memories_retrieved: int = 0
    spoken_chars: int = 0
    user_text: str = ""

    def mark(self, name: Mark, at: float | None = None) -> None:
        # First mark wins: later chunks of the same stage must not move "first audio".
        self.marks.setdefault(name, now_ms() if at is None else at)

    def _span(self, start: str, end: str) -> int | None:
        if start in self.marks and end in self.marks:
            return round(self.marks[end] - self.marks[start])
        return None

    def stage_ms(self) -> dict[str, int | None]:
        return {
            "endpoint": self._span("speech_end", "endpoint"),
            "stt_final": self._span("endpoint", "stt_final"),
            "memory_recall": self._span("memory_recall_start", "memory_recall"),
            "llm_first_token": self._span("llm_request", "llm_first_token"),
            # Generating up to the first sentence boundary; TTS cannot start before it.
            "llm_first_sentence": self._span("llm_first_token", "tts_request"),
            "tts_first_audio": self._span("tts_request", "tts_first_audio"),
            # Measured from the user's last speech frame, so endpointing counts against us.
            "time_to_first_audio": self._span("speech_end", "tts_first_audio"),
        }

    def record(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "ms": self.stage_ms(),
            "barge_in": self.barge_in,
            "tool_calls": self.tool_calls,
            "memories_retrieved": self.memories_retrieved,
            "backend": self.backend,
            "spoken_chars": self.spoken_chars,
        }


class TurnLog:
    """Appends one JSON line per turn to results/turns.jsonl."""

    def __init__(self, results_dir: Path) -> None:
        self._path = results_dir / "turns.jsonl"

    def write(self, timer: TurnTimer) -> dict[str, Any]:
        record = timer.record()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record
