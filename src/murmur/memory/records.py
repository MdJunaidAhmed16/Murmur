"""The memory record."""

from dataclasses import dataclass
from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    created_at: datetime
    last_used_at: datetime | None
    use_count: int
    source_turn: str | None
    superseded_by: str | None
    # Set by decay: kept, but retrieval stops surfacing it ahead of fresher facts.
    low_priority: bool

    @property
    def active(self) -> bool:
        return self.superseded_by is None
