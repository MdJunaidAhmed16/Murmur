"""SQLite persistence. One file, no server, survives restarts."""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from murmur.memory.embed import Vector
from murmur.memory.records import Memory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT,
    use_count     INTEGER NOT NULL DEFAULT 0,
    source_turn   TEXT,
    superseded_by TEXT,
    low_priority  INTEGER NOT NULL DEFAULT 0,
    embedding     BLOB NOT NULL
);
"""


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class MemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Writes come from worker threads (asyncio.to_thread); the lock serialises them.
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self.lock = threading.Lock()

    def load(self) -> list[tuple[Memory, Vector]]:
        rows = self._db.execute(
            "SELECT id, text, created_at, last_used_at, use_count, source_turn, superseded_by,"
            " low_priority, embedding FROM memories"
        ).fetchall()
        return [
            (
                Memory(
                    id=r[0],
                    text=r[1],
                    created_at=datetime.fromisoformat(r[2]),
                    last_used_at=_ts(r[3]),
                    use_count=r[4],
                    source_turn=r[5],
                    superseded_by=r[6],
                    low_priority=bool(r[7]),
                ),
                np.frombuffer(r[8], dtype=np.float32).copy(),
            )
            for r in rows
        ]

    def insert(self, memory: Memory, embedding: Vector) -> None:
        self._db.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.id,
                memory.text,
                memory.created_at.isoformat(),
                memory.last_used_at.isoformat() if memory.last_used_at else None,
                memory.use_count,
                memory.source_turn,
                memory.superseded_by,
                int(memory.low_priority),
                embedding.astype(np.float32).tobytes(),
            ),
        )

    def update(self, memory: Memory) -> None:
        self._db.execute(
            "UPDATE memories SET last_used_at = ?, use_count = ?, superseded_by = ?,"
            " low_priority = ? WHERE id = ?",
            (
                memory.last_used_at.isoformat() if memory.last_used_at else None,
                memory.use_count,
                memory.superseded_by,
                int(memory.low_priority),
                memory.id,
            ),
        )

    def delete(self, ids: list[str]) -> None:
        self._db.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in ids])

    def close(self) -> None:
        self._db.close()
