"""Memory operations over the store and an in-memory vector snapshot."""

import math
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from murmur.config import Settings
from murmur.memory.embed import Embedder, FastEmbedder, Vector
from murmur.memory.records import Memory, utc_now
from murmur.memory.store import MemoryStore

# Ranking boosts on top of cosine similarity. Small on purpose: relevance decides what surfaces,
# recency and use only break ties between similarly relevant facts.
_RECENCY_WEIGHT = 0.05
_RECENCY_HALF_LIFE_DAYS = 30.0
_USE_WEIGHT = 0.05
_LOW_PRIORITY_PENALTY = 0.15


def open_memory(settings: Settings) -> "MemoryService":
    return MemoryService(
        MemoryStore(settings.memory_db),
        FastEmbedder(settings.embed_model, settings.models_dir),
        settings.recall_min_similarity,
    )


@dataclass(frozen=True)
class _Snapshot:
    records: tuple[Memory, ...]
    matrix: npt.NDArray[np.float32]


class MemoryService:
    """Reads (recall, similar) are pure and lock-free; mutations rebuild the snapshot."""

    def __init__(self, store: MemoryStore, embedder: Embedder, min_similarity: float) -> None:
        self._store = store
        self._embedder = embedder
        # bge similarities are compressed: unrelated facts still score ~0.4-0.5 (see R2).
        self._min_similarity = min_similarity
        self._vectors: dict[str, Vector] = {}
        self._records: dict[str, Memory] = {}
        for memory, vector in store.load():
            self._records[memory.id] = memory
            self._vectors[memory.id] = vector
        self._snapshot = self._build_snapshot()

    def _build_snapshot(self) -> _Snapshot:
        records = tuple(self._records.values())
        if not records:
            return _Snapshot((), np.zeros((0, self._embedder.dim), dtype=np.float32))
        return _Snapshot(records, np.stack([self._vectors[m.id] for m in records]))

    def _similarities(self, text: str) -> tuple[_Snapshot, npt.NDArray[np.float32]]:
        snapshot = self._snapshot  # one reference: a concurrent mutation swaps, never edits
        if not snapshot.records:
            return snapshot, np.zeros(0, dtype=np.float32)
        return snapshot, snapshot.matrix @ self._embedder.embed_query(text)

    # Reads: safe on the turn path, no side effects.

    def recall(self, query: str, k: int, now: datetime | None = None) -> list[Memory]:
        now = now or utc_now()
        snapshot, sims = self._similarities(query)
        scored: list[tuple[float, Memory]] = []
        for memory, sim in zip(snapshot.records, sims, strict=True):
            if not memory.active or sim < self._min_similarity:
                continue
            scored.append((float(sim) + self._boost(memory, now), memory))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored[:k]]

    def _boost(self, memory: Memory, now: datetime) -> float:
        age_days = (now - (memory.last_used_at or memory.created_at)).total_seconds() / 86400
        recency = math.pow(0.5, max(age_days, 0.0) / _RECENCY_HALF_LIFE_DAYS)
        use = min(math.log1p(memory.use_count) / math.log(10), 1.0)
        penalty = _LOW_PRIORITY_PENALTY if memory.low_priority else 0.0
        return _RECENCY_WEIGHT * recency + _USE_WEIGHT * use - penalty

    def similar(
        self, text: str, threshold: float, include_superseded: bool
    ) -> list[tuple[Memory, float]]:
        snapshot, sims = self._similarities(text)
        pairs = [
            (memory, float(sim))
            for memory, sim in zip(snapshot.records, sims, strict=True)
            if sim >= threshold and (include_superseded or memory.active)
        ]
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)

    def neighbours(self, memory_id: str, threshold: float) -> list[tuple[Memory, float]]:
        """Active memories similar to an existing one, most similar first."""
        snapshot = self._snapshot
        if memory_id not in self._vectors:
            return []
        sims = snapshot.matrix @ self._vectors[memory_id]
        pairs = [
            (memory, float(sim))
            for memory, sim in zip(snapshot.records, sims, strict=True)
            if memory.id != memory_id and memory.active and sim >= threshold
        ]
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)

    def get(self, memory_id: str) -> Memory | None:
        return self._records.get(memory_id)

    def all(self, include_superseded: bool) -> list[Memory]:
        return [m for m in self._snapshot.records if include_superseded or m.active]

    # Mutations: off the turn path only. Callers run them via asyncio.to_thread.

    def add(self, texts: list[str], source_turn: str | None) -> list[Memory]:
        if not texts:
            return []
        vectors = self._embedder.embed_passages(texts)
        now = utc_now()
        added = [
            Memory(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                text=text,
                created_at=now,
                last_used_at=None,
                use_count=0,
                source_turn=source_turn,
                superseded_by=None,
                low_priority=False,
            )
            for text in texts
        ]
        with self._store.lock:
            for memory, vector in zip(added, vectors, strict=True):
                self._store.insert(memory, vector)
                self._records[memory.id] = memory
                self._vectors[memory.id] = vector
            self._snapshot = self._build_snapshot()
        return added

    def mark_used(self, ids: list[str]) -> None:
        now = utc_now()
        self._apply(
            {
                i: replace(m, use_count=m.use_count + 1, last_used_at=now, low_priority=False)
                for i in ids
                if (m := self._records.get(i))
            }
        )

    def supersede(self, old_id: str, new_id: str) -> None:
        # The old fact is kept, pointing at its replacement: "what did you think before?"
        if (old := self._records.get(old_id)) and new_id in self._records:
            self._apply({old_id: replace(old, superseded_by=new_id)})

    def decay(self, stale_after: timedelta, now: datetime | None = None) -> int:
        now = now or utc_now()
        stale = {
            m.id: replace(m, low_priority=True)
            for m in self._records.values()
            if m.active
            and not m.low_priority
            and now - (m.last_used_at or m.created_at) > stale_after
        }
        self._apply(stale)
        return len(stale)

    def _apply(self, changed: dict[str, Memory]) -> None:
        if not changed:
            return
        with self._store.lock:
            for memory in changed.values():
                self._store.update(memory)
                self._records[memory.id] = memory
            self._snapshot = self._build_snapshot()

    def forget(self, description: str, threshold: float) -> list[Memory]:
        """Hard delete what matches. Keeping something the user asked to forget is wrong."""
        matches = self.similar(description, threshold, include_superseded=True)
        return self.delete([memory.id for memory, _ in matches])

    def delete(self, ids: list[str]) -> list[Memory]:
        """Hard delete, including the history each one superseded. Returns what was removed."""
        doomed = set(ids) & self._records.keys()
        frontier = set(doomed)
        while frontier:
            frontier = {m.id for m in self._records.values() if m.superseded_by in frontier}
            frontier -= doomed
            doomed |= frontier
        removed = [self._records[i] for i in doomed]
        with self._store.lock:
            self._store.delete(list(doomed))
            for i in doomed:
                del self._records[i]
                del self._vectors[i]
            self._snapshot = self._build_snapshot()
        return removed
