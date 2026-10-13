"""Text embeddings. bge-small on CPU: ~6 ms per query, well inside the 40 ms recall budget."""

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> Vector: ...

    def embed_passages(self, texts: list[str]) -> list[Vector]: ...


class FastEmbedder:
    dim = 384

    def __init__(self, model: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        # Two threads: one query at a time, and STT/TTS need the remaining cores.
        self._model = TextEmbedding(model, cache_dir=str(cache_dir), threads=2)

    def embed_query(self, text: str) -> Vector:
        # bge uses an instruction prefix for queries; query_embed adds it. Outputs are unit length.
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.passage_embed(texts)]
