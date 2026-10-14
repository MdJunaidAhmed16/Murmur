"""Splits a token stream into sentences so synthesis starts before generation ends."""

import re

# End punctuation, optional closing quote/bracket, then whitespace. Newlines also end a sentence.
_SENTENCE = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
# For the first chunk only: a clause break also counts.
_CLAUSE = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|(?<=[,;:—])\s+|\n+")


class SentenceChunker:
    def __init__(self, min_chars: int = 8, first_clause: bool = False, clause_min_chars: int = 24):
        # Below min_chars, "Hi." or "Dr." waits for more text rather than becoming its own TTS call.
        self._min_chars = min_chars
        # Time to first audio grows with the first chunk's length (TTS synthesises a chunk whole),
        # so a long opening sentence is cut at its first clause. Later chunks stay whole sentences
        # because they synthesise while earlier audio plays. Short clauses ("Well,") still wait.
        self._first_clause = first_clause
        self._clause_min_chars = clause_min_chars
        self._emitted = 0
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        chunks: list[str] = []
        start = 0
        while True:
            clause_ok = self._first_clause and self._emitted == 0
            pattern = _CLAUSE if clause_ok else _SENTENCE
            match = next(
                (
                    m
                    for m in pattern.finditer(self._buffer, start)
                    if len(self._buffer[start : m.end()].strip()) >= self._min_for(m, clause_ok)
                ),
                None,
            )
            if match is None:
                break
            chunks.append(self._buffer[start : match.end()].strip())
            start = match.end()
            self._emitted += 1
        self._buffer = self._buffer[start:]
        return chunks

    def _min_for(self, match: re.Match[str], clause_ok: bool) -> int:
        is_clause = clause_ok and self._buffer[match.start() - 1 : match.start()] in ",;:—"
        return self._clause_min_chars if is_clause else self._min_chars

    def flush(self) -> str:
        rest, self._buffer = self._buffer.strip(), ""
        return rest
