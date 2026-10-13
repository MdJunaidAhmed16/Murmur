"""Splits a token stream into sentences so synthesis starts before generation ends."""

import re

# End punctuation, optional closing quote/bracket, then whitespace. Newlines also end a sentence.
_BOUNDARY = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")


class SentenceChunker:
    def __init__(self, min_chars: int = 8) -> None:
        # Below min_chars, "Hi." or "Dr." waits for more text rather than becoming its own TTS call.
        self._min_chars = min_chars
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        sentences: list[str] = []
        start = 0
        for match in _BOUNDARY.finditer(self._buffer):
            candidate = self._buffer[start : match.end()].strip()
            if len(candidate) >= self._min_chars:
                sentences.append(candidate)
                start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str:
        rest, self._buffer = self._buffer.strip(), ""
        return rest
