"""The TTS interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TextToSpeech(ABC):
    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield mono int16 PCM chunks at sample_rate as soon as each is ready."""
