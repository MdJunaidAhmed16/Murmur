"""The STT interface: one stream per utterance."""

from abc import ABC, abstractmethod

from murmur.audio.frames import Float32Audio


class TranscriptionStream(ABC):
    @abstractmethod
    def feed(self, frame: Float32Audio) -> None: ...

    @abstractmethod
    async def partial(self) -> str:
        """Best guess so far; may be revised."""

    @abstractmethod
    async def finalize(self) -> str: ...


class SpeechToText(ABC):
    @abstractmethod
    def stream(self) -> TranscriptionStream: ...
