"""One LLM interface for both backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLM(ABC):
    backend_name: str

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Yield text deltas as they are generated."""
