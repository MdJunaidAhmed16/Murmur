"""One LLM interface for both backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = field(default=())
    tool_call_id: str | None = None


class LLM(ABC):
    backend_name: str

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[str | ToolCall]:
        """Yield text deltas as generated; tool calls are yielded whole, once complete."""

    @abstractmethod
    async def complete(self, messages: list[ChatMessage], max_tokens: int) -> str:
        """Deterministic, non-streaming. For background work such as fact extraction."""
