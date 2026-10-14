"""Backend selection lives here and nowhere else."""

from murmur.config import Settings
from murmur.llm.base import LLM, ChatMessage
from murmur.llm.hosted import openrouter
from murmur.llm.local import vllm

__all__ = ["LLM", "ChatMessage", "build_llm"]


def build_llm(settings: Settings) -> LLM:
    return openrouter(settings) if settings.backend == "hosted" else vllm(settings)
