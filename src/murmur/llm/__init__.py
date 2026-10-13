"""Backend selection lives here and nowhere else."""

from murmur.config import Settings
from murmur.llm.base import LLM, ChatMessage
from murmur.llm.local import VllmLLM

__all__ = ["LLM", "ChatMessage", "build_llm"]


def build_llm(settings: Settings) -> LLM:
    if settings.backend == "hosted":
        raise NotImplementedError("hosted backend lands on day 9 (llm/hosted.py)")
    return VllmLLM(settings.llm_base_url, settings.llm_model, settings.llm_max_tokens)
