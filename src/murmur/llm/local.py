"""vLLM on the local GPU, through its OpenAI-compatible server."""

from murmur.config import Settings
from murmur.llm.openai_chat import OpenAIChatLLM


def vllm(settings: Settings) -> OpenAIChatLLM:
    return OpenAIChatLLM(
        backend_name="local",
        base_url=settings.llm_base_url,
        # vLLM ignores the key, but the client refuses to start without one.
        api_key="unused",
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        stream_options={"temperature": 0.7},
        complete_options={"temperature": 0.0},
    )
