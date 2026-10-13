"""vLLM through its OpenAI-compatible server."""

from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from murmur.llm.base import LLM, ChatMessage


def _to_openai(message: ChatMessage) -> ChatCompletionMessageParam:
    return cast(ChatCompletionMessageParam, {"role": message.role, "content": message.content})


class VllmLLM(LLM):
    backend_name = "local"

    def __init__(self, base_url: str, model: str, max_tokens: int) -> None:
        # vLLM ignores the key, but the client refuses to start without one.
        self._client = AsyncOpenAI(base_url=base_url, api_key="unused")
        self._model = model
        self._max_tokens = max_tokens

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[_to_openai(m) for m in messages],
            max_tokens=self._max_tokens,
            temperature=0.7,
            stream=True,
        )
        try:
            async for chunk in response:
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    yield delta
        finally:
            # Closing the stream on cancellation aborts the request in vLLM, freeing the slot.
            await response.close()
