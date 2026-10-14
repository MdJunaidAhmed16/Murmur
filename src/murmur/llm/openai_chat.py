"""The OpenAI chat-completions protocol, which both backends speak (vLLM and OpenRouter)."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from murmur.llm.base import LLM, ChatMessage, ToolCall, ToolSpec


def to_openai(message: ChatMessage) -> ChatCompletionMessageParam:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return cast(ChatCompletionMessageParam, payload)


def tool_param(spec: ToolSpec) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


@dataclass
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)

    def finish(self) -> ToolCall:
        raw = "".join(self.arguments) or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            # Small models emit malformed JSON; surface it to the tool as-is rather than crash.
            arguments = {"_raw": raw}
        return ToolCall(id=self.id, name=self.name, arguments=arguments)


class OpenAIChatLLM(LLM):
    """Backends differ only in where they point and which request options they send."""

    def __init__(
        self,
        *,
        backend_name: str,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        stream_options: dict[str, Any],
        complete_options: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.backend_name = backend_name
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=headers)
        self._model = model
        self._max_tokens = max_tokens
        self._stream_options = stream_options
        self._complete_options = complete_options

    async def stream(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[str | ToolCall]:
        extra: dict[str, Any] = {"tools": [tool_param(t) for t in tools]} if tools else {}
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[to_openai(m) for m in messages],
            max_tokens=self._max_tokens,
            stream=True,
            **self._stream_options,
            **extra,
        )
        calls: dict[int, _PartialCall] = {}
        try:
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
                for part in delta.tool_calls or []:
                    call = calls.setdefault(part.index, _PartialCall())
                    call.id = part.id or call.id
                    if part.function and part.function.name:
                        call.name = part.function.name
                    if part.function and part.function.arguments:
                        call.arguments.append(part.function.arguments)
        finally:
            # Closing the stream on cancellation aborts the request upstream, freeing the slot.
            await response.close()
        for index in sorted(calls):
            yield calls[index].finish()

    async def complete(self, messages: list[ChatMessage], max_tokens: int) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[to_openai(m) for m in messages],
            max_tokens=max_tokens,
            **self._complete_options,
        )
        return response.choices[0].message.content or ""
