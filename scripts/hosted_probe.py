"""First-token latency of hosted models through the real backend code. A few cents at most.

uv run python scripts/hosted_probe.py anthropic/claude-opus-5.5 anthropic/claude-haiku-5.5
"""

import asyncio
import sys

import numpy as np

from murmur.agent.prompts import build_system_prompt
from murmur.config import get_settings
from murmur.llm.base import ChatMessage
from murmur.llm.hosted import openrouter
from murmur.pipeline.metrics import now_ms

QUESTIONS = [
    "What is my account balance?",
    "How much money can I withdraw per day from the ATM?",
    "Can I use my card in another country?",
    "Tell me a fun fact about octopuses.",
    "What's a good way to start the morning?",
]


async def probe(model: str) -> None:
    settings = get_settings().model_copy(update={"hosted_model": model, "llm_max_tokens": 96})
    llm = openrouter(settings)
    first: list[float] = []
    for question in QUESTIONS:
        messages = [ChatMessage("system", build_system_prompt([])), ChatMessage("user", question)]
        start = now_ms()
        reply: list[str] = []
        async for delta in llm.stream(messages):
            if isinstance(delta, str):
                if not reply:
                    first.append(now_ms() - start)
                reply.append(delta)
        print(f"  {first[-1]:6.0f} ms  {''.join(reply)[:90]!r}")
    print(f"{model}: first token p50 {np.percentile(first, 50):.0f} ms, max {max(first):.0f} ms")


async def main(models: list[str]) -> int:
    for model in models:
        await probe(model)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or [get_settings().hosted_model])))
