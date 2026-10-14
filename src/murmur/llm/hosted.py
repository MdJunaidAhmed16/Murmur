"""Claude through OpenRouter, for the always-on demo. Same protocol as vLLM, different options."""

from typing import Any

from murmur.config import Settings
from murmur.llm.openai_chat import OpenAIChatLLM


def request_options(settings: Settings) -> dict[str, Any]:
    # No temperature: Claude 5.x rejects sampling parameters. Reasoning effort is the lever,
    # and a spoken reply wants the lowest one; every extra thought is time before first audio.
    return {"extra_body": {"reasoning": {"effort": settings.hosted_reasoning_effort}}}


def openrouter(settings: Settings) -> OpenAIChatLLM:
    if settings.openrouter_api_key is None:
        raise RuntimeError("MURMUR_BACKEND=hosted needs MURMUR_OPENROUTER_API_KEY")
    options = request_options(settings)
    return OpenAIChatLLM(
        backend_name="hosted",
        base_url=settings.hosted_base_url,
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.hosted_model,
        max_tokens=settings.llm_max_tokens,
        stream_options=options,
        complete_options=options,
        # OpenRouter's optional attribution headers: the app shows up by name in its dashboard.
        headers={"HTTP-Referer": "https://github.com/MdJunaidAhmed16/Murmur", "X-Title": "Murmur"},
    )
