"""The agent loop with working memory: the current session, bounded by turn count."""

from collections.abc import AsyncIterator

from murmur.llm.base import LLM, ChatMessage


class Agent:
    def __init__(self, llm: LLM, system_prompt: str, max_turns: int) -> None:
        self._llm = llm
        self._system = ChatMessage("system", system_prompt)
        self._max_messages = max_turns * 2
        self.history: list[ChatMessage] = []

    def _prompt(self) -> list[ChatMessage]:
        # +1: the last max_turns complete exchanges plus the user message being answered.
        return [self._system, *self.history[-(self._max_messages + 1) :]]

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        """Stream the reply. The caller commits what was actually spoken via commit_reply."""
        self.history.append(ChatMessage("user", user_text))
        async for delta in self._llm.stream(self._prompt()):
            yield delta

    def commit_reply(self, spoken_text: str) -> None:
        # Record what was spoken, not what was generated; barge-in truncates mid-reply.
        if spoken_text:
            self.history.append(ChatMessage("assistant", spoken_text))
