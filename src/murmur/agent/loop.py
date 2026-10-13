"""The agent loop with working memory: the current session, bounded by turn count."""

from collections.abc import AsyncIterator

import structlog

from murmur.agent.grounding import shares_content
from murmur.agent.prompts import build_system_prompt
from murmur.agent.tools import ToolBox, offered
from murmur.llm.base import LLM, ChatMessage, ToolCall
from murmur.mcp_servers.memory import FORGOTTEN_PREFIX, FORGOTTEN_SEPARATOR

log = structlog.get_logger()

# One tool round and one answer. A small model that keeps calling tools is looping, not helping.
_MAX_LLM_CALLS = 2


class Agent:
    def __init__(self, llm: LLM, max_turns: int, tools: ToolBox | None = None) -> None:
        self._llm = llm
        self._tools = tools
        self._max_messages = max_turns * 2
        self.history: list[ChatMessage] = []
        self.last_tool_calls: list[str] = []

    def _prompt(self, memories: list[str], scratch: list[ChatMessage]) -> list[ChatMessage]:
        # +1: the last max_turns complete exchanges plus the user message being answered.
        window = self.history[-(self._max_messages + 1) :]
        return [ChatMessage("system", build_system_prompt(memories)), *window, *scratch]

    def last_reply(self) -> str:
        return next((m.content for m in reversed(self.history) if m.role == "assistant"), "")

    async def respond(self, user_text: str, memories: list[str]) -> AsyncIterator[str]:
        """Stream the reply. The caller commits what was actually spoken via commit_reply."""
        self.history.append(ChatMessage("user", user_text))
        self.last_tool_calls = []
        # Tool exchanges live only for this turn; history keeps what was said aloud.
        scratch: list[ChatMessage] = []
        specs = offered(self._tools.specs, user_text) if self._tools else None
        for _ in range(_MAX_LLM_CALLS):
            calls: list[ToolCall] = []
            async for event in self._llm.stream(self._prompt(memories, scratch), specs):
                if isinstance(event, ToolCall):
                    calls.append(event)
                else:
                    yield event
            if not calls or self._tools is None:
                return
            scratch.append(ChatMessage("assistant", "", tool_calls=tuple(calls)))
            for call in calls:
                self.last_tool_calls.append(call.name)
                result = await self._tools.call(call.name, call.arguments)
                log.info("tool_call", name=call.name, arguments=call.arguments, result=result)
                if result.startswith(FORGOTTEN_PREFIX):
                    self._scrub(result.removeprefix(FORGOTTEN_PREFIX).split(FORGOTTEN_SEPARATOR))
                scratch.append(ChatMessage("tool", result, tool_call_id=call.id))
            specs = None  # the follow-up call answers; it does not get to call tools again

    def _scrub(self, forgotten: list[str]) -> None:
        # Deleting the fact from long-term memory is not enough if the session transcript still
        # says it: the model answers from history. The current request itself stays.
        current = self.history[-1]
        kept = [
            m
            for m in self.history[:-1]
            if not any(shares_content(m.content, fact) for fact in forgotten)
        ]
        log.info("history_scrubbed", removed=len(self.history) - 1 - len(kept))
        self.history = [*kept, current]

    def retract_unanswered(self) -> None:
        """Drop a trailing user message that got no reply; its words move to the next turn."""
        if self.history and self.history[-1].role == "user":
            self.history.pop()

    def commit_reply(self, spoken_text: str) -> None:
        # Record what was spoken, not what was generated; barge-in truncates mid-reply.
        if spoken_text:
            self.history.append(ChatMessage("assistant", spoken_text))
