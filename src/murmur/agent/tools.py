"""Tools the agent can call, consumed over MCP."""

import re
from typing import Any, Protocol

from mcp import Client

from murmur.llm.base import ToolSpec

# A tool is offered only when the user's words call for it. A 1.5B model offered "forget" on
# every turn called it on plain statements ("I play badminton"), losing the fact (R2).
# Tools not listed here are always offered.
TRIGGERS: dict[str, re.Pattern[str]] = {
    "forget": re.compile(r"\b(forget|delete|erase|remove|wipe|unlearn)\b", re.IGNORECASE),
}


def offered(specs: list[ToolSpec], user_text: str) -> list[ToolSpec]:
    return [s for s in specs if s.name not in TRIGGERS or TRIGGERS[s.name].search(user_text)]


class ToolBox(Protocol):
    specs: list[ToolSpec]

    async def call(self, name: str, arguments: dict[str, Any]) -> str: ...


class McpToolBox:
    """An MCP client exposing an allow-listed subset of a server's tools to the model.

    Fewer tools means fewer chances for a 1.5B model to call the wrong one.
    """

    def __init__(self, client: Client, specs: list[ToolSpec]) -> None:
        self._client = client
        self.specs = specs

    @classmethod
    async def connect(cls, client: Client, allow: set[str]) -> "McpToolBox":
        listing = await client.list_tools()
        specs = [
            ToolSpec(name=t.name, description=t.description or "", parameters=t.input_schema)
            for t in listing.tools
            if t.name in allow
        ]
        return cls(client, specs)

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in {s.name for s in self.specs}:
            return f"Unknown tool {name}."
        result = await self._client.call_tool(name, arguments)
        text = " ".join(c.text for c in result.content if c.type == "text")
        return f"Error: {text}" if result.is_error else text
