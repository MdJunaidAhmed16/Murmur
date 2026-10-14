"""Tools the agent can call, consumed over MCP."""

import re
from typing import Any, Protocol

from mcp import Client

from murmur.llm.base import ToolSpec
from murmur.mcp_servers.tools import DIVIDE, TIMES

# A tool is offered only when the user's words call for it. A 1.5B model offered "forget" on
# every turn called it on plain statements ("I play badminton"), losing the fact (R2). It also
# keeps tool schemas out of most prompts, which is prefill time saved. Tools not listed here
# are always offered.
TRIGGERS: dict[str, re.Pattern[str]] = {
    "forget": re.compile(r"\b(forget|delete|erase|remove|wipe|unlearn)\b", re.IGNORECASE),
    "current_time": re.compile(
        r"\b(time|date|day|today|tonight|clock|o'clock|morning|evening|month|year)\b",
        re.IGNORECASE,
    ),
    "calculate": re.compile(
        rf"\d\s*[-+*/x{TIMES}{DIVIDE}^%]\s*\d"
        r"|\b(plus|minus|times|divided|multiplied|percent|square root|calculate|how much is"
        r"|what is \d)\b",
        re.IGNORECASE,
    ),
}


def offered(specs: list[ToolSpec], user_text: str) -> list[ToolSpec]:
    return [s for s in specs if s.name not in TRIGGERS or TRIGGERS[s.name].search(user_text)]


class ToolBox(Protocol):
    specs: list[ToolSpec]

    async def call(self, name: str, arguments: dict[str, Any]) -> str: ...


class McpToolBox:
    """MCP clients exposing an allow-listed subset of their servers' tools to the model.

    Fewer tools means fewer chances for a 1.5B model to call the wrong one.
    """

    def __init__(self, routes: dict[str, Client], specs: list[ToolSpec]) -> None:
        self._routes = routes
        self.specs = specs

    @classmethod
    async def connect(cls, clients: list[Client], allow: set[str]) -> "McpToolBox":
        routes: dict[str, Client] = {}
        specs: list[ToolSpec] = []
        for client in clients:
            for tool in (await client.list_tools()).tools:
                if tool.name in allow and tool.name not in routes:
                    routes[tool.name] = client
                    specs.append(ToolSpec(tool.name, tool.description or "", tool.input_schema))
        return cls(routes, specs)

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        client = self._routes.get(name)
        if client is None:
            return f"Unknown tool {name}."
        result = await client.call_tool(name, arguments)
        text = " ".join(c.text for c in result.content if c.type == "text")
        return f"Error: {text}" if result.is_error else text
