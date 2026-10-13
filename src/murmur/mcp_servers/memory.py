"""The memory MCP server.

In the app it runs in-process (the agent connects with an in-memory transport, no subprocess on
the turn path). Standalone, `python -m murmur.mcp_servers.memory` serves the same store over
stdio for any MCP client. Recall on the turn path does not go through MCP; it is too close to
the latency budget to pay for a protocol hop.
"""

import asyncio
from collections.abc import Awaitable, Callable

from mcp.server.mcpserver import MCPServer

from murmur.config import get_settings
from murmur.memory import MemoryService, open_memory

# The agent reads this prefix to scrub forgotten facts from the session transcript too.
FORGOTTEN_PREFIX = "Forgotten: "
FORGOTTEN_SEPARATOR = "; "


def build_memory_server(
    memory: MemoryService,
    forget_threshold: float,
    before_forget: Callable[[], Awaitable[None]] | None = None,
) -> MCPServer:
    server = MCPServer(
        "murmur-memory",
        instructions="Long-term memory of facts about the user of the Murmur voice agent.",
    )

    @server.tool()
    async def forget(description: str) -> str:
        """Permanently delete what you remember about the user matching a short description.

        Use when the user asks you to forget something. Describe the fact in third person,
        for example "plays badminton" or "sister's name".
        """
        if before_forget is not None:
            # A fact from the previous turn may still be queued for writing; land it first.
            await before_forget()
        removed = await asyncio.to_thread(memory.forget, description, forget_threshold)
        if not removed:
            return f"Nothing matching '{description}' was stored."
        return FORGOTTEN_PREFIX + FORGOTTEN_SEPARATOR.join(m.text for m in removed)

    @server.tool()
    async def recall(query: str, k: int = 5) -> list[str]:
        """Facts about the user most relevant to a query."""
        return [m.text for m in await asyncio.to_thread(memory.recall, query, k)]

    @server.tool()
    async def remember(fact: str) -> str:
        """Store one atomic fact about the user, in third person."""
        added = await asyncio.to_thread(memory.add, [fact], None)
        return f"Stored {added[0].id}."

    @server.tool()
    def history() -> list[str]:
        """Every stored fact, including superseded ones and what replaced them."""
        lines = []
        for m in sorted(memory.all(include_superseded=True), key=lambda m: m.created_at):
            replaced = memory.get(m.superseded_by) if m.superseded_by else None
            suffix = f" (no longer true; replaced by: {replaced.text})" if replaced else ""
            lines.append(f"{m.created_at:%Y-%m-%d} {m.text}{suffix}")
        return lines

    return server


def main() -> None:
    settings = get_settings()
    build_memory_server(open_memory(settings), settings.forget_threshold).run("stdio")


if __name__ == "__main__":
    main()
