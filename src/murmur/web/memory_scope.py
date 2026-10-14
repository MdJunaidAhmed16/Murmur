"""One memory store with its worker and MCP server: shared for personal use, per visitor in demo.

A store, the background worker that writes to it, and the MCP server that forgets from it
belong together; opening and closing them as a unit keeps any of them from outliving the rest.
"""

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import structlog
from mcp import Client

from murmur.agent.memory_jobs import MemoryJobs
from murmur.config import Settings
from murmur.llm import LLM
from murmur.mcp_servers.memory import build_memory_server
from murmur.memory import MemoryService, open_memory
from murmur.memory.embed import Embedder

log = structlog.get_logger()

# Visitor ids come from the browser and become file names: hex only, so no path tricks.
_VISITOR_ID = re.compile(r"^[a-f0-9]{16,64}$")
# Writes still queued when a visitor leaves get this long to land before the worker stops.
_FLUSH_ON_CLOSE_S = 10.0


@dataclass
class MemoryScope:
    memory: MemoryService
    jobs: MemoryJobs
    client: Client  # MCP client of this store's memory server


def visitor_store(settings: Settings, visitor_id: str) -> Path | None:
    if not _VISITOR_ID.match(visitor_id):
        return None
    return settings.demo_dir / f"{visitor_id}.db"


@asynccontextmanager
async def open_scope(
    settings: Settings, llm: LLM, embedder: Embedder, path: Path
) -> AsyncIterator[MemoryScope]:
    memory = await asyncio.to_thread(open_memory, settings, path, embedder)
    jobs = MemoryJobs.from_settings(llm, memory, settings)
    server = build_memory_server(memory, settings.forget_threshold, before_forget=jobs.flush)
    async with Client(server) as client:
        worker = asyncio.create_task(jobs.run())
        try:
            await jobs.decay()
            yield MemoryScope(memory, jobs, client)
        finally:
            try:
                await asyncio.wait_for(jobs.flush(), _FLUSH_ON_CLOSE_S)
            except TimeoutError:
                log.warning("memory_writes_dropped_on_close", store=str(path))
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            memory.close()
