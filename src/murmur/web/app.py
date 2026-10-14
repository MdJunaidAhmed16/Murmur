"""FastAPI app: the demo page, the audio WebSocket, and what the agent remembers."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mcp import Client

from murmur.agent.loop import Agent
from murmur.agent.tools import McpToolBox
from murmur.audio.frames import FRAME_SAMPLES, INPUT_SAMPLE_RATE
from murmur.audio.vad import SileroVAD
from murmur.config import Settings, get_settings
from murmur.llm import LLM, build_llm
from murmur.mcp_servers.tools import build_tools_server
from murmur.memory import MemoryService, open_memory
from murmur.memory.embed import FastEmbedder
from murmur.pipeline.metrics import TurnLog
from murmur.pipeline.orchestrator import Session
from murmur.stt.local import WhisperSTT
from murmur.tts.local import KokoroTTS
from murmur.web.memory_scope import MemoryScope, open_scope, visitor_store

log = structlog.get_logger()
STATIC_DIR = Path(__file__).parent / "static"
# Recall is injected into the prompt, not called as a tool. Each of these is further gated on
# the user's words (agent/tools.py TRIGGERS).
AGENT_TOOLS = {"forget", "current_time", "calculate"}


@dataclass
class Engines:
    settings: Settings
    stt: WhisperSTT
    tts: KokoroTTS
    llm: LLM
    embedder: FastEmbedder
    tools_client: Client
    # One store for personal use; None in demo mode, where each visitor opens their own.
    shared_scope: MemoryScope | None
    turn_log: TurnLog
    # One session at a time (CLAUDE.md scope): the GPU serves one conversation.
    session_lock: asyncio.Lock


def load_models(settings: Settings) -> tuple[WhisperSTT, KokoroTTS, FastEmbedder]:
    stt = WhisperSTT(
        settings.stt_model, settings.stt_device, settings.stt_compute_type, settings.models_dir
    )
    tts = KokoroTTS(
        settings.models_dir, settings.tts_voice, settings.tts_speed, settings.tts_device
    )
    embedder = FastEmbedder(settings.embed_model, settings.models_dir)
    # The first call pays for kernel loading and allocation; keep it out of turn one's numbers.
    stt.warm_up()
    tts.warm_up()
    embedder.embed_query("warm up")
    return stt, tts, embedder


class WebSocketOutbound:
    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        # Generation and synthesis both send; frames must not interleave mid-write.
        self._lock = asyncio.Lock()

    async def audio(self, pcm: bytes) -> None:
        async with self._lock:
            await self._ws.send_bytes(pcm)

    async def event(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            await self._ws.send_json(payload)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stt, tts, embedder = await asyncio.to_thread(load_models, settings)
        llm = build_llm(settings)
        async with AsyncExitStack() as stack:
            tools_client = await stack.enter_async_context(
                Client(build_tools_server(settings.timezone))
            )
            shared = None
            if not settings.demo_mode:
                shared = await stack.enter_async_context(
                    open_scope(settings, llm, embedder, settings.memory_db)
                )
            app.state.engines = Engines(
                settings=settings,
                stt=stt,
                tts=tts,
                llm=llm,
                embedder=embedder,
                tools_client=tools_client,
                shared_scope=shared,
                turn_log=TurnLog(settings.results_dir),
                session_lock=asyncio.Lock(),
            )
            log.info(
                "ready",
                url=f"http://{settings.host}:{settings.port}",
                backend=llm.backend_name,
                demo_mode=settings.demo_mode,
                memories=len(shared.memory.all(include_superseded=False)) if shared else None,
                tts_providers=tts.providers,
            )
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        # Lifespan has finished loading the models by the time requests are served.
        return {"ok": True}

    @app.get("/api/memories")
    async def memories(visitor: str = "") -> dict[str, Any]:
        engines: Engines = app.state.engines
        return {"memories": await _list_memories(engines, visitor)}

    @app.websocket("/ws")
    async def audio_socket(websocket: WebSocket, visitor: str = "") -> None:
        engines: Engines = websocket.app.state.engines
        if engines.session_lock.locked():
            await websocket.close(code=1013, reason="someone else is talking to it; try again soon")
            return
        async with engines.session_lock, AsyncExitStack() as stack:
            if engines.shared_scope is not None:
                scope = engines.shared_scope
            else:
                path = visitor_store(engines.settings, visitor)
                if path is None:
                    await websocket.close(code=1008, reason="missing or invalid visitor id")
                    return
                scope = await stack.enter_async_context(
                    open_scope(engines.settings, engines.llm, engines.embedder, path)
                )
            await websocket.accept()
            await _serve_session(websocket, engines, scope)

    return app


def _memory_view(memory: MemoryService) -> list[dict[str, Any]]:
    rows = []
    for m in sorted(memory.all(include_superseded=True), key=lambda m: m.created_at):
        replaced = memory.get(m.superseded_by) if m.superseded_by else None
        rows.append(
            {
                "text": m.text,
                "since": m.created_at.date().isoformat(),
                "replaced_by": replaced.text if replaced else None,
            }
        )
    return rows


async def _list_memories(engines: Engines, visitor: str) -> list[dict[str, Any]]:
    if engines.shared_scope is not None:
        return _memory_view(engines.shared_scope.memory)
    path = visitor_store(engines.settings, visitor)
    if path is None:
        raise HTTPException(400, "missing or invalid visitor id")
    if not path.exists():
        return []
    # A short-lived read of the visitor's own file; WAL lets it run beside a live session.
    memory = await asyncio.to_thread(open_memory, engines.settings, path, engines.embedder)
    try:
        return _memory_view(memory)
    finally:
        memory.close()


def _parse_control(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


async def _serve_session(websocket: WebSocket, engines: Engines, scope: MemoryScope) -> None:
    settings = engines.settings
    outbound = WebSocketOutbound(websocket)
    tools = await McpToolBox.connect([scope.client, engines.tools_client], AGENT_TOOLS)
    session = Session(
        settings=settings,
        vad=SileroVAD(settings.models_dir / "silero_vad.onnx"),
        stt=engines.stt,
        tts=engines.tts,
        agent=Agent(engines.llm, settings.working_memory_turns, tools),
        memory=scope.memory,
        memory_jobs=scope.jobs,
        backend_name=engines.llm.backend_name,
        outbound=outbound,
        turn_log=engines.turn_log,
    )
    await outbound.event(
        {
            "type": "config",
            "input_sample_rate": INPUT_SAMPLE_RATE,
            "frame_samples": FRAME_SAMPLES,
            "output_sample_rate": engines.tts.sample_rate,
            "backend": engines.llm.backend_name,
            "demo_mode": settings.demo_mode,
        }
    )
    log.info("session_open", backend=engines.llm.backend_name)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.handle_frame(data)
            elif (text := message.get("text")) is not None:
                await session.handle_control(_parse_control(text))
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        log.info("session_closed")
