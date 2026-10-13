"""FastAPI app: the demo page and the audio WebSocket."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from murmur.agent.loop import Agent
from murmur.agent.prompts import SYSTEM_PROMPT
from murmur.audio.frames import FRAME_SAMPLES, INPUT_SAMPLE_RATE
from murmur.audio.vad import SileroVAD
from murmur.config import Settings, get_settings
from murmur.llm import LLM, build_llm
from murmur.pipeline.metrics import TurnLog
from murmur.pipeline.orchestrator import Session
from murmur.stt.local import WhisperSTT
from murmur.tts.local import KokoroTTS

log = structlog.get_logger()
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class Engines:
    settings: Settings
    stt: WhisperSTT
    tts: KokoroTTS
    llm: LLM
    turn_log: TurnLog
    # Single user, one session at a time (CLAUDE.md scope).
    session_lock: asyncio.Lock


def load_engines(settings: Settings) -> Engines:
    stt = WhisperSTT(
        settings.stt_model, settings.stt_device, settings.stt_compute_type, settings.models_dir
    )
    tts = KokoroTTS(settings.models_dir, settings.tts_voice, settings.tts_speed)
    # The first call pays for kernel loading and allocation; keep it out of turn one's numbers.
    stt.warm_up()
    tts.warm_up()
    return Engines(
        settings=settings,
        stt=stt,
        tts=tts,
        llm=build_llm(settings),
        turn_log=TurnLog(settings.results_dir),
        session_lock=asyncio.Lock(),
    )


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
        app.state.engines = await asyncio.to_thread(load_engines, settings)
        log.info("ready", url=f"http://{settings.host}:{settings.port}")
        yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def audio_socket(websocket: WebSocket) -> None:
        engines: Engines = websocket.app.state.engines
        if engines.session_lock.locked():
            await websocket.close(code=1013, reason="another session is active")
            return
        async with engines.session_lock:
            await websocket.accept()
            await _serve_session(websocket, engines)

    return app


async def _serve_session(websocket: WebSocket, engines: Engines) -> None:
    settings = engines.settings
    outbound = WebSocketOutbound(websocket)
    session = Session(
        settings=settings,
        vad=SileroVAD(settings.models_dir / "silero_vad.onnx"),
        stt=engines.stt,
        tts=engines.tts,
        agent=Agent(engines.llm, SYSTEM_PROMPT, settings.working_memory_turns),
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
        }
    )
    log.info("session_open")
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.handle_frame(data)
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        log.info("session_closed")
