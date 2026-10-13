import asyncio
import json
from pathlib import Path

from murmur.agent.loop import Agent
from murmur.agent.memory_jobs import MemoryJobs
from murmur.config import Settings
from murmur.memory import MemoryService
from murmur.pipeline.metrics import TurnLog
from murmur.pipeline.orchestrator import Session
from tests.fakes import (
    SILENCE_FRAME,
    SPEECH_FRAME,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    LoudnessVAD,
    RecordingOutbound,
    WordEmbedder,
    make_memory,
)


def make_session(
    tmp_path: Path,
    stt: FakeSTT,
    llm: FakeLLM,
    tts: FakeTTS,
    memory: MemoryService | None = None,
) -> tuple[Session, Agent, RecordingOutbound]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        min_speech_ms=64,
        endpoint_silence_ms=64,
        stt_partial_interval_ms=0,
        results_dir=tmp_path,
    )
    memory = memory or make_memory(tmp_path / "memory.db")
    agent = Agent(llm, max_turns=4)
    out = RecordingOutbound()
    session = Session(
        settings=settings,
        vad=LoudnessVAD(),
        stt=stt,
        tts=tts,
        agent=agent,
        memory=memory,
        memory_jobs=MemoryJobs.from_settings(llm, memory, settings),
        backend_name=llm.backend_name,
        outbound=out,
        turn_log=TurnLog(tmp_path),
    )
    return session, agent, out


async def speak(session: Session, speech_frames: int = 5, silence_frames: int = 3) -> None:
    for _ in range(speech_frames):
        await session.handle_frame(SPEECH_FRAME)
    for _ in range(silence_frames):
        await session.handle_frame(SILENCE_FRAME)


async def wait_turn(session: Session) -> None:
    turn = session._turn
    assert turn is not None
    await asyncio.wait_for(asyncio.shield(turn), timeout=2)


async def test_full_turn_speaks_and_records(tmp_path: Path) -> None:
    stt = FakeSTT(["what time is it"])
    tts = FakeTTS()
    session, agent, out = make_session(
        tmp_path, stt, FakeLLM("It is noon. Have a good lunch."), tts
    )

    await speak(session)
    await wait_turn(session)

    assert out.of_type("user") == [{"type": "user", "text": "what time is it"}]
    assert tts.requests == ["It is noon.", "Have a good lunch."]
    assert out.audio_chunks == 6
    # Preroll means the frames before SPEECH_START still reach STT.
    assert stt.streams[0].frames >= 5
    assert agent.history[-1].content == "It is noon. Have a good lunch."

    record = json.loads((tmp_path / "turns.jsonl").read_text())
    assert record["barge_in"] is False
    assert record["ms"]["time_to_first_audio"] is not None
    assert record["spoken_chars"] == len("It is noon. Have a good lunch.")


async def test_barge_in_cancels_turn_and_keeps_only_spoken_text(tmp_path: Path) -> None:
    stt = FakeSTT(["tell me a story", "stop"])
    llm = FakeLLM("Once upon a time there was a fox. It lived in a wood. The end.", delay=0.01)
    tts = FakeTTS(delay=0.05)
    session, agent, out = make_session(tmp_path, stt, llm, tts)

    await speak(session)
    first_turn = session._turn
    assert first_turn is not None
    await asyncio.wait_for(out.wait_for_audio(3), timeout=2)  # first sentence fully sent

    await speak(session, silence_frames=0)

    assert first_turn.cancelled()
    assert out.of_type("clear")
    record = json.loads((tmp_path / "turns.jsonl").read_text().splitlines()[0])
    assert record["barge_in"] is True
    assert agent.history[-1].content == "Once upon a time there was a fox."
    await session.close()


async def test_speech_clears_browser_audio_after_turn_finished_sending(tmp_path: Path) -> None:
    # The turn is done once audio is sent, but the browser may still be playing it.
    stt = FakeSTT(["tell me a story", "wait"])
    session, _, out = make_session(
        tmp_path, stt, FakeLLM("A long reply. With two parts."), FakeTTS()
    )

    await speak(session)
    await wait_turn(session)
    assert not out.of_type("clear")

    await speak(session, silence_frames=0)

    assert out.of_type("clear")
    await session.close()


async def test_early_endpoint_carries_words_into_the_next_turn(tmp_path: Path) -> None:
    stt = FakeSTT(["please forget what I told you", "about my sister"])
    llm = FakeLLM("Done, forgotten.", delay=0.05)  # slow: no audio before the user resumes
    session, agent, out = make_session(tmp_path, stt, llm, FakeTTS())

    await speak(session)
    await asyncio.sleep(0.01)
    await speak(session)  # user carries on after a pause
    await wait_turn(session)

    asked = llm.prompts[-1][-1]
    assert asked.role == "user" and asked.content == "please forget what I told you about my sister"
    assert [m.content for m in agent.history if m.role == "user"] == [asked.content]
    assert [e["text"] for e in out.of_type("user")][-1] == asked.content


async def test_recalled_memories_reach_the_prompt(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "memory.db")
    memory.add(["Has a sister called Afifa", "Drives a red Swift"], None)
    llm = FakeLLM("Afifa, of course.")
    session, _, out = make_session(
        tmp_path, FakeSTT(["what is my sister called"]), llm, FakeTTS(), memory
    )

    await speak(session)
    await wait_turn(session)

    system = llm.prompts[0][0].content
    assert "Has a sister called Afifa" in system
    assert "Drives a red Swift" not in system
    assert out.of_type("recalled") == [
        {"type": "recalled", "memories": ["Has a sister called Afifa"]}
    ]
    record = out.of_type("metrics")
    assert record and record[0]["memories_retrieved"] == 1
    assert record[0]["ms"]["memory_recall"] is not None


async def test_recall_over_budget_answers_without_memories(tmp_path: Path) -> None:
    # 100 ms embedding against the 40 ms budget.
    memory = make_memory(tmp_path / "memory.db", WordEmbedder(delay_s=0.1))
    memory.add(["Has a sister called Afifa"], None)
    llm = FakeLLM("I am not sure.")
    session, _, out = make_session(
        tmp_path, FakeSTT(["what is my sister called"]), llm, FakeTTS(), memory
    )

    await speak(session)
    await wait_turn(session)

    assert "Afifa" not in llm.prompts[0][0].content
    assert out.audio_chunks > 0  # the turn still spoke
    assert json.loads((tmp_path / "turns.jsonl").read_text())["ms"]["memory_recall"] < 100


async def test_noise_without_words_is_not_a_turn(tmp_path: Path) -> None:
    session, agent, _ = make_session(tmp_path, FakeSTT([""]), FakeLLM("unused"), FakeTTS())

    await speak(session)
    await wait_turn(session)

    assert agent.history == []
    assert not (tmp_path / "turns.jsonl").exists()


async def test_short_blip_never_opens_a_turn(tmp_path: Path) -> None:
    session, _, out = make_session(tmp_path, FakeSTT([]), FakeLLM("unused"), FakeTTS())

    await speak(session, speech_frames=1)

    assert session._turn is None
    assert not out.of_type("speech_start")
