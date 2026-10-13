import asyncio
from datetime import timedelta
from pathlib import Path

from mcp import Client

from murmur.agent.memory_jobs import (
    MemoryJobs,
    TurnForMemory,
    is_grounded,
    parse_facts,
    parse_verdict,
    worth_extracting,
)
from murmur.mcp_servers.memory import build_memory_server
from murmur.memory import MemoryService
from tests.fakes import FakeLLM, make_memory


def make_jobs(memory: MemoryService, llm: FakeLLM) -> MemoryJobs:
    return MemoryJobs(
        llm,
        memory,
        duplicate_threshold=0.95,
        related_threshold=0.3,
        stale_after=timedelta(days=30),
    )


def turn(text: str, skip: bool = False) -> TurnForMemory:
    return TurnForMemory("turn_x", text, "", (), skip_extraction=skip)


def test_parse_facts_and_verdicts() -> None:
    assert parse_facts("NONE") == []
    assert parse_facts("Facts:\n- Has a dog\n-  Lives in Pune \nnoise") == [
        "Has a dog",
        "Lives in Pune",
    ]
    assert parse_facts("- Sibling named [Sister's Name]\n- Has a sister") == ["Has a sister"]
    assert parse_verdict("Replaced.") == "replaced"
    assert parse_verdict("I think they are unrelated") == "keep"  # unparseable is safe


def test_only_turns_about_the_user_are_extracted() -> None:
    assert worth_extracting("I just moved to Bangalore", "")
    assert worth_extracting("Mostly old ghazals.", "What music do you like?")
    assert not worth_extracting("What's the capital of Australia?", "")
    assert not worth_extracting("Tell me a joke.", "Here is one. Want another?")


def test_grounding_drops_invented_facts() -> None:
    assert is_grounded("Has a sister called Afifa", "my sister Afifa is visiting")
    assert is_grounded("Has a sister called Afifa", "Afifa. What is your sister called?")
    # Shares only frame words ("the user lives in"), so it was invented, not heard.
    assert not is_grounded("The user lives in New York City", "Tell me a joke, the user said")


async def drain(jobs: MemoryJobs) -> None:
    worker = asyncio.create_task(jobs.run())
    await jobs.flush()
    worker.cancel()


async def test_extracted_facts_are_written(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    llm = FakeLLM(completions=["- Has a sister called Afifa\n- Plays badminton"])
    jobs = make_jobs(memory, llm)

    jobs.submit(turn("my sister Afifa and I play badminton"))
    await drain(jobs)

    assert {m.text for m in memory.all(include_superseded=False)} == {
        "Has a sister called Afifa",
        "Plays badminton",
    }


async def test_contradiction_supersedes_and_duplicate_merges(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    [chennai] = memory.add(["Lives in Chennai city"], None)
    llm = FakeLLM(
        completions=[
            "- Lives in Bangalore city",  # extraction
            "replaced",  # judge: Chennai vs Bangalore
            "- Lives in Bangalore city",  # extraction of a repeat
        ]
    )
    jobs = make_jobs(memory, llm)

    jobs.submit(turn("I moved to Bangalore"))
    jobs.submit(turn("as I said, I live in Bangalore"))
    await drain(jobs)

    active = memory.all(include_superseded=False)
    assert [m.text for m in active] == ["Lives in Bangalore city"]
    old = memory.get(chennai.id)
    assert old is not None and old.superseded_by == active[0].id
    assert active[0].use_count == 1  # the repeat merged into it


async def test_forget_turn_writes_nothing(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    llm = FakeLLM(completions=["- Plays badminton"])
    jobs = make_jobs(memory, llm)

    jobs.submit(turn("forget that I play badminton", skip=True))
    await drain(jobs)

    assert memory.all(include_superseded=True) == []
    assert llm.completion_prompts == []


async def test_worker_waits_for_quiet_unless_flushed(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    llm = FakeLLM(completions=["- Plays badminton"])
    jobs = make_jobs(memory, llm)
    jobs.voice_idle.clear()  # a turn is running
    worker = asyncio.create_task(jobs.run())

    jobs.submit(turn("I play badminton"))
    await asyncio.sleep(0.05)
    assert llm.completion_prompts == []  # held back while the voice loop is busy

    await asyncio.wait_for(jobs.flush(), timeout=1)  # forget needs it now
    assert len(memory.all(include_superseded=False)) == 1
    worker.cancel()


async def test_forget_over_mcp_lands_pending_writes_first(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    jobs = make_jobs(memory, FakeLLM(completions=["- Plays badminton on weekdays"]))
    jobs.voice_idle.clear()
    worker = asyncio.create_task(jobs.run())
    jobs.submit(turn("I play badminton on weekdays"))  # not yet written
    server = build_memory_server(memory, forget_threshold=0.5, before_forget=jobs.flush)

    async with Client(server) as client:
        result = await client.call_tool("forget", {"description": "plays badminton"})

    text = " ".join(c.text for c in result.content if c.type == "text")
    assert "Forgotten: Plays badminton on weekdays" in text
    assert memory.all(include_superseded=True) == []
    worker.cancel()
