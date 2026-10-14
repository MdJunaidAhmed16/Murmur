from pathlib import Path

import pytest

from murmur.config import Settings
from murmur.llm import build_llm
from murmur.llm.hosted import request_options
from murmur.web.memory_scope import open_scope, visitor_store
from tests.fakes import FakeLLM, WordEmbedder


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_hosted_backend_needs_a_key() -> None:
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_llm(settings(backend="hosted"))


def test_hosted_backend_is_selected_by_the_flag_alone() -> None:
    llm = build_llm(settings(backend="hosted", openrouter_api_key="sk-or-test"))
    assert llm.backend_name == "hosted"
    assert build_llm(settings()).backend_name == "local"


def test_hosted_requests_send_effort_and_no_sampling_parameters() -> None:
    # Claude 5.x rejects temperature/top_p; a stray one would fail every request.
    options = request_options(settings(hosted_reasoning_effort="low"))
    assert options == {"extra_body": {"reasoning": {"effort": "low"}}}


@pytest.mark.parametrize("visitor", ["", "../../etc/passwd", "ABCD" * 4, "a" * 15])
def test_visitor_ids_that_could_escape_the_directory_are_refused(visitor: str) -> None:
    assert visitor_store(settings(), visitor) is None


def test_each_visitor_gets_their_own_store(tmp_path: Path) -> None:
    s = settings(demo_dir=tmp_path)
    a = visitor_store(s, "a" * 32)
    b = visitor_store(s, "b" * 32)
    assert a is not None and b is not None and a != b
    assert a.parent == b.parent == tmp_path


async def test_scope_lands_pending_writes_before_closing(tmp_path: Path) -> None:
    llm = FakeLLM(completions=["- Plays badminton"])
    path = tmp_path / "v.db"
    s = settings()
    async with open_scope(s, llm, WordEmbedder(), path) as scope:
        scope.jobs.voice_idle.clear()  # a turn is still "running" when the visitor leaves
        from murmur.agent.memory_jobs import TurnForMemory

        scope.jobs.submit(TurnForMemory("t", "I play badminton", "", (), skip_extraction=False))

    async with open_scope(s, llm, WordEmbedder(), path) as reopened:
        assert [m.text for m in reopened.memory.all(include_superseded=False)] == [
            "Plays badminton"
        ]
