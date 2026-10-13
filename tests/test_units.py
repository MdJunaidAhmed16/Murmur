from murmur.agent.loop import Agent
from murmur.audio.vad import EndpointDetector, VadEvent
from murmur.pipeline.metrics import TurnTimer
from murmur.tts.text import SentenceChunker
from tests.fakes import FakeLLM


def test_endpoint_needs_sustained_speech_then_silence() -> None:
    detector = EndpointDetector(endpoint_silence_ms=96, min_speech_ms=64)  # 3 and 2 frames
    events = [detector.update(s) for s in [True, True, False, False, True, False, False, False]]
    assert events == [None, VadEvent.SPEECH_START, None, None, None, None, None, VadEvent.ENDPOINT]


def test_chunker_hands_off_at_sentence_boundaries() -> None:
    chunker = SentenceChunker()
    out: list[str] = []
    for delta in ["Sure thing", ". The weather", " is fine today! And", " tomorrow"]:
        out += chunker.feed(delta)
    assert out == ["Sure thing.", "The weather is fine today!"]
    assert chunker.flush() == "And tomorrow"


def test_chunker_holds_short_fragments() -> None:
    chunker = SentenceChunker(min_chars=8)
    assert chunker.feed("Hi. ") == []
    assert chunker.feed("How are you today? ") == ["Hi. How are you today?"]


def test_timer_first_mark_wins_and_spans_compute() -> None:
    timer = TurnTimer(backend="local")
    timer.mark("speech_end", 1000.0)
    timer.mark("endpoint", 1200.0)
    timer.mark("llm_first_token", 1350.0)
    timer.mark("tts_request", 1500.0)
    timer.mark("tts_first_audio", 1700.0)
    timer.mark("tts_first_audio", 1900.0)
    ms = timer.stage_ms()
    assert ms["endpoint"] == 200
    assert ms["llm_first_sentence"] == 150
    assert ms["tts_first_audio"] == 200
    assert ms["time_to_first_audio"] == 700
    assert ms["memory_recall"] is None


async def test_agent_bounds_working_memory() -> None:
    llm = FakeLLM("ok")
    agent = Agent(llm, "system", max_turns=2)
    for i in range(5):
        async for _ in agent.respond(f"q{i}"):
            pass
        agent.commit_reply(f"a{i}")
    prompt = llm.prompts[-1]
    assert prompt[0].role == "system"
    assert [m.content for m in prompt[1:]] == ["q2", "a2", "q3", "a3", "q4"]
