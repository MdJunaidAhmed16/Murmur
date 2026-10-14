from typing import Any

from murmur.agent.loop import Agent
from murmur.audio.vad import EndpointDetector, VadEvent
from murmur.llm.base import ToolCall, ToolSpec
from murmur.pipeline.metrics import TurnTimer
from murmur.pipeline.playback import PlaybackLedger
from murmur.tts.text import SentenceChunker
from tests.fakes import FakeLLM


def test_endpoint_needs_sustained_speech_then_silence() -> None:
    # 3 and 2 frames; onset equal to min_speech means no separate onset.
    detector = EndpointDetector(endpoint_silence_ms=96, min_speech_ms=64, onset_ms=64)
    events = [detector.update(s) for s in [True, True, False, False, True, False, False, False]]
    assert events == [None, VadEvent.SPEECH_START, None, None, None, None, None, VadEvent.ENDPOINT]


def test_onset_comes_first_and_aborts_if_speech_never_confirms() -> None:
    detector = EndpointDetector(endpoint_silence_ms=96, min_speech_ms=128, onset_ms=64)
    cough = [detector.update(s) for s in [True, True, True, False]]
    assert cough == [None, VadEvent.ONSET, None, VadEvent.ABORT]
    talk = [detector.update(s) for s in [True, True, True, True, False]]
    assert talk == [None, VadEvent.ONSET, None, VadEvent.SPEECH_START, None]


def test_chunker_cuts_the_first_long_sentence_at_a_clause() -> None:
    chunker = SentenceChunker(first_clause=True)
    out: list[str] = []
    for delta in [
        "Well, ",
        "the quickest way to get there, ",
        "honestly, is the metro. ",
        "Next, ",
    ]:
        out += chunker.feed(delta)
    out += chunker.feed("take line two, then walk. ")
    # "Well," is too short to stand alone; later chunks stay whole sentences.
    assert out == [
        "Well, the quickest way to get there,",
        "honestly, is the metro.",
        "Next, take line two, then walk.",
    ]


def test_playback_ledger_maps_position_to_words() -> None:
    ledger = PlaybackLedger(sample_rate=24_000, session_offset=1000)
    ledger.begin("One two three four.")
    ledger.add_audio(400)
    ledger.begin("Five six.")
    ledger.add_audio(200)

    assert ledger.heard_text(1000) == ""
    assert ledger.heard_text(1200) == "One two"
    assert ledger.heard_text(1400) == "One two three four."
    assert ledger.heard_text(1450) == "One two three four."  # a quarter of two words rounds down
    assert ledger.heard_text(1600) == "One two three four. Five six."
    assert ledger.sent_samples == 600
    ledger.begin("Seven.")  # synthesis started, no audio sent yet
    assert ledger.heard_text(1600) == "One two three four. Five six."


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


class RecordingToolBox:
    def __init__(self) -> None:
        self.specs = [ToolSpec("forget", "Forget a fact.", {"type": "object", "properties": {}})]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "Forgotten: Plays badminton"


async def test_agent_runs_a_tool_then_speaks_the_follow_up() -> None:
    call = ToolCall("call_1", "forget", {"description": "plays badminton"})
    llm = FakeLLM(scripted=[[call], ["Done, ", "I have forgotten that."]])
    tools = RecordingToolBox()
    agent = Agent(llm, max_turns=2, tools=tools)

    spoken = [d async for d in agent.respond("forget that I play badminton", [])]

    assert "".join(spoken) == "Done, I have forgotten that."
    assert tools.calls == [("forget", {"description": "plays badminton"})]
    assert agent.last_tool_calls == ["forget"]
    follow_up = llm.prompts[1]
    assert follow_up[-1].role == "tool" and follow_up[-1].tool_call_id == "call_1"
    assert llm.tools_offered[1] is None  # no second tool round


async def test_forget_is_only_offered_when_asked_for() -> None:
    llm = FakeLLM(scripted=[["Nice."], ["Sure."]])
    agent = Agent(llm, max_turns=2, tools=RecordingToolBox())

    async for _ in agent.respond("I play badminton on weekday evenings", []):
        pass
    async for _ in agent.respond("please forget that", []):
        pass

    assert llm.tools_offered[0] == []
    assert [s.name for s in llm.tools_offered[1] or []] == ["forget"]


async def test_forgetting_scrubs_the_session_transcript_too() -> None:
    call = ToolCall("call_1", "forget", {"description": "badminton"})
    llm = FakeLLM(scripted=[["Great, badminton is fun."], [call], ["Done."], ["Not sure."]])
    agent = Agent(llm, max_turns=4, tools=RecordingToolBox())

    for text in ["I play badminton on weekdays", "forget that I play badminton"]:
        async for _ in agent.respond(text, []):
            pass
        agent.commit_reply("ok")

    remaining = " ".join(m.content for m in agent.history)
    assert "badminton" not in remaining.replace("forget that I play badminton", "")


async def test_agent_bounds_working_memory() -> None:
    llm = FakeLLM("ok")
    agent = Agent(llm, max_turns=2)
    for i in range(5):
        async for _ in agent.respond(f"q{i}", []):
            pass
        agent.commit_reply(f"a{i}")
    prompt = llm.prompts[-1]
    assert prompt[0].role == "system"
    assert [m.content for m in prompt[1:]] == ["q2", "a2", "q3", "a3", "q4"]
