import pytest
from mcp import Client

from murmur.agent.tools import McpToolBox, offered
from murmur.llm.base import ToolSpec
from murmur.mcp_servers.tools import build_tools_server, evaluate, spoken_number

SPECS = [ToolSpec(n, "", {}) for n in ("forget", "current_time", "calculate")]


@pytest.mark.parametrize(
    ("expression", "value"),
    [("18.5 * 4 / 2", 37.0), ("2^10", 1024.0), ("sqrt(144) + 1", 13.0), ("-3 + 1,000", 997.0)],
)
def test_calculator_does_arithmetic(expression: str, value: float) -> None:
    assert evaluate(expression) == value


@pytest.mark.parametrize("expression", ["__import__('os')", "open('x')", "2 ** 999999", "a + 1"])
def test_calculator_refuses_anything_but_arithmetic(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate(expression)


def test_numbers_read_aloud_cleanly() -> None:
    assert spoken_number(37.0) == "37"
    assert spoken_number(1 / 3) == "0.333333"


@pytest.mark.parametrize(
    ("said", "tools"),
    [
        ("I play badminton on weekdays", []),
        ("what time is it in London", ["current_time"]),
        ("what's 18 times 4", ["calculate"]),
        ("what's 12 * 7", ["calculate"]),
        ("please forget my address", ["forget"]),
    ],
)
def test_tools_are_offered_only_when_the_words_call_for_them(said: str, tools: list[str]) -> None:
    assert [s.name for s in offered(SPECS, said)] == tools


async def test_toolbox_routes_calls_to_the_right_server() -> None:
    async with Client(build_tools_server("Asia/Kolkata")) as client:
        box = await McpToolBox.connect([client], {"calculate", "current_time"})
        assert {s.name for s in box.specs} == {"calculate", "current_time"}
        assert await box.call("calculate", {"expression": "6 * 7"}) == "42"
        assert "Asia/Kolkata" in await box.call("current_time", {})
        assert "Unknown timezone" in await box.call("current_time", {"timezone": "Mars/Base"})
        assert await box.call("forget", {"description": "x"}) == "Unknown tool forget."
