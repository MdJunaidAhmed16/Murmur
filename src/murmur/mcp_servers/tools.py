"""The tools MCP server: small on purpose (docs/04: two or three tools).

current_time covers what the model cannot know; calculate covers what a 1.5B model gets wrong.
Standalone: `python -m murmur.mcp_servers.tools` serves them over stdio.
"""

import ast
import math
import operator
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.mcpserver import MCPServer

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCTIONS: dict[str, Callable[[float], float]] = {"sqrt": math.sqrt}
# The multiplication and division signs, which STT sometimes writes for "times" and "divided by".
TIMES, DIVIDE = chr(0xD7), chr(0xF7)
# Big exponents are a cheap way to stall the event loop; nobody asks a voice agent for 9**9**9.
_MAX_EXPONENT = 100


def evaluate(expression: str) -> float:
    """Arithmetic only, by walking the AST. Never eval(): the text comes from a model."""

    def walk(node: ast.AST) -> float:
        match node:
            case ast.Constant(value=int() | float() as value) if not isinstance(value, bool):
                return float(value)
            case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
                lhs, rhs = walk(left), walk(right)
                if isinstance(op, ast.Pow) and abs(rhs) > _MAX_EXPONENT:
                    raise ValueError("exponent too large")
                return _BINARY[type(op)](lhs, rhs)
            case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
                return _UNARY[type(op)](walk(operand))
            case ast.Call(func=ast.Name(id=name), args=[arg]) if name in _FUNCTIONS:
                return _FUNCTIONS[name](walk(arg))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")

    # Commas are thousands separators.
    cleaned = (
        expression.replace("^", "**").replace(TIMES, "*").replace(DIVIDE, "/").replace(",", "")
    )
    return walk(ast.parse(cleaned, mode="eval").body)


def spoken_number(value: float) -> str:
    # Read aloud, so no trailing ".0" and no 15-digit floats.
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def build_tools_server(default_timezone: str) -> MCPServer:
    server = MCPServer("murmur-tools", instructions="Small utilities for the Murmur voice agent.")

    @server.tool()
    def current_time(timezone: str = "") -> str:
        """The current date and time. Optional IANA timezone such as "Europe/London"."""
        try:
            zone = ZoneInfo(timezone or default_timezone)
        except (ZoneInfoNotFoundError, ValueError):
            return f"Unknown timezone '{timezone}'."
        now = datetime.now(zone)
        return now.strftime(f"%A %-d %B %Y, %-I:%M %p ({zone.key})")

    @server.tool()
    def calculate(expression: str) -> str:
        """Evaluate arithmetic exactly, e.g. "18.5 * 4 / 3" or "sqrt(144)". Use for any maths."""
        try:
            return spoken_number(evaluate(expression))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return f"Could not calculate '{expression}': {exc}"

    return server


def main() -> None:
    from murmur.config import get_settings

    build_tools_server(get_settings().timezone).run("stdio")


if __name__ == "__main__":
    main()
