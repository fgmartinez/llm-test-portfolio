"""
Helpers for the clinic agent tool-routing contract tests.

Why a separate module?
----------------------
The test file should read as a spec: "for each golden, assert the route".
Slug/ID generation, DeepEval adapters, and argument matching are
implementation details that clutter that spec, so they live here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from deepeval.test_case import ToolCall

from tests._goldens import input_id


def golden_id(golden: dict) -> str:
    """Stable, readable pytest ID per golden — enables ``pytest -k``."""
    expected = (
        "no-tools"
        if not golden["expected_tools"]
        else "->".join(golden["expected_tools"])
    )
    return f"{input_id(golden, limit=55)}__{expected}"


def to_deepeval_tool_calls(tool_calls: list) -> list[ToolCall]:
    """Adapt internal ToolInvocation objects to DeepEval's ToolCall shape."""
    return [
        ToolCall(
            name=call.name,
            input_parameters=(
                call.input if isinstance(call.input, dict) else {"raw_input": call.input}
            ),
            output=call.observation,
        )
        for call in tool_calls
    ]


def _normalise(value: Any) -> str:
    text = str(value).lower().replace("dr.", "dr")
    return re.sub(r"[^a-z0-9]+", "", text)


def argument_matches(actual: Any, expected_key: str, expected_value: Any) -> bool:
    """
    Match expected tool args resiliently across LangChain/ReAct input shapes.

    LangChain may surface ``tool_input`` as a dict for structured calls or as
    a raw string for ReAct text inputs. Prefer key-specific matching when a
    dict is available; fall back to searching the raw payload.
    """
    expected = _normalise(expected_value)

    if isinstance(actual, dict):
        if expected_key in actual:
            return expected in _normalise(actual[expected_key])
        return any(expected in _normalise(value) for value in actual.values())

    return expected in _normalise(actual)


def format_tool_trace(golden: dict, result) -> str:
    """
    Human-readable trace of a single agent run — intended for ``-s`` output.

    Shows the input, every tool invocation with its args and observation, and
    the final answer so the SDET can verify routing decisions at a glance.
    """
    separator = "-" * 72
    lines = [
        "",
        separator,
        f"INPUT   : {golden['user_input']}",
        f"EXPECTED: {golden.get('expected_tools', '(not asserted here)')}",
        f"ACTUAL  : {result.tools_called}",
    ]

    if result.tool_calls:
        lines.append("TRACE   :")
        for i, call in enumerate(result.tool_calls, start=1):
            args = (
                json.dumps(call.input, ensure_ascii=False)
                if isinstance(call.input, dict)
                else str(call.input)
            )
            lines.append(f"  [{i}] {call.name}")
            lines.append(f"       args        : {args}")
            lines.append(f"       observation : {call.observation}")
    else:
        lines.append("TRACE   : (no tools called)")

    lines.append(f"OUTPUT  : {result.output}")
    lines.append(separator)
    return "\n".join(lines)


def assert_expected_arguments(golden: dict, result) -> None:
    """
    Verify every expected tool call carries the expected argument values.

    Tool-name correctness is necessary but not sufficient: a booking tool
    called with the wrong doctor is still a routing failure.
    """
    expected_calls = golden.get("expected_tool_calls", [])
    if not expected_calls:
        return

    assert len(result.tool_calls) >= len(expected_calls), (
        f"Expected at least {len(expected_calls)} tool call(s), "
        f"got {len(result.tool_calls)}. Actual: {result.tools_called}"
    )

    for expected_call, actual_call in zip(
        expected_calls, result.tool_calls, strict=False
    ):
        expected_args = expected_call.get("expected_args", {})
        if not expected_args:
            continue

        missing_or_wrong = {
            key: value
            for key, value in expected_args.items()
            if not argument_matches(actual_call.input, key, value)
        }
        assert not missing_or_wrong, (
            f"Argument mismatch on {actual_call.name}. "
            f"Expected {expected_args}, got {actual_call.input}. "
            f"Missing/wrong: {missing_or_wrong}"
        )
