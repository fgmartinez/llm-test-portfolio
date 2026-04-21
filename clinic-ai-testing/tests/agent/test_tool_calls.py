"""
Tool-call contract tests for the clinic ReAct agent.

SDET rationale
--------------
Tool selection is a routing contract, not a fuzzy answer-quality check. A
single wrong tool call can book when the user only asked a question, skip a
required availability check, or hide a bad action behind a plausible response.

For that reason these tests are parametrized at golden level:
- each golden is independently selectable with ``pytest -k``;
- failures point to the exact scenario, input, expected route and actual trace;
- one bad case does not obscure the rest of the tool-routing matrix.

Reporting is delegated to ``ToolCorrectnessMetric``: when a test fails,
``metric.reason`` is the library-generated diagnostic (expected vs actual tools,
score). No hand-rolled formatter is needed.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from tests._goldens import AGENT_GOLDENS
from tests.agent._helpers import (
    assert_expected_arguments,
    golden_id,
    to_deepeval_tool_calls,
)


def _as_llm_test_case(golden: dict, result) -> LLMTestCase:
    return LLMTestCase(
        input=golden["user_input"],
        actual_output=result.output,
        expected_output=golden["expected_output"],
        tools_called=to_deepeval_tool_calls(result.tool_calls),
        expected_tools=[ToolCall(name=name) for name in golden["expected_tools"]],
    )


def _assert_tool_correctness(golden: dict, result) -> None:
    """Score the run with ToolCorrectnessMetric and require the exact route."""
    assert result.tools_called == golden["expected_tools"], (
        f"Unexpected tool sequence for {golden['user_input']!r}.\n"
        f"Expected: {golden['expected_tools']}  Actual: {result.tools_called}"
    )

    metric = ToolCorrectnessMetric()
    metric.measure(_as_llm_test_case(golden, result))
    assert metric.success, (
        f"Tool routing failed for {golden['user_input']!r}.\n"
        f"Expected: {golden['expected_tools']}  Actual: {result.tools_called}\n"
        f"DeepEval reason: {metric.reason}"
    )


@pytest.mark.parametrize("golden", AGENT_GOLDENS, ids=golden_id)
def test_expected_tool_sequence_per_golden(run_agent_for_golden, golden):
    """
    Each golden is a standalone routing contract.

    ``ToolCorrectnessMetric`` compares expected vs actual tool sequences.
    ``metric.reason`` surfaces why it failed without a custom formatter.
    """
    result = run_agent_for_golden(golden)
    _assert_tool_correctness(golden, result)


@pytest.mark.parametrize(
    "golden",
    [g for g in AGENT_GOLDENS if g.get("forbidden_tools")],
    ids=golden_id,
)
def test_forbidden_tools_are_never_called(run_agent_for_golden, golden):
    """Scenario-level safety guard: forbidden tools must never be called."""
    result = run_agent_for_golden(golden)
    forbidden = set(golden["forbidden_tools"])
    unexpected = forbidden.intersection(result.tools_called)

    assert not unexpected, (
        f"Forbidden tools called for {golden['user_input']!r}.\n"
        f"Forbidden: {sorted(forbidden)}  Actual: {result.tools_called}"
    )


@pytest.mark.parametrize(
    "golden",
    [g for g in AGENT_GOLDENS if g.get("expected_tool_calls")],
    ids=golden_id,
)
def test_expected_tool_arguments_per_golden(run_agent_for_golden, golden):
    """
    Tool names are necessary but not sufficient: booking/availability flows
    must also carry the right argument values (doctor, date, time, patient).
    """
    result = run_agent_for_golden(golden)
    _assert_tool_correctness(golden, result)
    assert_expected_arguments(golden, result)
