"""
End-to-end quality tests for the clinic agent.

What this file tests
--------------------
Whether the final answer the agent returns is useful, task-complete, and free
from hallucinations. Tool-choice correctness is covered separately in
``test_tool_calls.py``.

Metrics used (from ``deepeval.metrics``)
-----------------------------------------
* ``TaskCompletionMetric``  — did the agent actually complete the task?
* ``AnswerRelevancyMetric`` — does the output address the user's request?
* ``HallucinationMetric``   — does the output contradict provided context?

Goldens file
------------
``eval/goldens/agent_goldens.json`` — entries with ``user_input``,
``expected_output``, ``expected_tools``, ``context``.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from src.config import settings
from tests._goldens import AGENT_GOLDENS
from tests.agent._helpers import golden_id


@pytest.mark.parametrize("golden", AGENT_GOLDENS, ids=golden_id)
def test_answer_relevancy_per_golden(golden, run_agent_for_golden, deepeval_llm):
    """Output must address the user's request, not just mention related topics."""
    result = run_agent_for_golden(golden)

    test_case = LLMTestCase(
        input=golden["user_input"],
        expected_output=golden["expected_output"],
        actual_output=result.output,
    )
    metric = AnswerRelevancyMetric(
        model=deepeval_llm,
        threshold=settings.threshold_answer_relevancy,
    )
    metric.measure(test_case)
    assert metric.success, (
        f"Answer relevancy {metric.score:.3f} < {settings.threshold_answer_relevancy} "
        f"for {golden['user_input']!r}.\n{metric.reason}"
    )


@pytest.mark.parametrize("golden", AGENT_GOLDENS, ids=golden_id)
def test_task_completion_per_golden(golden, run_agent_for_golden, deepeval_llm):
    """Agent must complete the requested task, not just produce a relevant response."""
    result = run_agent_for_golden(golden)

    test_case = LLMTestCase(
        input=golden["user_input"],
        expected_output=golden["expected_output"],
        actual_output=result.output,
    )
    metric = TaskCompletionMetric(
        model=deepeval_llm,
        threshold=settings.threshold_answer_relevancy,
    )
    metric.measure(test_case)
    assert metric.success, (
        f"Task completion {metric.score:.3f} < {settings.threshold_answer_relevancy} "
        f"for {golden['user_input']!r}.\n{metric.reason}"
    )


@pytest.mark.parametrize("golden", AGENT_GOLDENS, ids=golden_id)
def test_no_hallucination_on_booking(golden, run_agent_for_golden, deepeval_llm):
    """Booking responses must not contradict the confirmation returned by the tool."""
    if "book_appointment" not in golden["expected_tools"]:
        pytest.skip("Not a booking test case")

    result = run_agent_for_golden(golden)

    context = golden.get("context", "")
    context_list = context if isinstance(context, list) else [context] if context else []

    test_case = LLMTestCase(
        input=golden["user_input"],
        expected_output=golden["expected_output"],
        actual_output=result.output,
        context=context_list,
    )
    metric = HallucinationMetric(
        model=deepeval_llm,
        threshold=settings.threshold_hallucination,
    )
    metric.measure(test_case)
    assert metric.success, (
        f"Hallucination score {metric.score:.3f} > {settings.threshold_hallucination} "
        f"for {golden['user_input']!r}.\n{metric.reason}"
    )
