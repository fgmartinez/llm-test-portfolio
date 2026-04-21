"""
Bias tests for the clinic agent / RAG pipeline.

What this file tests
--------------------
That responses do NOT contain biased language when users phrase questions
with gender, age, or demographic markers. We want the model to reject biased
premises and give neutral, equal-treatment answers.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase

from src.config import settings
from tests._goldens import SAFETY_GOLDENS, input_id


BIAS_GOLDENS = [
    golden for golden in SAFETY_GOLDENS if golden.get("category") == "bias"
]


@pytest.mark.parametrize("golden", BIAS_GOLDENS, ids=input_id)
def test_bias_on_agent_output(golden, run_agent_for_golden, deepeval_llm):
    result = run_agent_for_golden(golden)

    test_case = LLMTestCase(
        input=golden["user_input"],
        expected_output=golden["expected_output"],
        actual_output=result.output,
    )

    metric = BiasMetric(
        threshold=settings.threshold_bias,
        model=deepeval_llm,
    )
    metric.measure(test_case)

    assert metric.success, (
        f"Bias score {metric.score:.3f} > {settings.threshold_bias} for "
        f"{golden['user_input']!r}.\n{metric.reason}"
    )
