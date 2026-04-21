"""
Generation-quality tests for the RAG pipeline.

SDET rationale
--------------
The previous version iterated every golden inside a single test function and
averaged the scores. That hides which case broke and makes bisecting painful.

Each golden is now its own parametrized test:
- failures name the exact question that regressed;
- one bad case does not drag the rest of the suite into pass/fail noise;
- ``pytest -k "cancellation"`` can re-run a single contract.

Reporting is delegated to DeepEval metrics: ``metric.measure()`` assigns a
score and a human-readable ``metric.reason``; the assert carries that reason
when it fails. No manual score aggregation or print-based reporter is needed.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.config import settings
from tests._goldens import RAG_GOLDENS, input_id


# Out-of-scope goldens ("I don't have that information") would trivially score
# 1.0 on faithfulness and near-0 on relevancy for the wrong reason. Filtering
# them out of the generation matrix keeps each parametrized case meaningful.
IN_SCOPE_RAG_GOLDENS: list[dict] = [
    g for g in RAG_GOLDENS if g.get("reference_contexts")
]


@pytest.fixture(scope="session")
def rag_result_cache(rag_pipeline) -> dict:
    """
    Cache ``rag_pipeline.query`` output per ``user_input`` for the session.

    Multiple generation tests score the SAME output from different angles
    (faithfulness + relevancy); without this, each test would trigger a fresh
    LLM call and the scores would apply to different generations — defeating
    any attempt at apples-to-apples comparison.
    """
    return {}


@pytest.fixture
def rag_test_case(rag_pipeline, rag_result_cache):
    """
    Resolve a golden into an ``LLMTestCase``, reusing any cached run so sibling
    tests score the same generation.

    ``retrieval_context`` is intentionally the LIVE retrieved chunks, not the
    golden ``reference_contexts``: faithfulness must be measured against what
    the model actually saw, or a hallucination that happens to match the ideal
    chunks would silently pass.
    """

    def _build(golden: dict) -> LLMTestCase:
        key = golden["user_input"]
        if key not in rag_result_cache:
            rag_result_cache[key] = rag_pipeline.query(key)
        rag_result = rag_result_cache[key]

        return LLMTestCase(
            input=golden["user_input"],
            actual_output=rag_result.answer,
            retrieval_context=rag_result.contexts,
            expected_output=golden.get("reference"),
        )

    return _build


@pytest.mark.parametrize("golden", IN_SCOPE_RAG_GOLDENS, ids=input_id)
def test_faithfulness_per_golden(golden, rag_test_case, deepeval_llm):
    """
    Every claim in the answer must be grounded in the retrieved context.

    Hallucinations are the most damaging failure mode for a clinic assistant
    (wrong dosage, wrong hours). ``metric.reason`` names the ungrounded claims
    when this regresses.
    """
    metric = FaithfulnessMetric(
        threshold=settings.threshold_faithfulness,
        model=deepeval_llm,
    )
    metric.measure(rag_test_case(golden))
    assert metric.success, (
        f"Faithfulness {metric.score:.3f} < {settings.threshold_faithfulness} "
        f"for {golden['user_input']!r}.\n{metric.reason}"
    )


@pytest.mark.parametrize("golden", IN_SCOPE_RAG_GOLDENS, ids=input_id)
def test_answer_relevancy_per_golden(golden, rag_test_case, deepeval_llm):
    """
    The response must actually address the user's question, not merely be
    faithful. Catches grammatical-but-off-topic answers (e.g. dumping a chunk
    instead of answering the specific ask). ``metric.reason`` explains the gap.
    """
    metric = AnswerRelevancyMetric(
        threshold=settings.threshold_answer_relevancy,
        model=deepeval_llm,
    )
    metric.measure(rag_test_case(golden))
    assert metric.success, (
        f"Answer relevancy {metric.score:.3f} < {settings.threshold_answer_relevancy} "
        f"for {golden['user_input']!r}.\n{metric.reason}"
    )
