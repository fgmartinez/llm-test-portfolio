"""
Retrieval-layer tests for the RAG pipeline.

SDET rationale
--------------
Previously these tests ran each metric once over the *whole* golden set and
asserted the averaged score. Aggregated asserts hide which query regressed
and make it impossible to re-run the one that broke. The suite now fans out
one pytest case per golden — pytest's native failure reporter already names
the failing case and surfaces the score diff, so no hand-rolled formatter is
needed.

Why non-LLM scorers?
--------------------
LLM-based RAGAS metrics are brittle with local Ollama: format-repair loops
stall retrieval tests and turn them into infrastructure tests. The curated
``reference_contexts`` in the goldens are exact canonical snippets, so
partial string matching is both faster and more stable here.
"""

from __future__ import annotations

import re

import pytest
from rapidfuzz import fuzz
from ragas import SingleTurnSample

from src.config import settings
from tests._goldens import RAG_GOLDENS, input_id


# Only goldens with curated reference snippets are scorable: without them
# there is no ground truth to match retrieved chunks against.
SCORABLE_RAG_GOLDENS: list[dict] = [
    g for g in RAG_GOLDENS if g.get("reference_contexts")
]

# A match threshold of 0.95 on partial-ratio requires near-verbatim overlap,
# which is what we want for a reference snippet curated to appear inside a
# retrieved chunk.
_MATCH_THRESHOLD = 0.95


def _normalize(text: str) -> str:
    """Strip markdown/whitespace noise so fuzzy matching stays stable."""
    text = text.lower().replace("**", "")
    text = text.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()


def _partial_match_score(left: str, right: str) -> float:
    return fuzz.partial_ratio(_normalize(left), _normalize(right)) / 100.0


def _best_reference_match(sample: SingleTurnSample, candidate: str) -> float:
    return max(
        (_partial_match_score(candidate, ref) for ref in sample.reference_contexts or []),
        default=0.0,
    )


def _context_recall(sample: SingleTurnSample) -> float:
    """Fraction of curated references found inside any retrieved chunk."""
    references = sample.reference_contexts or []
    if not references:
        return 0.0

    matched = sum(
        1
        for ref in references
        if max(
            (_partial_match_score(ref, ret) for ret in sample.retrieved_contexts or []),
            default=0.0,
        )
        >= _MATCH_THRESHOLD
    )
    return matched / len(references)


def _context_precision(sample: SingleTurnSample) -> float:
    """Fraction of retrieved chunks containing at least one curated reference."""
    retrieved = sample.retrieved_contexts or []
    if not retrieved:
        return 0.0
    matched = sum(
        1 for chunk in retrieved if _best_reference_match(sample, chunk) >= _MATCH_THRESHOLD
    )
    return matched / len(retrieved)


def _context_relevance(sample: SingleTurnSample) -> float:
    """Best alignment between any retrieved chunk and any curated reference."""
    return max(
        (_best_reference_match(sample, chunk) for chunk in sample.retrieved_contexts or []),
        default=0.0,
    )


@pytest.fixture(scope="session")
def retrieval_sample_cache(rag_pipeline) -> dict[str, SingleTurnSample]:
    """
    Cache one ``SingleTurnSample`` per golden so all retrieval tests score
    against the SAME retrieved chunks. Different scorers over different
    retrievals would not be comparable.
    """
    return {}


@pytest.fixture
def retrieval_sample(rag_pipeline, retrieval_sample_cache):
    def _build(golden: dict) -> SingleTurnSample:
        key = golden["user_input"]
        if key not in retrieval_sample_cache:
            retrieval_sample_cache[key] = SingleTurnSample(
                user_input=golden["user_input"],
                reference=golden["reference"],
                reference_contexts=golden.get("reference_contexts", []),
                retrieved_contexts=rag_pipeline.retrieve_contexts(golden["user_input"]),
                response="",
            )
        return retrieval_sample_cache[key]

    return _build


@pytest.mark.parametrize("golden", SCORABLE_RAG_GOLDENS, ids=input_id)
def test_context_recall_per_golden(golden, retrieval_sample):
    """
    Did the retriever surface the curated evidence needed to answer?

    First-line guardrail for missing chunks caused by bad embeddings, poor
    chunk boundaries, or an undersized ``retrieval_k``.
    """
    sample = retrieval_sample(golden)
    score = _context_recall(sample)

    assert score >= settings.threshold_context_recall, (
        f"Recall {score:.3f} < {settings.threshold_context_recall}. "
        f"Check chunk_size={settings.chunk_size}, "
        f"chunk_overlap={settings.chunk_overlap}, "
        f"retrieval_k={settings.retrieval_k}."
    )


@pytest.mark.parametrize("golden", SCORABLE_RAG_GOLDENS, ids=input_id)
def test_context_precision_per_golden(golden, retrieval_sample):
    """
    Are the retrieved chunks mostly relevant, or is the set noisy?

    Catches noisy retrieval and bad ranking without depending on an LLM-as-
    judge that may be unstable locally.
    """
    sample = retrieval_sample(golden)
    score = _context_precision(sample)

    assert score >= settings.threshold_context_precision, (
        f"Precision {score:.3f} < {settings.threshold_context_precision}. "
        f"Consider reducing retrieval_k={settings.retrieval_k}, tuning chunk "
        f"size, or adding re-ranking."
    )


@pytest.mark.parametrize("golden", SCORABLE_RAG_GOLDENS, ids=input_id)
def test_context_relevance_per_golden(golden, retrieval_sample):
    """
    Does retrieval surface at least one chunk aligned with the intended
    answer? Lighter-weight than recall and independent of ranking depth.
    """
    sample = retrieval_sample(golden)
    score = _context_relevance(sample)

    assert score >= settings.threshold_context_relevance, (
        f"Relevance {score:.3f} < {settings.threshold_context_relevance}. "
        "Retrieval is surfacing chunks that do not closely match the curated "
        "references."
    )
