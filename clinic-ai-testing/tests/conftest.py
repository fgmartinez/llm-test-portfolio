"""
Shared pytest fixtures for the Clinic AI Testing suite.

Infrastructure fixtures (``rag_pipeline``, ``clinic_agent``, ``ragas_llm``,
``ragas_embeddings``, ``deepeval_llm``) are all ``scope="session"``: they are
expensive to initialise — the vector store opens a ChromaDB connection, the
LLM opens a network connection — and must be shared across every test that
requests them. Creating a new instance per test would make the suite take hours.

Golden dataset fixtures (``rag_goldens``, ``agent_goldens``, ``safety_goldens``)
are thin wrappers around the module-level lists in ``tests/_goldens.py``. The
lists are the single source of truth: ``@pytest.mark.parametrize`` decorators
import them directly (fixtures are not resolvable at collection time).

Agent run caching (``agent_run_cache``, ``run_agent_for_golden``) ensures each
unique user_input pays the LLM cost only once per session. A single golden may
be scored by several test functions (tool names, argument values, output
quality); the cache lets all of them share one execution.

Golden dataset shape reference
-------------------------------
RAG golden:
    { user_input, reference, reference_contexts }

Agent golden:
    { user_input, expected_output, expected_tools, forbidden_tools,
      expected_tool_calls, context }

Safety golden:
    { user_input, actual_output, expected_output, category }
"""

from pathlib import Path
from typing import Any, Callable

import pytest
from deepeval.models.llms.ollama_model import OllamaModel

from src.config import settings
from src.llm.factory import get_ragas_embeddings, get_ragas_llm_wrapper
from tests._goldens import AGENT_GOLDENS, RAG_GOLDENS, SAFETY_GOLDENS

try:
    from pytest_metadata.plugin import metadata_key
except ImportError:  # pragma: no cover — only needed when pytest-html is installed
    metadata_key = None


def pytest_configure(config):
    """Create the report output directory and populate pytest-html metadata."""
    Path("reports").mkdir(exist_ok=True)

    if metadata_key is None:
        return

    metadata = config.stash[metadata_key]
    metadata["Project"] = "Clinic AI Testing"
    metadata["LLM Provider"] = settings.llm_provider
    metadata["Ollama Agent Model"] = settings.ollama_model
    metadata["Ollama Evaluator Model"] = settings.ollama_evaluator_model
    metadata["Agent Goldens"] = str(len(AGENT_GOLDENS))
    metadata["RAG Goldens"] = str(len(RAG_GOLDENS))
    metadata["Safety Goldens"] = str(len(SAFETY_GOLDENS))


def pytest_html_report_title(report):
    """Override the default pytest-html report title."""
    report.title = "Clinic AI Testing Report"


# ── Infrastructure fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rag_pipeline():
    """
    Single ``RAGPipeline`` instance shared across the entire test session.

    Initialisation sequence:
    1. Opens the persisted ChromaDB collection built by ``python -m src.rag.ingest``.
    2. Constructs a retriever with ``retrieval_k`` from ``src.config.settings``.
    3. Instantiates the LLM for the configured provider (Ollama / Anthropic / OpenAI).

    Pre-condition: ``python -m src.rag.ingest`` must have been run at least once
    before executing retrieval or generation tests.
    """
    from src.rag.pipeline import RAGPipeline

    return RAGPipeline()


@pytest.fixture(scope="session")
def clinic_agent():
    """
    Single ``ClinicAgent`` ReAct instance shared across the entire test session.

    Initialisation sequence:
    1. Builds the ReAct prompt from ``REACT_TEMPLATE`` (tool names + descriptions).
    2. Wires the LLM to the three clinic tools via ``create_react_agent()``.
    3. Wraps everything in an ``AgentExecutor`` with
       ``return_intermediate_steps=True`` so the tool trace is available for
       ``ToolCorrectnessMetric`` assertions.
    """
    from src.agent.agent import ClinicAgent

    return ClinicAgent()


@pytest.fixture(scope="session")
def ragas_llm():
    """
    ``LangchainLLMWrapper`` around the configured chat model.

    RAGAS metrics that require an LLM judge (``LLMContextRecall``,
    ``LLMContextPrecision``, ``NoiseSensitivity``, ``Faithfulness``) consume
    this wrapper rather than a raw LangChain ``BaseChatModel``.
    """
    return get_ragas_llm_wrapper()


@pytest.fixture(scope="session")
def ragas_embeddings():
    """
    ``LangchainEmbeddingsWrapper`` around the configured embeddings model.

    ``AnswerRelevancy`` uses cosine similarity between the question and answer
    vectors, so it requires the same embedding function used during ingestion.
    """
    return get_ragas_embeddings()


@pytest.fixture(scope="session")
def deepeval_llm():
    """
    ``OllamaModel`` judge for DeepEval metrics.

    Pass as the ``model`` parameter to ``FaithfulnessMetric``,
    ``AnswerRelevancyMetric``, ``BiasMetric``, ``TaskCompletionMetric``,
    ``HallucinationMetric``, and ``GEval`` to run all evaluations locally
    through Ollama without requiring an OpenAI API key.

    This intentionally uses ``OLLAMA_EVALUATOR_MODEL`` rather than the
    application model so agent quality can be judged by a different local LLM.
    """
    return OllamaModel(
        model=settings.ollama_evaluator_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


# ── Golden dataset fixtures ────────────────────────────────────────────────────
# Goldens are the manually curated ground-truth dataset for this project.
# Each golden is an (input → expected output) pair authored by a domain expert
# and is the foundation for every quantitative evaluation.
#
# The lists are loaded once at import time in ``tests/_goldens.py``. These
# fixtures expose that single source so individual tests can dependency-inject
# the dataset. ``@pytest.mark.parametrize`` decorators should import the
# module-level constants directly — fixtures are not resolvable at collection time.


@pytest.fixture(scope="session")
def rag_goldens() -> list[dict]:
    """Session-scoped fixture wrapping ``tests._goldens.RAG_GOLDENS``."""
    return RAG_GOLDENS


@pytest.fixture(scope="session")
def agent_goldens() -> list[dict]:
    """Session-scoped fixture wrapping ``tests._goldens.AGENT_GOLDENS``."""
    return AGENT_GOLDENS


@pytest.fixture(scope="session")
def safety_goldens() -> list[dict]:
    """Session-scoped fixture wrapping ``tests._goldens.SAFETY_GOLDENS``."""
    return SAFETY_GOLDENS


# ── Agent run cache ────────────────────────────────────────────────────────────
# Each golden triggers a real LLM call. A single golden may be scored by
# several test functions (tool names, argument values, output quality, safety).
# The cache ensures each unique user_input pays the LLM cost only once per session.


@pytest.fixture(scope="session")
def agent_run_cache() -> dict[str, Any]:
    """Session-scoped dict keyed by ``user_input`` string."""
    return {}


@pytest.fixture
def run_agent_for_golden(
    clinic_agent, agent_run_cache: dict[str, Any]
) -> Callable[[dict], Any]:
    """
    Return a callable that executes ``clinic_agent`` once per unique input.

    Subsequent calls for the same ``user_input`` return the cached
    ``AgentResult``, so sibling test functions score the same run without
    incurring additional LLM costs.

    Pass ``-s`` / ``--capture=no`` to pytest to see the per-run tool trace
    printed to stdout after each execution.
    """
    from tests.agent._helpers import format_tool_trace

    def _run(golden: dict) -> Any:
        key = golden["user_input"]
        if key not in agent_run_cache:
            agent_run_cache[key] = clinic_agent.run(key)
        result = agent_run_cache[key]
        print(format_tool_trace(golden, result))
        return result

    return _run
