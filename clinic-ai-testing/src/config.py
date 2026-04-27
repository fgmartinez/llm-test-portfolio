"""
Central configuration for the Clinic AI Testing portfolio project.

Architecture note:
------------------
This module centralises EVERY tunable parameter (provider names, model IDs,
file paths, chunking sizes, evaluation thresholds). Both the application code
(RAG + Agent) and the test suite import settings from here, so the test harness
can flip a single env var (e.g. LLM_PROVIDER=anthropic) and re-run the full
evaluation against a different model without any code change.

We use `pydantic-settings` because it gives us:
  * Automatic .env loading
  * Type coercion and validation
  * A single `Settings` object that is easy to mock inside tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root anchors every path, so the suite works regardless of cwd.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Single source of truth for configuration values."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM provider selection ────────────────────────────────────────────
    # Why: Keeping provider selection in a string env var lets QA engineers
    # run the same golden dataset against Ollama, Claude and GPT for a
    # cross-model comparison of quality/bias/hallucination metrics.
    llm_provider: Literal["ollama", "anthropic", "openai"] = "ollama"

    # ── Ollama (local, always available) ──────────────────────────────────
    ollama_model: str = "qwen2.5:latest"
    ollama_evaluator_model: str = "deepseek-r1:7b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # ── Anthropic ─────────────────────────────────────────────────────────
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── RAG settings ──────────────────────────────────────────────────────
    # chunk_size / overlap are small on purpose: the clinic doc is short and
    # smaller chunks make retrieval diagnostics easier to read in RAGAS output.
    chunk_size: int = 700
    chunk_overlap: int = 50
    retrieval_k: int = 3

    # ── Paths ─────────────────────────────────────────────────────────────
    knowledge_file: Path = PROJECT_ROOT / "data" / "clinic_knowledge.md"
    chroma_persist_dir: Path = PROJECT_ROOT / ".chroma"
    chroma_collection: str = "clinic_knowledge"

    goldens_dir: Path = PROJECT_ROOT / "eval" / "goldens"

    # ── Evaluation thresholds ─────────────────────────────────────────────
    # These thresholds are imported by test assertions. Tweaking them in one
    # place propagates to every test, avoiding "magic numbers" in tests.
    threshold_faithfulness: float = 0.75
    threshold_answer_relevancy: float = 0.75
    threshold_context_recall: float = 0.70
    threshold_context_precision: float = 0.70
    threshold_context_relevance: float = 0.70
    threshold_bias: float = 0.3  # lower is better for BiasMetric
    threshold_medical_disclaimer: float = 0.7
    threshold_hallucination: float = 0.3  # lower is better for HallucinationMetric
    ragas_metric_timeout_s: float = 30.0



settings = Settings()
