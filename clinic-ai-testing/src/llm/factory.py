"""
Multi-provider LLM + Embeddings factory.

Why this exists
---------------
A QA-focused portfolio must demonstrate how evaluation metrics behave across
DIFFERENT model providers. Rather than hard-coding any single provider, this
factory centralises provider selection and returns three artifacts the rest of
the code needs:

  1. `get_llm()`               → a LangChain BaseChatModel (for RAG + agent)
  2. `get_ragas_llm_wrapper()` → LangchainLLMWrapper (RAGAS needs this type)
  3. `get_ragas_embeddings()`  → LangchainEmbeddingsWrapper (RAGAS needs this)

Ollama is always the safe default so the suite can run offline with no keys.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from src.config import settings


# ── Private builders ────────────────────────────────────────────────────────
# Each provider has its own builder. Keeping them isolated makes it trivial to
# add a fourth provider later (e.g. Gemini) without touching the public API.

def _build_ollama() -> tuple[BaseChatModel, Embeddings]:
    # Imported lazily so users without Ollama installed don't pay the cost.
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,  # deterministic answers → stable test assertions
    )
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    return llm, embeddings


def _build_anthropic() -> tuple[BaseChatModel, Embeddings]:
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY missing — set it in .env")

    llm = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
    )
    # Anthropic does not provide first-party embeddings, so we fall back to
    # Ollama embeddings. This is fine because embeddings are provider-agnostic
    # for our retrieval quality tests.
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    return llm, embeddings


def _build_openai() -> tuple[BaseChatModel, Embeddings]:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY missing — set it in .env")

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    return llm, embeddings


_BUILDERS = {
    "ollama": _build_ollama,
    "anthropic": _build_anthropic,
    "openai": _build_openai,
}


# ── Public API ──────────────────────────────────────────────────────────────
# Cached so the rest of the code can call these freely without re-instantiating
# network clients on every invocation.

@lru_cache(maxsize=1)
def _build() -> tuple[BaseChatModel, Embeddings]:
    """Build LLM + Embeddings for the provider specified in settings."""
    provider = settings.llm_provider
    if provider not in _BUILDERS:
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
    return _BUILDERS[provider]()


def get_llm() -> BaseChatModel:
    """Return the raw LangChain chat model for the agent / RAG chain."""
    llm, _ = _build()
    return llm


def get_embeddings() -> Embeddings:
    """Return the raw LangChain embeddings model for Chroma ingestion."""
    _, emb = _build()
    return emb


def get_ragas_llm_wrapper() -> LangchainLLMWrapper:
    """Wrap the LLM so RAGAS metrics can consume it."""
    return LangchainLLMWrapper(get_llm())


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Wrap embeddings so RAGAS metrics (e.g. AnswerRelevancy) can use them."""
    return LangchainEmbeddingsWrapper(get_embeddings())
