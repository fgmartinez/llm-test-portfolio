"""
Minimal 3-step RAG pipeline: embed question → retrieve top-k → generate answer.

Design notes
------------
* No query rewriting, no re-ranking, no conversational memory. The whole point
  of the portfolio is to evaluate a PLAIN pipeline so metric differences reveal
  the value-add (or regression) of each future improvement.
* `RAGResult` exposes `question`, `answer`, `contexts` because those are the
  exact field names RAGAS expects when building a `SingleTurnSample`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.llm.factory import get_llm
from src.rag.ingest import load_vector_store
from src.rag.retrieval import HybridRetriever


@dataclass
class RAGResult:
    """Shape that RAGAS metrics can consume directly."""

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a helpful assistant for a medical clinic. "
    "Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say you don't have that "
    "information. For medication or treatment questions, always remind "
    "the user to consult their doctor."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


class RAGPipeline:
    """Thin wrapper around retriever + LLM. Stateless and easy to mock in tests."""

    def __init__(self) -> None:
        # Lazy-load the persisted Chroma store so the pipeline can be constructed
        # inside test fixtures even if ingestion hasn't been re-run.
        self._store = load_vector_store()
        self._retriever = HybridRetriever(self._store)
        self._llm = get_llm()

    def retrieve_contexts(self, question: str) -> list[str]:
        """Return only retrieved contexts, without paying generation cost."""
        docs = self._retriever.invoke(question)
        return [d.page_content for d in docs]

    def query(self, question: str) -> RAGResult:
        """Run the 3-step chain and return a RAGResult."""
        # 1. Retrieve
        contexts = self.retrieve_contexts(question)

        # 2. Assemble prompt
        prompt_value = _PROMPT.format_prompt(
            context="\n\n---\n\n".join(contexts),
            question=question,
        )

        # 3. Generate
        response = self._llm.invoke(prompt_value.to_messages())
        answer = response.content if hasattr(response, "content") else str(response)

        return RAGResult(question=question, answer=answer, contexts=contexts)
