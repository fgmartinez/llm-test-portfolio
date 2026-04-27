"""
Hybrid retrieval for the clinic knowledge base.

Dense vector search is good at broad semantic similarity, but it can miss exact
clinic facts such as doctor names, weekdays, insurance providers, and fees.
This module blends Chroma vector ranking with a small lexical scorer so exact
terms can lift the right chunk into the final context set.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "have",
    "how",
    "i",
    "if",
    "is",
    "it",
    "me",
    "on",
    "or",
    "our",
    "the",
    "to",
    "day",
    "what",
    "when",
    "which",
    "with",
    "you",
    "same",
}

_SYNONYMS = {
    "doctor": {"doctor", "doctors", "physician", "medicine"},
    "doctors": {"doctor", "doctors", "physician", "medicine"},
    "family": {"family"},
    "accepted": {"accepted", "accept", "insurance", "network"},
    "accept": {"accepted", "accept", "insurance", "network"},
}


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    tokens = [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_SYNONYMS.get(token, ()))
    return expanded


def _doc_id(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or doc.page_content)


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class HybridRetriever:
    """Return focused contexts using vector similarity plus lexical matching."""

    def __init__(self, store: Chroma) -> None:
        self._store = store

    def invoke(self, question: str) -> list[Document]:
        all_docs = self._load_all_docs()
        if not all_docs:
            return []

        by_id = {_doc_id(doc): doc for doc in all_docs}
        vector_scores = self._vector_scores(question, by_id)
        lexical_scores, term_weights = self._lexical_scores(question, all_docs)

        vector_norm = _minmax(vector_scores)
        lexical_norm = _minmax(lexical_scores)

        combined: list[tuple[float, float, float, Document]] = []
        for doc_id, doc in by_id.items():
            vector = vector_norm.get(doc_id, 0.0)
            lexical = lexical_norm.get(doc_id, 0.0)
            score = (0.35 * vector) + (0.65 * lexical)
            combined.append((score, lexical, vector, doc))

        combined.sort(key=lambda item: item[:3], reverse=True)

        return self._select_contexts(question, combined, term_weights)

    def _select_contexts(
        self,
        question: str,
        ranked_docs: list[tuple[float, float, float, Document]],
        term_weights: dict[str, float],
    ) -> list[Document]:
        selected: list[Document] = []
        covered_terms: set[str] = set()
        query_terms = set(_tokens(question))
        rare_terms = {
            term for term, weight in term_weights.items() if term in query_terms and weight >= 1.6
        }
        best_score = ranked_docs[0][0] if ranked_docs else 0.0

        for score, lexical, vector, doc in ranked_docs:
            if len(selected) >= settings.retrieval_k:
                break
            if score <= 0:
                continue

            doc_terms = set(_tokens(doc.page_content))
            newly_covered = (doc_terms & rare_terms) - covered_terms
            first_context = not selected
            strong_followup = bool(newly_covered) and (
                score >= best_score * 0.25 or lexical >= 0.15
            )

            if first_context or strong_followup:
                selected.append(doc)
                covered_terms.update(doc_terms & query_terms)

        return selected or ([ranked_docs[0][3]] if ranked_docs else [])

    def _load_all_docs(self) -> list[Document]:
        rows = self._store.get(include=["documents", "metadatas"])
        documents = rows.get("documents") or []
        metadatas = rows.get("metadatas") or [{} for _ in documents]
        return [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(documents, metadatas)
        ]

    def _vector_scores(self, question: str, by_id: dict[str, Document]) -> dict[str, float]:
        # Chroma distance is lower-is-better, so invert it into higher-is-better.
        vector_hits = self._store.similarity_search_with_score(
            question,
            k=min(len(by_id), max(settings.retrieval_k * 4, 8)),
        )
        return {
            _doc_id(doc): 1.0 / (1.0 + max(distance, 0.0))
            for doc, distance in vector_hits
        }

    def _lexical_scores(
        self, question: str, docs: list[Document]
    ) -> tuple[dict[str, float], dict[str, float]]:
        query_terms = Counter(_tokens(question))
        if not query_terms:
            return {}, {}

        doc_term_sets = [set(_tokens(doc.page_content)) for doc in docs]
        doc_count = max(len(doc_term_sets), 1)
        term_weights = {
            term: math.log((doc_count + 1) / (1 + sum(term in terms for terms in doc_term_sets))) + 1
            for term in query_terms
        }
        total_weight = sum(term_weights[term] * count for term, count in query_terms.items())

        scores: dict[str, float] = {}
        for doc in docs:
            doc_terms = Counter(_tokens(doc.page_content))
            overlap = 0.0
            for term, query_count in query_terms.items():
                if term in doc_terms:
                    overlap += min(query_count, doc_terms[term]) * term_weights[term]
            scores[_doc_id(doc)] = overlap / max(total_weight, 1.0)
        return scores, term_weights
