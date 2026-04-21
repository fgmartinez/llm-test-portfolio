"""
Ingestion script: load the clinic markdown document, split it into chunks,
embed each chunk, and persist them inside ChromaDB.

Why a dedicated ingestion step?
-------------------------------
Keeping ingestion separate from querying is standard RAG hygiene and — more
importantly for this portfolio — it allows the testing layer to operate on a
stable, pre-built vector store. Retrieval-quality tests (RAGAS context_recall,
context_precision) must run against the SAME index every time, otherwise the
numbers would drift run-to-run.

Usage:
    python -m src.rag.ingest
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter

from src.config import settings
from src.llm.factory import get_embeddings


def build_vector_store() -> Chroma:
    """Load → split → embed → persist. Returns the Chroma handle."""
    # 1. Load the markdown knowledge base as a single Document.
    loader = TextLoader(str(settings.knowledge_file), encoding="utf-8")
    docs = loader.load()

    # 2. Split. MarkdownTextSplitter keeps headings intact, which noticeably
    #    improves retrieval precision on documents like ours that are
    #    organised by H2 sections (hours, insurance, doctors, ...).
    splitter = MarkdownTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed + persist. We point Chroma at a persistent directory so the
    #    tests reuse the same collection without re-embedding.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.chroma_collection,
        persist_directory=str(settings.chroma_persist_dir),
    )
    return vector_store


def load_vector_store() -> Chroma:
    """Open the persisted store without re-ingesting. Used by the pipeline."""
    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=get_embeddings(),
    )


if __name__ == "__main__":
    store = build_vector_store()
    print(f"Ingested {store._collection.count()} chunks into "  # noqa: SLF001
          f"'{settings.chroma_collection}'.")
