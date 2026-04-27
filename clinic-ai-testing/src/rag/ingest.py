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
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import settings
from src.llm.factory import get_embeddings


def _split_clinic_markdown(markdown: str) -> list[Document]:
    """Split clinic knowledge into answer-sized markdown sections."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=False,
    )
    section_docs = header_splitter.split_text(markdown)

    # Keep short sections intact, but split any future long section at paragraph
    # boundaries. This prevents one broad "doctors" chunk from hiding individual
    # doctor availability facts.
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n- ", "\n", " "],
    )

    chunks: list[Document] = []
    for doc in section_docs:
        if len(doc.page_content) <= settings.chunk_size:
            chunks.append(doc)
            continue
        chunks.extend(fallback_splitter.split_documents([doc]))

    for index, chunk in enumerate(chunks):
        section = chunk.metadata.get("section", "root")
        subsection = chunk.metadata.get("subsection", "")
        chunk.metadata["chunk_id"] = f"{index:03d}:{section}:{subsection}"
    return chunks


def build_vector_store() -> Chroma:
    """Load → split → embed → persist. Returns the Chroma handle."""
    # 1. Load the markdown knowledge base as a single Document.
    loader = TextLoader(str(settings.knowledge_file), encoding="utf-8")
    docs = loader.load()

    # 2. Split by markdown headings first. Clinic facts are naturally scoped by
    #    section/subsection: one doctor, one policy, one insurance list, etc.
    chunks = _split_clinic_markdown(docs[0].page_content)

    # 3. Replace the existing collection so changed chunking/settings do not
    #    leave stale vectors from a previous ingestion run.
    existing_store = Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=get_embeddings(),
    )
    try:
        existing_store.delete_collection()
    except ValueError:
        pass

    # 4. Embed + persist. We point Chroma at a persistent directory so the
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
