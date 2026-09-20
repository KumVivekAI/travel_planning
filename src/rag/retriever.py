"""RAG retrieval utilities."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.config import TOP_K_RETRIEVAL, VECTOR_STORE_DIR
from src.rag.ingestion import build_vector_store, load_vector_store


def ensure_vector_store(store_dir: Path = VECTOR_STORE_DIR):
    """Load vector store or build it if missing."""
    if not store_dir.exists() or not any(store_dir.iterdir()):
        return build_vector_store(store_dir=store_dir)
    return load_vector_store(store_dir)


def retrieve_context(query: str, top_k: int = TOP_K_RETRIEVAL) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and format them for the LLM."""
    vector_store = ensure_vector_store()
    documents: list[Document] = vector_store.similarity_search(query, k=top_k)

    if not documents:
        return "", []

    context_parts = []
    sources = []

    for index, doc in enumerate(documents, start=1):
        title = doc.metadata.get("source_title", doc.metadata.get("source", "Unknown"))
        url = doc.metadata.get("source_url", "")
        context_parts.append(f"[KB-{index}] {title}\n{doc.page_content}")
        sources.append({"title": title, "url": url, "excerpt": doc.page_content[:200]})

    return "\n\n---\n\n".join(context_parts), sources
