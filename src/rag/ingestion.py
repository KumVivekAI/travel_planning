"""Knowledge base ingestion: load, chunk, embed, and store documents."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
)


def _extract_metadata(file_path: Path) -> dict:
    """Parse source title and URL from markdown header lines."""
    text = file_path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    url_match = re.search(r"\*\*URL:\*\*\s+(.+)$", text, re.MULTILINE)

    return {
        "source": file_path.name,
        "source_title": title_match.group(1).strip() if title_match else file_path.stem,
        "source_url": url_match.group(1).strip() if url_match else "",
    }


def load_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> list:
    """Load markdown travel guides from the knowledge base directory."""
    loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    for doc in documents:
        file_path = Path(doc.metadata.get("source", ""))
        if file_path.exists():
            doc.metadata.update(_extract_metadata(file_path))

    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.split_documents(documents)


def build_vector_store(
    kb_dir: Path = KNOWLEDGE_BASE_DIR,
    store_dir: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """Build and persist a FAISS vector store from knowledge base documents."""
    documents = load_documents(kb_dir)
    if not documents:
        raise FileNotFoundError(f"No documents found in {kb_dir}")

    chunks = chunk_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_dir))
    return vector_store


def load_vector_store(store_dir: Path = VECTOR_STORE_DIR) -> FAISS:
    """Load a persisted FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
