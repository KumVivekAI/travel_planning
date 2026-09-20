"""Build the FAISS vector store from knowledge base documents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.ingestion import build_vector_store


if __name__ == "__main__":
    print("Building vector store from knowledge base...")
    store = build_vector_store()
    print(f"Done. Indexed {store.index.ntotal} chunks.")
