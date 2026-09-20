"""Application configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

DESTINATION = "Singapore"
DESTINATION_LATITUDE = 1.3521
DESTINATION_LONGITUDE = 103.8198
DESTINATION_TIMEZONE = "Asia/Singapore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM: OpenAI (set OPENAI_API_KEY in .env). Embeddings run locally via HuggingFace.
DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
