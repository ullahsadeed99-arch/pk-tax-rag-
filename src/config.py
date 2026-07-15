"""Central configuration for the Pakistan Tax Law RAG system.

All tunables live here so behavior can be changed via environment
variables without touching code. See .env.example for the full list.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
REPORTS_DIR = ROOT_DIR / "reports"
QA_DIR = ROOT_DIR / "qa"


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- Embeddings ---
    # Default is a free, offline, multilingual model (handles English + Urdu +
    # mixed queries) so the MVP runs with zero API keys. Swap to OpenAI for
    # higher-precision retrieval once budget allows (see reports/cost_estimation.md).
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")  # "local" | "openai"
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    # --- LLM (generation) ---
    # Provider auto-selected at runtime if "auto": prefers Anthropic, falls back
    # to OpenAI, based on whichever API key is present.
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")  # "auto" | "anthropic" | "openai"
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    # --- Chunking ---
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "850"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "175"))

    # --- Retrieval ---
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "8"))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "8"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "6"))
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.28"))

    # --- Chroma ---
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "pk_tax_law")

    # --- Conversation ---
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))

    # --- Feature flags ---
    enable_hybrid_retrieval: bool = _bool("ENABLE_HYBRID_RETRIEVAL", True)

    api_keys: dict = field(default_factory=lambda: {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    })


settings = Settings()
