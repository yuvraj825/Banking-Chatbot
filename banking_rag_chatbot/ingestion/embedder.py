"""
embedder.py
Embeds document chunks using HuggingFace's all-MiniLM-L6-v2 (free, no API key).
Upserts new vectors into a FAISS index stored on disk.

Why all-MiniLM-L6-v2?
  – 384-dim embeddings, very fast on CPU
  – Excellent semantic similarity for English banking text
  – Completely free to use locally
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone

from langchain.schema import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore" / "faiss_index"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_META_FILE = FAISS_INDEX_DIR / "index_meta.json"


# ── Embedding model (singleton) ───────────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("[Embedder] Loading embedding model (first run may download ~90MB)…")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[Embedder] Embedding model ready.")
    return _embeddings


# ── Index meta helpers ────────────────────────────────────────────────────────

def _load_meta() -> dict:
    if INDEX_META_FILE.exists():
        with open(INDEX_META_FILE) as f:
            return json.load(f)
    return {"doc_count": 0, "last_updated": None}


def _save_meta(meta: dict) -> None:
    with open(INDEX_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def index_exists() -> bool:
    """Return True if a saved FAISS index is present on disk."""
    return (FAISS_INDEX_DIR / "index.faiss").exists()


def load_vectorstore() -> FAISS:
    """Load the existing FAISS index from disk."""
    if not index_exists():
        raise FileNotFoundError(
            f"No FAISS index found at {FAISS_INDEX_DIR}. "
            "Run embed_and_save() first."
        )
    embeddings = _get_embeddings()
    store = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[Embedder] FAISS index loaded from {FAISS_INDEX_DIR}.")
    return store


def embed_and_save(chunks: list[Document]) -> FAISS:
    """
    Embed a list of chunks and upsert them into the FAISS index.

    – If an index already exists on disk, it merges new vectors in.
    – If no index exists, a new one is created from scratch.

    Args:
        chunks: List of LangChain Document chunks (output of chunker.py).

    Returns:
        The updated FAISS vectorstore object.
    """
    if not chunks:
        print("[Embedder] No chunks to embed — skipping.")
        return load_vectorstore() if index_exists() else None

    embeddings = _get_embeddings()
    print(f"[Embedder] Embedding {len(chunks)} chunk(s)…")

    if index_exists():
        # Load and merge
        store = load_vectorstore()
        store.add_documents(chunks)
        print(f"[Embedder] Upserted {len(chunks)} chunk(s) into existing index.")
    else:
        # Create fresh index
        store = FAISS.from_documents(chunks, embeddings)
        print(f"[Embedder] Created new FAISS index with {len(chunks)} chunk(s).")

    store.save_local(str(FAISS_INDEX_DIR))

    # Update meta
    meta = _load_meta()
    meta["doc_count"] = meta.get("doc_count", 0) + len(chunks)
    meta["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_meta(meta)

    print(f"[Embedder] Index saved → {FAISS_INDEX_DIR}")
    return store


def get_index_stats() -> dict:
    """Return metadata about the current FAISS index."""
    meta = _load_meta()
    meta["index_exists"] = index_exists()
    return meta


if __name__ == "__main__":
    # Quick test with dummy chunks
    from langchain.schema import Document
    dummy_chunks = [
        Document(page_content="The repo rate is 6.5%.", metadata={"source": "test", "doc_type": "rbi_circular"}),
        Document(page_content="Home loan EMI calculator for SBI.", metadata={"source": "sbi", "doc_type": "bank_faq"}),
    ]
    store = embed_and_save(dummy_chunks)
    results = store.similarity_search("What is the repo rate?", k=1)
    print("Top result:", results[0].page_content)
    print("Stats:", get_index_stats())
