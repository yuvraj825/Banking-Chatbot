"""
deduplicator.py
SHA-256 based deduplication to avoid re-ingesting documents already in the FAISS index.
Hashes are stored in data/hashes.json.
"""

import hashlib
import json
import os
from pathlib import Path

HASHES_FILE = Path(__file__).resolve().parent.parent / "data" / "hashes.json"


def _load_hashes() -> dict:
    """Load the existing hash store from disk."""
    if not HASHES_FILE.exists():
        return {}
    with open(HASHES_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_hashes(hashes: dict) -> None:
    """Persist the hash store to disk."""
    HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def compute_hash(text: str) -> str:
    """Return SHA-256 hex digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_new_document(text: str, source: str) -> tuple[bool, str]:
    """
    Check whether this document is new (not yet ingested).

    Args:
        text:   Full text content of the document.
        source: A unique identifier for the source (URL or file path).

    Returns:
        (is_new: bool, hash: str)
        is_new = True  → document has never been seen before
        is_new = False → document was already ingested (same hash)
    """
    hashes = _load_hashes()
    doc_hash = compute_hash(text)

    # Check both: same source updated, or identical content from different source
    existing_hashes = set(hashes.values())
    if doc_hash in existing_hashes:
        return False, doc_hash

    return True, doc_hash


def register_document(source: str, doc_hash: str) -> None:
    """
    Mark a document as ingested by storing its hash.

    Args:
        source:   Unique source identifier (URL or file path).
        doc_hash: SHA-256 hash of the document text.
    """
    hashes = _load_hashes()
    hashes[source] = doc_hash
    _save_hashes(hashes)


def get_all_hashes() -> dict:
    """Return the full hash store {source: hash}."""
    return _load_hashes()


def clear_hashes() -> None:
    """Reset the hash store (use with caution — forces full re-ingestion)."""
    _save_hashes({})
    print("[Deduplicator] Hash store cleared.")


if __name__ == "__main__":
    # Quick smoke test
    sample_text = "This is a sample RBI circular about repo rate."
    is_new, h = is_new_document(sample_text, "https://rbi.org.in/test")
    print(f"Is new: {is_new}, Hash: {h[:12]}...")
    if is_new:
        register_document("https://rbi.org.in/test", h)
    is_new2, h2 = is_new_document(sample_text, "https://rbi.org.in/test")
    print(f"After registration — Is new: {is_new2}")
