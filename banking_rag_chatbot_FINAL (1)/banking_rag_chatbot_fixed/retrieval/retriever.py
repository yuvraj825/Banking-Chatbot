"""
retrieval/retriever.py
Loads the FAISS index from disk and returns a LangChain retriever.
Compatible with langchain >= 1.0 / langchain-community >= 0.4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from ingestion.embedder import load_vectorstore, get_index_stats

TOP_K = 4


def get_retriever(search_type: str = "similarity", k: int = TOP_K) -> VectorStoreRetriever:
    """
    Load the FAISS vectorstore and return a configured retriever.

    Args:
        search_type: "similarity" or "mmr" (maximum marginal relevance).
        k:           Number of chunks to retrieve per query.

    Returns:
        A LangChain VectorStoreRetriever.
    """
    store = load_vectorstore()

    if search_type == "mmr":
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )
    else:
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    print(f"[Retriever] Ready — type={search_type}, k={k}.")
    return retriever


def retrieve(query: str, k: int = TOP_K) -> list:
    """
    Convenience function: run a one-off retrieval query.

    Args:
        query: Natural language question.
        k:     Number of chunks to return.

    Returns:
        List of LangChain Document objects ranked by similarity.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    print(f"[Retriever] '{query[:60]}' -> {len(docs)} chunk(s).")
    return docs


def get_vectorstore() -> FAISS:
    """Return the raw FAISS vectorstore."""
    return load_vectorstore()


if __name__ == "__main__":
    stats = get_index_stats()
    print("Index stats:", stats)
    if stats["index_exists"]:
        results = retrieve("What is the current repo rate in India?")
        for i, doc in enumerate(results):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:200])
            print("Source:", doc.metadata.get("source"))
    else:
        print("No FAISS index found. Run the ingestion pipeline first.")
