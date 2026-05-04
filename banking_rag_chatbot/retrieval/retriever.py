"""
retrieval/retriever.py
Loads the FAISS index from disk and returns a LangChain retriever
configured to fetch the top-4 most semantically similar chunks per query.
"""

from langchain_community.vectorstores import FAISS
from langchain.schema import BaseRetriever

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.embedder import load_vectorstore, get_index_stats

# Number of chunks to retrieve per query
TOP_K = 4


def get_retriever(search_type: str = "similarity", k: int = TOP_K) -> BaseRetriever:
    """
    Load the FAISS vectorstore and return a configured retriever.

    Args:
        search_type: "similarity" (default) or "mmr" (maximum marginal relevance
                     for more diverse results).
        k:           Number of chunks to retrieve per query.

    Returns:
        A LangChain VectorStoreRetriever ready for use in a RAG chain.
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
    Convenience function: load retriever and run a one-off query.

    Args:
        query: The user's natural language question.
        k:     Number of chunks to return.

    Returns:
        List of LangChain Document objects ranked by similarity.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    print(f"[Retriever] Query: '{query[:60]}' → {len(docs)} chunk(s) returned.")
    return docs


def get_vectorstore() -> FAISS:
    """Return the raw FAISS vectorstore (for direct similarity search)."""
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
