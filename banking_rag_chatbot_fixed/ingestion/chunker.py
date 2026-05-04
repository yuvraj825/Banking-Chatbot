"""
ingestion/chunker.py
Splits LangChain Documents into smaller overlapping chunks.
Compatible with langchain >= 1.0.

Chunk size : ~500 tokens (~2000 chars at 4 chars/token)
Overlap    : ~50 tokens  (~200 chars)
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500 * 4      # ~500 tokens in characters
CHUNK_OVERLAP = 50 * 4    # ~50 tokens in characters

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_documents(docs: list) -> list:
    """
    Split a list of LangChain Documents into overlapping chunks.
    Each chunk inherits parent metadata plus chunk_index and total_chunks.

    Args:
        docs: List of LangChain Document objects (output of loader.py).

    Returns:
        List of chunked Document objects.
    """
    if not docs:
        print("[Chunker] No documents to chunk.")
        return []

    all_chunks = []

    for doc in docs:
        if not doc.page_content.strip():
            continue

        splits = _SPLITTER.split_text(doc.page_content)

        for i, chunk_text in enumerate(splits):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(splits),
                }
            )
            all_chunks.append(chunk_doc)

    print(
        f"[Chunker] {len(docs)} doc(s) -> {len(all_chunks)} chunk(s) "
        f"(avg {len(all_chunks) // max(len(docs), 1)} per doc)."
    )
    return all_chunks


def chunk_text(text: str, metadata: dict = None) -> list:
    """
    Convenience wrapper: chunk a raw string directly.

    Args:
        text:     Raw text to split.
        metadata: Optional metadata dict for each chunk.

    Returns:
        List of Document chunks.
    """
    doc = Document(page_content=text, metadata=metadata or {})
    return chunk_documents([doc])


if __name__ == "__main__":
    sample = """
    The Reserve Bank of India (RBI) is India's central bank and regulatory body
    responsible for the regulation of the Indian banking system.

    The RBI controls monetary policy in India and issues the Indian rupee.

    The RBI has set the repo rate at 6.50% in its latest Monetary Policy Committee meeting.
    """ * 5

    chunks = chunk_text(sample, metadata={"source": "test", "doc_type": "rbi_circular"})
    print(f"Produced {len(chunks)} chunks. First chunk preview:")
    print(chunks[0].page_content[:200])
    print("Metadata:", chunks[0].metadata)
