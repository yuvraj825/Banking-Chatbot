"""
chunker.py
Splits LangChain Documents into smaller overlapping chunks using
RecursiveCharacterTextSplitter. Preserves all metadata from source documents.

Chunk size : 500 tokens  (~2 000 characters at ~4 chars/token)
Overlap    : 50 tokens   (~200 characters)
"""

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Splitter configuration ────────────────────────────────────────────────────
# We work in approximate token counts.
# RecursiveCharacterTextSplitter uses character count, so multiply by ~4.

CHUNK_SIZE = 500 * 4      # ≈ 500 tokens
CHUNK_OVERLAP = 50 * 4    # ≈ 50 tokens

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of LangChain Documents into chunks.

    Each chunk inherits the full metadata of its parent document and gets
    an additional 'chunk_index' field for traceability.

    Args:
        docs: List of LangChain Document objects (output of loader.py).

    Returns:
        List of chunked Document objects.
    """
    if not docs:
        print("[Chunker] No documents to chunk.")
        return []

    all_chunks: list[Document] = []

    for doc in docs:
        if not doc.page_content.strip():
            continue

        splits = _SPLITTER.split_text(doc.page_content)

        for i, chunk_text in enumerate(splits):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,          # Inherit source, doc_type, date_scraped
                    "chunk_index": i,
                    "total_chunks": len(splits),
                }
            )
            all_chunks.append(chunk_doc)

    print(
        f"[Chunker] {len(docs)} document(s) → {len(all_chunks)} chunk(s) "
        f"(avg {len(all_chunks) // max(len(docs), 1)} per doc)."
    )
    return all_chunks


def chunk_text(text: str, metadata: dict | None = None) -> list[Document]:
    """
    Convenience wrapper: chunk a raw string directly.

    Args:
        text:     Raw text to split.
        metadata: Optional dict of metadata to attach to each chunk.

    Returns:
        List of Document chunks.
    """
    doc = Document(page_content=text, metadata=metadata or {})
    return chunk_documents([doc])


if __name__ == "__main__":
    sample = """
    The Reserve Bank of India (RBI) is India's central bank and regulatory body
    responsible for the regulation of the Indian banking system. It is under the
    ownership of the Ministry of Finance, Government of India.

    The RBI controls monetary policy in India and issues the Indian rupee. It also
    manages the country's foreign exchange under the Foreign Exchange Management Act, 1999.

    The RBI has set the repo rate at 6.50% in its latest Monetary Policy Committee meeting.
    This decision was taken to control inflation while supporting growth.
    """ * 5  # Repeat to force chunking

    chunks = chunk_text(sample, metadata={"source": "test", "doc_type": "rbi_circular"})
    print(f"Produced {len(chunks)} chunks. First chunk preview:")
    print(chunks[0].page_content[:200])
    print("Metadata:", chunks[0].metadata)
