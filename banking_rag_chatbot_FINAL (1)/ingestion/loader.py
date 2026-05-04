"""
ingestion/loader.py
Converts raw scraper output into LangChain Document objects.
Handles PDF, HTML, CSV, and plain-text sources.
Compatible with langchain >= 1.0 / langchain-community >= 0.4.
"""

from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_pdf(local_path: str, source_url: str) -> list:
    """Extract text from a local PDF using PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(local_path)
        pages = loader.load()
        for doc in pages:
            doc.metadata.update({
                "source": source_url,
                "doc_type": "rbi_circular",
                "date_scraped": _today_str(),
            })
        return pages
    except Exception as e:
        print(f"[Loader] PDF load failed for {local_path}: {e}")
        return []


def _load_html_text(text: str, source_url: str, doc_type: str) -> list:
    """Wrap pre-scraped HTML text into a LangChain Document."""
    if not text.strip():
        return []
    return [Document(
        page_content=text,
        metadata={
            "source": source_url,
            "doc_type": doc_type,
            "date_scraped": _today_str(),
        }
    )]


def _load_html_file(local_path: str, source_url: str) -> list:
    """Load an HTML file from disk."""
    try:
        from langchain_community.document_loaders import BSHTMLLoader
        loader = BSHTMLLoader(local_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": source_url,
                "doc_type": "html",
                "date_scraped": _today_str(),
            })
        return docs
    except Exception as e:
        print(f"[Loader] HTML file load failed for {local_path}: {e}")
        return []


def _load_csv(local_path: str, source_url: str) -> list:
    """Load a CSV file row-by-row as Documents."""
    try:
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file_path=local_path, metadata_columns=[])
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": source_url,
                "doc_type": "csv",
                "date_scraped": _today_str(),
            })
        return docs
    except Exception as e:
        print(f"[Loader] CSV load failed for {local_path}: {e}")
        return []


def _load_txt(local_path: str, source_url: str) -> list:
    """Load a plain text file."""
    try:
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [Document(
            page_content=content,
            metadata={
                "source": source_url,
                "doc_type": "text",
                "date_scraped": _today_str(),
            }
        )]
    except Exception as e:
        print(f"[Loader] TXT load failed for {local_path}: {e}")
        return []


def load_documents(scraper_entries: list) -> list:
    """
    Convert scraper metadata dicts into LangChain Documents.

    Args:
        scraper_entries: Output of run_all_scrapers().

    Returns:
        Flat list of LangChain Document objects with metadata.
    """
    all_docs = []

    for entry in scraper_entries:
        source_url = entry.get("source_url", "unknown")
        doc_type = entry.get("doc_type", "unknown")
        text = entry.get("text", "")
        local_path = entry.get("local_path")

        if local_path:
            path = Path(local_path)
            ext = path.suffix.lower()
            if ext == ".pdf":
                docs = _load_pdf(local_path, source_url)
            elif ext in (".html", ".htm"):
                docs = _load_html_file(local_path, source_url)
            elif ext == ".csv":
                docs = _load_csv(local_path, source_url)
            elif ext == ".txt":
                docs = _load_txt(local_path, source_url)
            else:
                print(f"[Loader] Unsupported file type: {ext} — skipping.")
                docs = []
        elif text:
            docs = _load_html_text(text, source_url, doc_type)
        else:
            print(f"[Loader] No content for {source_url} — skipping.")
            docs = []

        all_docs.extend(docs)

    print(f"[Loader] Loaded {len(all_docs)} document(s) total.")
    return all_docs


if __name__ == "__main__":
    dummy = [{
        "text": "The RBI has set the repo rate at 6.5% for Q1 2024.",
        "source_url": "https://rbi.org.in/test",
        "doc_type": "rbi_circular",
        "local_path": None,
    }]
    docs = load_documents(dummy)
    for d in docs:
        print(d.page_content[:80], "|", d.metadata)
