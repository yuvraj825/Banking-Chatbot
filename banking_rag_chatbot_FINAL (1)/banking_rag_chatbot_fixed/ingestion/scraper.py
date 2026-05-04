"""
scraper.py
Scrapes RBI circular PDFs and bank FAQ pages, plus watches the local data/raw/ folder
for manually dropped files.

Returns a list of dicts:
  [{"text": str, "source_url": str, "doc_type": str}, ...]
"""

import os
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import urllib.parse

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Source definitions ──────────────────────────────────────────────────────

SOURCES = {
    "rbi_circulars": {
        "url": "https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx",
        "doc_type": "rbi_circular",
    },
    "sbi_home_loan_faq": {
        "url": "https://sbi.co.in/web/home-loans/faq",
        "doc_type": "bank_faq",
    },
    "hdfc_loan_faq": {
        "url": "https://www.hdfcbank.com/personal/borrow/popular-loans",
        "doc_type": "bank_faq",
    },
    "icici_loan_faq": {
        "url": "https://www.icicibank.com/personal-banking/loans",
        "doc_type": "bank_faq",
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_get(url: str, timeout: int = 15):
    """GET with retry logic and timeout."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"[Scraper] Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def _download_pdf(pdf_url: str):
    """Download a PDF to data/raw/ and return its local path."""
    filename = urllib.parse.quote(pdf_url.split("/")[-1], safe="")[:100] + ".pdf"
    local_path = RAW_DATA_DIR / filename
    if local_path.exists():
        return local_path  # Already downloaded
    resp = _safe_get(pdf_url)
    if resp and resp.headers.get("Content-Type", "").startswith("application/pdf"):
        with open(local_path, "wb") as f:
            f.write(resp.content)
        print(f"[Scraper] Downloaded PDF: {filename}")
        return local_path
    return None


# ── Scrapers ─────────────────────────────────────────────────────────────────

def scrape_rbi_circulars(max_pdfs: int = 10) -> list[dict]:
    """
    Scrape the RBI circular index page, collect PDF links, download them,
    and return metadata entries (actual text extraction happens in loader.py).
    """
    results = []
    resp = _safe_get(SOURCES["rbi_circulars"]["url"])
    if not resp:
        print("[Scraper] Could not reach RBI circular index.")
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            full_url = urllib.parse.urljoin("https://www.rbi.org.in", href)
            pdf_links.append(full_url)
        if len(pdf_links) >= max_pdfs:
            break

    for pdf_url in pdf_links:
        local_path = _download_pdf(pdf_url)
        if local_path:
            results.append({
                "text": "",               # Filled by loader.py
                "source_url": pdf_url,
                "doc_type": "rbi_circular",
                "local_path": str(local_path),
            })

    print(f"[Scraper] RBI: found {len(results)} PDF(s).")
    return results


def scrape_bank_faq(name: str) -> list[dict]:
    """Generic FAQ scraper — extracts visible paragraph text from a bank FAQ page."""
    cfg = SOURCES.get(name)
    if not cfg:
        return []

    resp = _safe_get(cfg["url"])
    if not resp:
        print(f"[Scraper] Could not reach {name}.")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav/footer noise
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    headings = [h.get_text(separator=" ", strip=True) for h in soup.find_all(["h2", "h3", "h4"])]

    combined_text = "\n".join(filter(None, headings + paragraphs))

    if not combined_text.strip():
        print(f"[Scraper] {name}: no text extracted.")
        return []

    print(f"[Scraper] {name}: extracted {len(combined_text)} chars.")
    return [{
        "text": combined_text,
        "source_url": cfg["url"],
        "doc_type": cfg["doc_type"],
        "local_path": None,
    }]


def scan_local_raw_folder() -> list[dict]:
    """
    Detect any manually dropped files in data/raw/ (PDFs, CSVs, HTMLs).
    Returns metadata entries for loader.py to process.
    """
    supported = {".pdf", ".csv", ".html", ".htm", ".txt"}
    results = []
    for file_path in RAW_DATA_DIR.iterdir():
        if file_path.suffix.lower() in supported:
            doc_type = "pdf" if file_path.suffix.lower() == ".pdf" else \
                       "csv" if file_path.suffix.lower() == ".csv" else "html"
            results.append({
                "text": "",
                "source_url": file_path.name,
                "doc_type": doc_type,
                "local_path": str(file_path),
            })
    print(f"[Scraper] Local raw folder: {len(results)} file(s) found.")
    return results


def run_all_scrapers(max_rbi_pdfs: int = 10) -> list[dict]:
    """Run all scrapers and return a combined list of document metadata."""
    docs = []
    docs += scrape_rbi_circulars(max_pdfs=max_rbi_pdfs)
    docs += scrape_bank_faq("sbi_home_loan_faq")
    docs += scrape_bank_faq("hdfc_loan_faq")
    docs += scrape_bank_faq("icici_loan_faq")
    docs += scan_local_raw_folder()
    print(f"[Scraper] Total documents collected: {len(docs)}")
    return docs


if __name__ == "__main__":
    all_docs = run_all_scrapers(max_rbi_pdfs=3)
    for d in all_docs:
        print(f"  [{d['doc_type']}] {d['source_url']} | path: {d.get('local_path')}")
