"""
pipeline/updater.py
Orchestrates automatic knowledge base updates using APScheduler.
Runs the full ingestion pipeline: scrape → deduplicate → load → chunk → embed.

Schedule: Daily at 06:00 UTC (configurable).
Can also be triggered manually via CLI or the Streamlit sidebar.
"""

import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ingestion.scraper import run_all_scrapers
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_save, get_index_stats
from ingestion.deduplicator import is_new_document, register_document

# ── Logging setup ─────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "updater.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # ensure data/ exists before handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Run log (for Streamlit sidebar display) ───────────────────────────────────

RUN_LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "last_update.json"


def _save_run_log(new_docs: int, total_chunks: int) -> None:
    data = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "new_documents_ingested": new_docs,
        "new_chunks_added": total_chunks,
        "index_stats": get_index_stats(),
    }
    with open(RUN_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_run_log() -> dict:
    """Read the last update run log (for Streamlit sidebar)."""
    if RUN_LOG_FILE.exists():
        with open(RUN_LOG_FILE) as f:
            return json.load(f)
    return {"last_run": "Never", "new_documents_ingested": 0, "index_stats": {}}


# ── Core ingestion pipeline ───────────────────────────────────────────────────

def run_ingestion_pipeline(max_rbi_pdfs: int = 10) -> dict:
    """
    Full pipeline: scrape → deduplicate → load → chunk → embed.

    Returns:
        Summary dict with counts of new docs and chunks added.
    """
    logger.info("=" * 60)
    logger.info("Starting knowledge base update…")
    logger.info("=" * 60)

    # Step 1: Scrape
    scraper_entries = run_all_scrapers(max_rbi_pdfs=max_rbi_pdfs)

    # Step 2: Deduplicate
    new_entries = []
    skipped = 0
    for entry in scraper_entries:
        text = entry.get("text", "")
        source = entry.get("source_url", "unknown")

        # For PDF entries, text is filled by loader — skip dedup on empty text
        if not text and entry.get("local_path"):
            new_entries.append(entry)  # Always try to load local files
            continue

        is_new, doc_hash = is_new_document(text, source)
        if is_new:
            new_entries.append(entry)
            register_document(source, doc_hash)
        else:
            skipped += 1

    logger.info(f"Deduplication: {len(new_entries)} new | {skipped} already known.")

    if not new_entries:
        logger.info("No new documents found. Knowledge base is up to date.")
        _save_run_log(0, 0)
        return {"new_documents": 0, "new_chunks": 0, "skipped": skipped}

    # Step 3: Load
    docs = load_documents(new_entries)

    # Step 4: Chunk
    chunks = chunk_documents(docs)

    # Step 5: Embed + save
    if chunks:
        embed_and_save(chunks)

    stats = {"new_documents": len(new_entries), "new_chunks": len(chunks), "skipped": skipped}
    logger.info(f"Update complete: {stats}")
    _save_run_log(len(new_entries), len(chunks))
    return stats


# ── Scheduler ─────────────────────────────────────────────────────────────────

def start_scheduler(hour: int = 6, minute: int = 0, blocking: bool = True) -> None:
    """
    Start APScheduler to run the ingestion pipeline daily.

    Args:
        hour:     UTC hour to run (default: 6 = 06:00 UTC).
        minute:   Minute offset (default: 0).
        blocking: If True, block the process (use for standalone script).
                  If False, run in background thread (use for Streamlit).
    """
    SchedulerClass = BlockingScheduler if blocking else BackgroundScheduler
    scheduler = SchedulerClass(timezone="UTC")

    scheduler.add_job(
        func=run_ingestion_pipeline,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="daily_kb_update",
        name="Daily knowledge base refresh",
        replace_existing=True,
    )

    logger.info(f"Scheduler started — next run at {hour:02d}:{minute:02d} UTC daily.")

    if not blocking:
        scheduler.start()
        return scheduler

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banking RAG Knowledge Base Updater")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run the ingestion pipeline immediately instead of scheduling.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Start the daily scheduler (blocks the process).",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=10,
        help="Max RBI PDFs to scrape per run (default: 10).",
    )
    args = parser.parse_args()

    if args.run_now:
        result = run_ingestion_pipeline(max_rbi_pdfs=args.max_pdfs)
        print(f"\nDone: {result}")
    elif args.schedule:
        start_scheduler(hour=6, minute=0, blocking=True)
    else:
        parser.print_help()
