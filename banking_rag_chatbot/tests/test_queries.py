"""
tests/test_queries.py
Sample banking queries for manual smoke-testing the RAG pipeline.
Run this after setting up the FAISS index to verify the chain works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_QUERIES = [
    # RBI Policy
    "What is the current repo rate?",
    "What is the Cash Reserve Ratio set by RBI?",
    "Explain the Statutory Liquidity Ratio.",
    "What are RBI's KYC norms for opening a bank account?",

    # Home Loans
    "What are the eligibility criteria for an SBI home loan?",
    "How much home loan can I get with a salary of 60,000 per month?",
    "What is the maximum LTV ratio for home loans above 75 lakhs?",
    "Can I prepay my home loan without penalty?",

    # Personal Loans
    "What is the interest rate on HDFC personal loans?",
    "What documents do I need for an ICICI personal loan?",

    # General Banking
    "What is the difference between NEFT and RTGS?",
    "How does UPI work?",
    "What is a Non-Performing Asset?",
    "What are the tax benefits on home loan interest?",
    "What is PMJDY?",
]


def run_smoke_test():
    """Run all sample queries and print answers."""
    from pipeline.rag_chain import get_chain, query_chain
    from ingestion.embedder import index_exists

    if not index_exists():
        print("ERROR: No FAISS index found. Run the ingestion pipeline first.")
        print("  python pipeline/updater.py --run-now")
        return

    print("Loading chain…")
    chain = get_chain()
    print(f"Running {len(SAMPLE_QUERIES)} test queries...\n")

    results = []
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        print(f"[{i:02d}] Q: {q}")
        try:
            result = query_chain(chain, q)
            answer = result["answer"][:300]
            sources = [s["source"] for s in result["sources"]]
            print(f"     A: {answer}…")
            print(f"     Sources: {sources}\n")
            results.append({"question": q, "answer": answer, "sources": sources})
        except Exception as e:
            print(f"     ERROR: {e}\n")
            results.append({"question": q, "error": str(e)})

    passed = sum(1 for r in results if "error" not in r)
    print(f"\nSmoke test complete: {passed}/{len(SAMPLE_QUERIES)} queries answered.")
    return results


if __name__ == "__main__":
    run_smoke_test()
