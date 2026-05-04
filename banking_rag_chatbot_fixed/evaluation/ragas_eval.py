"""
evaluation/ragas_eval.py
Evaluates the RAG pipeline using RAGAS metrics.
Compatible with ragas >= 0.1.x and langchain >= 1.0.

Metrics:
  - faithfulness    : Is the answer grounded in retrieved context?
  - answer_relevancy: Does the answer address the question?
  - context_recall  : Are the relevant chunks retrieved?

Run after building the FAISS index:
  python evaluation/ragas_eval.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_FILE = Path(__file__).resolve().parent / "results.json"

TEST_QUESTIONS = [
    {
        "question": "What is the current repo rate set by RBI?",
        "ground_truth": "The RBI has set the repo rate at 6.50% as per the latest Monetary Policy Committee decision.",
    },
    {
        "question": "What is the Cash Reserve Ratio (CRR)?",
        "ground_truth": "CRR is the percentage of a bank's total deposits that it must keep in reserve with the RBI. It currently stands at 4%.",
    },
    {
        "question": "How much home loan can I get from SBI?",
        "ground_truth": "SBI offers home loans based on the applicant's income, credit score, and property value.",
    },
    {
        "question": "What documents are required for an HDFC personal loan?",
        "ground_truth": "HDFC requires identity proof, address proof, income proof, bank statements, and passport-size photographs.",
    },
    {
        "question": "What is the Statutory Liquidity Ratio (SLR)?",
        "ground_truth": "SLR is the minimum percentage of deposits that a bank must maintain in liquid assets like cash, gold, or government securities.",
    },
    {
        "question": "What is the MCLR and how does it affect my loan EMI?",
        "ground_truth": "MCLR is the minimum interest rate below which banks cannot lend. Changes in MCLR affect floating-rate loan EMIs.",
    },
    {
        "question": "What is the maximum loan-to-value ratio for home loans in India?",
        "ground_truth": "As per RBI guidelines, the LTV ratio for home loans above Rs 75 lakh is capped at 75%.",
    },
    {
        "question": "What are the RBI guidelines on KYC for bank accounts?",
        "ground_truth": "RBI mandates KYC norms requiring identity proof, address proof, and photograph for opening bank accounts.",
    },
    {
        "question": "What is a Non-Performing Asset (NPA)?",
        "ground_truth": "An NPA is a loan or advance where the interest or principal payment has been overdue for more than 90 days.",
    },
    {
        "question": "What is the minimum CIBIL score required for a home loan?",
        "ground_truth": "Most banks require a CIBIL score of 750 or above for home loan approval.",
    },
    {
        "question": "What is the prepayment penalty for home loans?",
        "ground_truth": "RBI guidelines prohibit prepayment penalties on floating-rate home loans.",
    },
    {
        "question": "What is a Recurring Deposit (RD)?",
        "ground_truth": "An RD is a term deposit where the customer deposits a fixed amount monthly for a predetermined tenure.",
    },
    {
        "question": "How does UPI work and who regulates it?",
        "ground_truth": "UPI is a real-time payment system operated by NPCI and regulated by RBI.",
    },
    {
        "question": "What is the PMJDY scheme?",
        "ground_truth": "Pradhan Mantri Jan Dhan Yojana provides zero-balance bank accounts and RuPay debit cards to unbanked citizens.",
    },
    {
        "question": "What are the tax benefits on home loan EMIs?",
        "ground_truth": "Under Section 80C, deduction up to Rs 1.5 lakh on principal; under Section 24(b), up to Rs 2 lakh on interest.",
    },
]


def run_evaluation(verbose: bool = True) -> dict:
    """
    Run RAGAS evaluation on the test set.
    Returns dict of metric scores and per-question results.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset
    except ImportError:
        print("ERROR: Install evaluation deps: pip install ragas datasets")
        return {}

    from pipeline.rag_chain import build_rag_chain, query_chain
    from retrieval.retriever import retrieve

    print("Loading RAG chain for evaluation...")
    chain_bundle = build_rag_chain()

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"Running {len(TEST_QUESTIONS)} test questions...\n")
    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["question"]
        gt = item["ground_truth"]
        if verbose:
            print(f"[{i:02d}/{len(TEST_QUESTIONS)}] {q[:60]}...")

        result = query_chain(chain_bundle, q)
        answer = result["answer"]

        context_docs = retrieve(q, k=4)
        contexts = [doc.page_content for doc in context_docs]

        eval_data["question"].append(q)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(gt)

    dataset = Dataset.from_dict(eval_data)

    print("\nRunning RAGAS metrics...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    score_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in score_dict.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<30} {score:.4f}  {bar}")
    print("=" * 50)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(TEST_QUESTIONS),
        "scores": score_dict,
        "per_question": [
            {
                "question": eval_data["question"][i],
                "answer": eval_data["answer"][i],
                "ground_truth": eval_data["ground_truth"][i],
            }
            for i in range(len(TEST_QUESTIONS))
        ],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {RESULTS_FILE}")
    return output


if __name__ == "__main__":
    run_evaluation(verbose=True)
