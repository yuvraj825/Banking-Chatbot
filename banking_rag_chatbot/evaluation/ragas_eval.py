"""
evaluation/ragas_eval.py
Evaluates the RAG pipeline using RAGAS metrics:
  - answer_faithfulness   : Is the answer grounded in the retrieved context?
  - answer_relevancy      : Does the answer address the question?
  - context_recall        : Are the relevant chunks retrieved?

Test set: 20 banking questions with ground truth answers.
Outputs a score report to console and saves results to evaluation/results.json.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_FILE = Path(__file__).resolve().parent / "results.json"

# ── Test dataset ──────────────────────────────────────────────────────────────

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
        "ground_truth": "SBI offers home loans based on the applicant's income, credit score, and property value. Loan amounts can go up to several crores depending on eligibility.",
    },
    {
        "question": "What documents are required for an HDFC personal loan?",
        "ground_truth": "HDFC requires identity proof, address proof, income proof (salary slips or ITR), bank statements, and passport-size photographs.",
    },
    {
        "question": "What is the Statutory Liquidity Ratio (SLR)?",
        "ground_truth": "SLR is the minimum percentage of deposits that a commercial bank must maintain in the form of liquid assets like cash, gold, or government securities.",
    },
    {
        "question": "What is the MCLR and how does it affect my loan EMI?",
        "ground_truth": "MCLR (Marginal Cost of Funds-based Lending Rate) is the minimum interest rate below which banks cannot lend. Changes in MCLR affect floating-rate loan EMIs.",
    },
    {
        "question": "What is the maximum loan-to-value ratio for home loans in India?",
        "ground_truth": "As per RBI guidelines, the LTV ratio for home loans above Rs 75 lakh is capped at 75%, meaning you need to bring a minimum 25% down payment.",
    },
    {
        "question": "How does ICICI calculate interest on personal loans?",
        "ground_truth": "ICICI Bank charges interest on personal loans at a flat or reducing balance rate depending on the product. Interest is computed on the outstanding principal.",
    },
    {
        "question": "What is the SARFAESI Act and how does it protect banks?",
        "ground_truth": "The SARFAESI Act, 2002 allows banks to recover bad loans by seizing and selling collateral without court intervention for loans above Rs 1 lakh.",
    },
    {
        "question": "What are the RBI guidelines on KYC for bank accounts?",
        "ground_truth": "RBI mandates Know Your Customer (KYC) norms requiring identity proof, address proof, and photograph for opening bank accounts, with periodic KYC updates.",
    },
    {
        "question": "What is a Non-Performing Asset (NPA)?",
        "ground_truth": "An NPA is a loan or advance where the interest or principal payment has been overdue for more than 90 days.",
    },
    {
        "question": "How does the RBI regulate NBFC lending?",
        "ground_truth": "RBI regulates NBFCs through registration requirements, capital adequacy norms, exposure limits, and mandatory reporting under its NBFC Master Directions.",
    },
    {
        "question": "What is the minimum CIBIL score required for a home loan?",
        "ground_truth": "Most banks require a CIBIL score of 750 or above for home loan approval, though exact cutoffs vary by lender.",
    },
    {
        "question": "What is the prepayment penalty for home loans?",
        "ground_truth": "As per RBI guidelines, banks cannot charge prepayment penalties on floating-rate home loans. Fixed-rate loans may have a penalty of up to 2% of the outstanding amount.",
    },
    {
        "question": "What is the difference between a savings account and a current account?",
        "ground_truth": "Savings accounts are for individuals, earn interest, and have transaction limits. Current accounts are for businesses, do not earn interest, and have no transaction limits.",
    },
    {
        "question": "What is a Recurring Deposit (RD)?",
        "ground_truth": "An RD is a term deposit where the customer deposits a fixed amount monthly for a predetermined tenure and earns interest at a fixed rate.",
    },
    {
        "question": "What are the RBI guidelines on bank locker facilities?",
        "ground_truth": "RBI's revised locker guidelines (2023) mandate banks to have a standardised locker agreement, limit liability to 100 times the annual locker rent for bank negligence.",
    },
    {
        "question": "How does UPI work and who regulates it?",
        "ground_truth": "UPI (Unified Payments Interface) is a real-time payment system operated by NPCI and regulated by RBI, allowing instant interbank transfers via mobile.",
    },
    {
        "question": "What is the PMJDY scheme?",
        "ground_truth": "Pradhan Mantri Jan Dhan Yojana is a government financial inclusion scheme providing zero-balance bank accounts, RuPay debit cards, and overdraft facilities to unbanked citizens.",
    },
    {
        "question": "What are the tax benefits on home loan EMIs?",
        "ground_truth": "Under Section 80C, you can claim deduction up to Rs 1.5 lakh on home loan principal repayment. Under Section 24(b), up to Rs 2 lakh can be claimed on interest paid.",
    },
]


# ── Evaluation logic ──────────────────────────────────────────────────────────

def run_evaluation(verbose: bool = True) -> dict:
    """
    Run RAGAS evaluation on the test set.

    Returns:
        Dict of metric scores and per-question results.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset
    except ImportError:
        print("ERROR: Install ragas and datasets: pip install ragas datasets")
        return {}

    from pipeline.rag_chain import build_rag_chain, query_chain
    from retrieval.retriever import retrieve

    print("Loading RAG chain for evaluation…")
    chain = build_rag_chain()

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"Running {len(TEST_QUESTIONS)} test questions…\n")
    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["question"]
        gt = item["ground_truth"]
        if verbose:
            print(f"[{i:02d}/{len(TEST_QUESTIONS)}] {q[:60]}…")

        # Get answer
        result = query_chain(chain, q)
        answer = result["answer"]

        # Get retrieved contexts separately
        context_docs = retrieve(q, k=4)
        contexts = [doc.page_content for doc in context_docs]

        eval_data["question"].append(q)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(gt)

    dataset = Dataset.from_dict(eval_data)

    print("\nRunning RAGAS metrics…")
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
