import json
import os
from pathlib import Path
from src.rag.answer import answer_question
ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "monitoring" / "reports"
REPORT_PATH = REPORT_DIR / "rag_eval_report.json"
TEST_CASES = [
    {
        "id": 1,
        "name": "General Credit Query",
        "query": "Incorrect information on my credit report from Equifax",
        "filters": None,
        "expect_results": True,
        "expect_sufficiency": ["HIGH", "MEDIUM", "LOW"],
    },
    {
        "id": 2,
        "name": "Product Filter (Credit Card)",
        "query": "Late fees and billing dispute",
        "filters": {"product": "credit card"},
        "expect_results": True,
        "check_filter": lambda item: "credit" in item["product"].lower() or "card" in item["product"].lower(),
    },
    {
        "id": 3,
        "name": "Company Filter (Equifax)",
        "query": "Credit reporting errors and dispute",
        "filters": {"company": "Equifax"},
        "expect_results": True,
        "check_filter": lambda item: "equifax" in item["company"].lower(),
    },
    {
        "id": 4,
        "name": "Date Start Filter",
        "query": "Dispute about loan payments",
        "filters": {"date_start": "2023-01-01"},
        "expect_results": True,
        "check_filter": lambda item: item["date"] >= "2023-01-01",
    },
    {
        "id": 5,
        "name": "Date End Filter",
        "query": "Credit score went down due to inquiry",
        "filters": {"date_end": "2024-12-31"},
        "expect_results": True,
        "check_filter": lambda item: item["date"] <= "2024-12-31",
    },
    {
        "id": 6,
        "name": "Issue Filter (Information)",
        "query": "Account information dispute",
        "filters": {"issue": "information"},
        "expect_results": True,
        "check_filter": lambda item: "information" in item["issue"].lower() or "info" in item["issue"].lower(),
    },
    {
        "id": 7,
        "name": "Adversarial Out of Scope â€” Capital of France",
        "query": "What is the capital of France?",
        "filters": None,
        "expect_results": False,
        "expect_sufficiency": ["NONE"],
    },
    {
        "id": 8,
        "name": "Adversarial Out of Scope â€” Baking recipes",
        "query": "Can you give me a recipe for chocolate cake?",
        "filters": None,
        "expect_results": False,
        "expect_sufficiency": ["NONE"],
    },
    {
        "id": 9,
        "name": "Adversarial Out of Scope â€” Ancient History",
        "query": "Tell me about the history of the Roman Empire.",
        "filters": None,
        "expect_results": False,
        "expect_sufficiency": ["NONE"],
    },
    {
        "id": 10,
        "name": "Combined Product and Company Filter",
        "query": "Problems with credit reporting",
        "filters": {"product": "Credit reporting", "company": "Equifax"},
        "expect_results": True,
        "check_filter": lambda item: "credit" in item["product"].lower() and "equifax" in item["company"].lower(),
    },
]
def run_eval() -> dict:
    print(f"\n{'='*55}")
    print("  Running RAG Evaluation â€” 10 Q&A Test Cases")
    print(f"{'='*55}")
    from src.rag.retrieve import load_resources
    try:
        index, metadata, model = load_resources()
    except Exception as e:
        print(f"Error loading FAISS resources: {e}")
        print("Please build index first using: python src/rag/build_index.py")
        return {"error": str(e)}
    results = []
    passed_count = 0
    for tc in TEST_CASES:
        print(f"\nTest {tc['id']}: {tc['name']}")
        print(f"  Query  : {tc['query']}")
        print(f"  Filters: {tc['filters']}")
        try:
            res = answer_question(
                query=tc["query"],
                filters=tc["filters"],
                index=index,
                metadata=metadata,
                model=model,
            )
            evidence_ids = res.get("evidence_ids", [])
            sufficiency = res.get("evidence_sufficiency", "NONE")
            answer = res.get("answer", "")
            passed = True
            failure_reasons = []
            if tc["expect_results"]:
                if len(evidence_ids) == 0:
                    passed = False
                    failure_reasons.append("Expected retrieved evidence, but got none.")
            else:
                if len(evidence_ids) > 0:
                    passed = False
                    failure_reasons.append(f"Expected no evidence retrieved, but got {len(evidence_ids)} records.")
            if "expect_sufficiency" in tc:
                if sufficiency not in tc["expect_sufficiency"]:
                    passed = False
                    failure_reasons.append(f"Expected sufficiency in {tc['expect_sufficiency']}, got '{sufficiency}'.")
            if tc.get("check_filter") and len(evidence_ids) > 0:
                from src.rag.retrieve import retrieve
                items = retrieve(
                    query=tc["query"],
                    filters=tc["filters"],
                    index=index,
                    metadata=metadata,
                    model=model,
                )
                for item in items:
                    if not tc["check_filter"](item):
                        passed = False
                        failure_reasons.append(
                            f"Filter check failed for item {item['complaint_id']}. "
                            f"Product: '{item['product']}', Company: '{item['company']}', "
                            f"Issue: '{item['issue']}', Date: '{item['date']}'."
                        )
                        break
            if passed:
                passed_count += 1
                status = "PASS"
            else:
                status = f"FAIL: {'; '.join(failure_reasons)}"
            print(f"  Status : {status}")
            print(f"  Sufficiency: {sufficiency}")
            print(f"  Evidence IDs count: {len(evidence_ids)}")
            results.append({
                "test_id": tc["id"],
                "name": tc["name"],
                "query": tc["query"],
                "filters": tc["filters"],
                "status": "PASS" if passed else "FAIL",
                "reasons": failure_reasons,
                "metrics": {
                    "evidence_sufficiency": sufficiency,
                    "evidence_count": len(evidence_ids),
                    "answer_length": len(answer),
                }
            })
        except Exception as ex:
            print(f"  Status : ERROR ({ex})")
            results.append({
                "test_id": tc["id"],
                "name": tc["name"],
                "query": tc["query"],
                "filters": tc["filters"],
                "status": "ERROR",
                "reasons": [str(ex)],
                "metrics": {}
            })
    pass_rate = passed_count / len(TEST_CASES)
    print(f"\n{'-'*55}")
    print(f"  Summary: {passed_count}/{len(TEST_CASES)} passed ({pass_rate*100:.1f}%)")
    print(f"{'-'*55}")
    report = {
        "passed_tests": passed_count,
        "total_tests": len(TEST_CASES),
        "pass_rate": pass_rate,
        "results": results
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {REPORT_PATH}")
    print(f"{'='*55}\n")
    return report
if __name__ == "__main__":
    run_eval()
