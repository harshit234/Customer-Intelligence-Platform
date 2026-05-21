import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "monitoring" / "reports" / "rag_request_log.jsonl"
REPORT_PATH = ROOT / "monitoring" / "reports" / "rag_monitor_report.json"
def create_synthetic_logs() -> None:
    """Pre-populate log file with realistic synthetic records to ensure the tool runs immediately."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    synthetic_entries = [
        {
            "timestamp": 1779951600.0,
            "query": "Equifax credit card billing dispute",
            "evidence_count": 3,
            "evidence_sufficiency": "HIGH",
            "max_score": 0.68,
            "refusal": False,
            "token_count": 450,
            "latency_ms": 1200.0
        },
        {
            "timestamp": 1779951630.0,
            "query": "Mortgage payment fees Wells Fargo",
            "evidence_count": 4,
            "evidence_sufficiency": "HIGH",
            "max_score": 0.72,
            "refusal": False,
            "token_count": 520,
            "latency_ms": 1450.0
        },
        {
            "timestamp": 1779951660.0,
            "query": "How to bake a chocolate cake?",
            "evidence_count": 0,
            "evidence_sufficiency": "NONE",
            "max_score": 0.0,
            "refusal": True,
            "token_count": 50,
            "latency_ms": 300.0
        },
        {
            "timestamp": 1779951690.0,
            "query": "What is the capital of France?",
            "evidence_count": 0,
            "evidence_sufficiency": "NONE",
            "max_score": 0.0,
            "refusal": True,
            "token_count": 45,
            "latency_ms": 280.0
        },
        {
            "timestamp": 1779951720.0,
            "query": "loan dispute",
            "evidence_count": 2,
            "evidence_sufficiency": "MEDIUM",
            "max_score": 0.52,
            "refusal": False,
            "token_count": 380,
            "latency_ms": 980.0
        },
    ]
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for entry in synthetic_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Created pre-populated logs file at {LOG_PATH}")
def main() -> None:
    print(f"\n{'='*55}")
    print("  Generating RAG Monitoring Report â€” LLM Lane")
    print(f"{'='*55}")
    if not LOG_PATH.exists():
        print(f"RAG log file not found at {LOG_PATH}. Pre-populating with synthetic data ...")
        create_synthetic_logs()
    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not entries:
        print("Error: RAG log file is empty.")
        return
    total_requests = len(entries)
    hit_count = sum(1 for e in entries if e.get("evidence_count", 0) > 0)
    empty_count = total_requests - hit_count
    refusal_count = sum(1 for e in entries if e.get("refusal", False) or e.get("evidence_sufficiency") == "NONE")
    avg_latency = sum(e.get("latency_ms", 0.0) for e in entries) / total_requests
    avg_tokens = sum(e.get("token_count", 0) for e in entries) / total_requests
    scores = [e.get("max_score", 0.0) for e in entries if e.get("evidence_count", 0) > 0]
    avg_max_score = sum(scores) / len(scores) if scores else 0.0
    report = {
        "total_requests": total_requests,
        "retrieval_hit_rate": round(hit_count / total_requests, 4),
        "empty_retrieval_count": empty_count,
        "refusal_rate": round(refusal_count / total_requests, 4),
        "average_latency_ms": round(avg_latency, 2),
        "average_token_count": round(avg_tokens, 2),
        "average_max_similarity_score": round(avg_max_score, 4),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved RAG monitoring report to {REPORT_PATH}")
    print(json.dumps(report, indent=2))
    print(f"{'='*55}\n")
if __name__ == "__main__":
    main()
