import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "faiss_index.bin"
METADATA_PATH = PROCESSED_DIR / "faiss_metadata.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_INDEX = None
_METADATA = None
_MODEL = None
def load_resources():
    """Lazy load embedding model, FAISS index, and metadata."""
    global _INDEX, _METADATA, _MODEL
    if _MODEL is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if _INDEX is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. Run build_index.py first."
            )
        print(f"Loading FAISS index from {INDEX_PATH} ...")
        _INDEX = faiss.read_index(str(INDEX_PATH))
    if _METADATA is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {METADATA_PATH}. Run build_index.py first."
            )
        print(f"Loading metadata from {METADATA_PATH} ...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _METADATA = json.load(f)
    return _INDEX, _METADATA, _MODEL
def retrieve(
    query: str,
    top_k: int = 5,
    threshold: float = 0.45,
    filters: dict | None = None,
    index=None,
    metadata=None,
    model=None,
) -> list[dict]:
    if index is None or metadata is None or model is None:
        r_index, r_metadata, r_model = load_resources()
        index = index or r_index
        metadata = metadata or r_metadata
        model = model or r_model
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    candidate_k = max(top_k * 10, 50)
    candidate_k = min(candidate_k, index.ntotal)
    if candidate_k == 0:
        return []
    scores, indices = index.search(query_vector.astype("float32"), k=candidate_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        if score < threshold:
            continue
        item = metadata[idx]
        if filters:
            match = True
            if "product" in filters and filters["product"]:
                if filters["product"].lower() not in item.get("product", "").lower():
                    match = False
            if "company" in filters and filters["company"]:
                if filters["company"].lower() not in item.get("company", "").lower():
                    match = False
            if "issue" in filters and filters["issue"]:
                if filters["issue"].lower() not in item.get("issue", "").lower():
                    match = False
            if "date_start" in filters and filters["date_start"]:
                if item.get("date", "") < filters["date_start"]:
                    match = False
            if "date_end" in filters and filters["date_end"]:
                if item.get("date", "") > filters["date_end"]:
                    match = False
            if not match:
                continue
        results.append({
            "complaint_id": item["complaint_id"],
            "score": float(score),
            "narrative": item["narrative"],
            "product": item["product"],
            "company": item["company"],
            "date": item["date"],
            "issue": item["issue"],
        })
        if len(results) >= top_k:
            break
    return results
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query vector store.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=3, help="Max results")
    parser.add_argument("--threshold", type=float, default=0.45, help="Similarity threshold")
    args = parser.parse_args()
    try:
        res = retrieve(args.query, top_k=args.top_k, threshold=args.threshold)
        print(f"\nFound {len(res)} results for query: '{args.query}'\n")
        for i, item in enumerate(res):
            print(f"[{i+1}] ID: {item['complaint_id']} | Score: {item['score']:.4f} | Date: {item['date']}")
            print(f"    Product: {item['product']} | Company: {item['company']}")
            print(f"    Snippet: {item['narrative'][:120]}...\n")
    except Exception as e:
        print(f"Error querying: {e}")
