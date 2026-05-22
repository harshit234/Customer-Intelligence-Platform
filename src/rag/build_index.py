
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# -- Paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw" / "cfpb_complaints_sample.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "faiss_index.bin"
METADATA_PATH = PROCESSED_DIR / "faiss_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RANDOM_STATE = 42

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from CFPB complaint narratives."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of complaints to index (default: 1000). Use -1 for all.",
    )
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  Building FAISS Index — CFPB Complaints")
    print(f"{'='*55}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"CFPB complaints file not found at {RAW_DATA_PATH}")

    # Load complaints
    print(f"Loading raw complaints from {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"Raw dataset shape: {df.shape}")

    # Filter out rows without narrative
    narrative_col = "Consumer complaint narrative"
    df = df.dropna(subset=[narrative_col])
    df = df[df[narrative_col].str.strip() != ""]
    print(f"Complaints with narrative: {len(df)}")

    # Sample if requested
    if args.sample_size > 0 and len(df) > args.sample_size:
        print(f"Sampling {args.sample_size} records (seed {RANDOM_STATE}) ...")
        df = df.sample(n=args.sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print(f"Number of complaints to embed: {len(df)}")

    # Load SentenceTransformer model
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dimension}")

    # Extract narratives and metadata
    narratives = df[narrative_col].astype(str).tolist()
    
    # Generate embeddings
    print("Generating embeddings ...")
    embeddings = model.encode(
        narratives,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # L2-normalize embeddings for cosine similarity via IndexFlatIP
    print("L2-normalizing embeddings ...")
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    print("Building FAISS index (IndexFlatIP) ...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    print(f"Index size (number of vectors): {index.ntotal}")

    # Save index
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index to {INDEX_PATH}")

    # Extract metadata matching the index rows
    metadata = []
    for idx, row in df.iterrows():
        metadata.append({
            "complaint_id": str(row.get("Complaint ID", "")),
            "product": str(row.get("Product", "")),
            "company": str(row.get("Company", "")),
            "date": str(row.get("Date received", "")),
            "issue": str(row.get("Issue", "")),
            "narrative": str(row.get(narrative_col, ""))
        })

    # Save metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")

    print(f"\n  [OK] Done. FAISS index and metadata successfully generated.")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
