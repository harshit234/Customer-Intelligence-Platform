"""
ingest.py â€” Downloads and prepares raw data for both ML and RAG lanes.
Usage:
    python src/data_pipeline/ingest.py
    python src/data_pipeline/ingest.py --complaints-only
    python src/data_pipeline/ingest.py --ml-only
"""
import os
import sys
import argparse
import zipfile
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "samples"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
BANK_URL = (
    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
)
CFPB_URL = (
    "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
)
CFPB_SAMPLE_SIZE = int(os.getenv("MAX_COMPLAINT_SAMPLE", 10_000))
BANK_SAMPLE_SIZE = 500
def download_file(url: str, dest: Path) -> Path:
    """Stream-download a file with a progress bar."""
    print(f"Downloading {url.split('/')[-1]} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest
def ingest_bank_marketing() -> pd.DataFrame:
    """Download UCI Bank Marketing, return full DataFrame."""
    zip_path = RAW_DIR / "bank_marketing.zip"
    if not zip_path.exists():
        download_file(BANK_URL, zip_path)
    extract_dir = RAW_DIR / "bank_marketing"
    if not extract_dir.exists():
        print("Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
    for nested_zip in list(extract_dir.glob("*.zip")):
        with zipfile.ZipFile(nested_zip, "r") as z:
            z.extractall(extract_dir)
    csv_candidates = list(extract_dir.rglob("bank-full.csv"))
    if not csv_candidates:
        csv_candidates = list(extract_dir.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            "bank-full.csv not found inside the zip. "
            "Check the UCI download manually."
        )
    df = pd.read_csv(csv_candidates[0], sep=";")
    df = df[df["campaign"] <= 50]
    out_path = RAW_DIR / "bank_marketing.csv"
    df.to_csv(out_path, index=False)
    print(f"Bank Marketing saved -> {out_path}  ({len(df):,} rows)")
    sample_path = SAMPLE_DIR / "bank_marketing_sample.csv"
    df.sample(n=min(BANK_SAMPLE_SIZE, len(df)), random_state=42).to_csv(
        sample_path, index=False
    )
    print(f"Sample saved -> {sample_path}  ({BANK_SAMPLE_SIZE} rows)")
    return df
def ingest_cfpb_complaints() -> pd.DataFrame:
    """
    Download CFPB complaint CSV, sample to CFPB_SAMPLE_SIZE rows,
    keep only relevant columns, redact obvious PII placeholders.
    """
    zip_path = RAW_DIR / "complaints.csv.zip"
    raw_csv = RAW_DIR / "complaints_full.csv"
    if not raw_csv.exists():
        if not zip_path.exists():
            download_file(CFPB_URL, zip_path)
        print("Extracting complaints ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        extracted = RAW_DIR / "complaints.csv"
        if extracted.exists():
            extracted.rename(raw_csv)
    print(f"Reading complaints (sampling {CFPB_SAMPLE_SIZE:,} rows) ...")
    chunks = []
    chunk_size = 50_000
    collected = 0
    for chunk in pd.read_csv(raw_csv, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
        collected += len(chunk)
        if collected >= CFPB_SAMPLE_SIZE * 3:
            break
    df = pd.concat(chunks, ignore_index=True)
    COLS = [
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Consumer complaint narrative",
        "Company",
        "State",
        "ZIP code",
        "Complaint ID",
        "Company response to consumer",
    ]
    available = [c for c in COLS if c in df.columns]
    df = df[available].copy()
    df = df.dropna(subset=["Consumer complaint narrative"])
    df = df[df["Consumer complaint narrative"].str.len() > 50]
    if len(df) > CFPB_SAMPLE_SIZE:
        df = df.sample(n=CFPB_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    df["Consumer complaint narrative"] = (
        df["Consumer complaint narrative"]
        .str.replace(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED]", regex=True)
        .str.replace(r"\b\d{9,}\b", "[REDACTED]", regex=True)
    )
    out_path = RAW_DIR / "cfpb_complaints_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"CFPB sample saved -> {out_path}  ({len(df):,} rows)")
    git_sample = df.drop(columns=["Consumer complaint narrative"], errors="ignore")
    git_sample = git_sample.head(200)
    git_sample_path = SAMPLE_DIR / "cfpb_metadata_sample.csv"
    git_sample.to_csv(git_sample_path, index=False)
    print(f"Git metadata sample saved -> {git_sample_path}  (200 rows, no narratives)")
    return df
def main():
    parser = argparse.ArgumentParser(description="Ingest project datasets.")
    parser.add_argument("--ml-only", action="store_true")
    parser.add_argument("--complaints-only", action="store_true")
    args = parser.parse_args()
    if args.complaints_only:
        ingest_cfpb_complaints()
    elif args.ml_only:
        ingest_bank_marketing()
    else:
        ingest_bank_marketing()
        ingest_cfpb_complaints()
    print("\n[OK] Ingestion complete.")
if __name__ == "__main__":
    main()
