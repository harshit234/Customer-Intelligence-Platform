import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "samples"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "y"
LEAKAGE_COLS = ["duration"]
YES_NO_COLS = ["default", "housing", "loan"]
EDUCATION_ORDER = {
    "unknown": 0,
    "primary": 1,
    "secondary": 2,
    "tertiary": 3,
}
OHE_COLS = ["job", "marital", "contact", "poutcome"]
MONTH_TO_QUARTER: dict[str, int] = {
    "jan": 1, "feb": 1, "mar": 1,
    "apr": 2, "may": 2, "jun": 2,
    "jul": 3, "aug": 3, "sep": 3,
    "oct": 4, "nov": 4, "dec": 4,
}
AGE_BINS = [0, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
def build_features(
    df: pd.DataFrame,
    q75_balance: float | None = None,
    expected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    df = df.copy()
    if TARGET_COL in df.columns:
        y = (df[TARGET_COL] == "yes").astype(int)
        y.name = "subscribed"
        df = df.drop(columns=[TARGET_COL])
    else:
        y = None
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  [features] Dropped leakage columns: {cols_to_drop}")
    for col in YES_NO_COLS:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)
    if "pdays" in df.columns:
        df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
        df = df.drop(columns=["pdays"])
    if "campaign" in df.columns:
        df["contact_intensity"] = df["campaign"].clip(upper=20)
    if "balance" in df.columns:
        if q75_balance is None:
            q75_balance = df["balance"].quantile(0.75)
        df["high_balance"] = (df["balance"] > q75_balance).astype(int)
    if "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"],
            bins=AGE_BINS,
            labels=range(len(AGE_LABELS)),
            right=False,
        ).astype(float)
    if "month" in df.columns:
        df["quarter"] = (
            df["month"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(MONTH_TO_QUARTER)
            .fillna(2)
            .astype(int)
        )
        df = df.drop(columns=["month"])
    if "education" in df.columns:
        df["education"] = (
            df["education"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(EDUCATION_ORDER)
            .fillna(0)
            .astype(int)
        )
    present_ohe = [c for c in OHE_COLS if c in df.columns]
    if present_ohe:
        df = pd.get_dummies(df, columns=present_ohe, drop_first=False, dtype=int)
    remaining_cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if remaining_cats:
        print(f"  [features] Fallback label-encoding: {remaining_cats}")
        for col in remaining_cats:
            df[col] = df[col].astype("category").cat.codes
    X = df.astype(float)
    if expected_columns is not None:
        X = X.reindex(columns=expected_columns, fill_value=0.0)
    pos_rate_str = f"{y.mean()*100:.1f}%" if y is not None else "N/A"
    print(
        f"  [features] Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} cols | "
        f"positive rate: {pos_rate_str}"
    )
    return X, y
def build_features_single_row(row_dict: dict, manifest: dict) -> pd.DataFrame:
    """
    Transform a single customer record payload (dict) into a 1-row DataFrame
    aligned with the feature columns from the manifest to prevent skew.
    """
    df = pd.DataFrame([row_dict])
    expected_columns = manifest.get("columns", [])
    q75_balance = manifest.get("q75_balance", None)
    X, _ = build_features(df, q75_balance=q75_balance, expected_columns=expected_columns)
    return X
def load_raw(sample: bool = False) -> pd.DataFrame:
    """Load raw bank marketing CSV (full or sample)."""
    if sample:
        path = SAMPLE_DIR / "bank_marketing_sample.csv"
    else:
        path = RAW_DIR / "bank_marketing.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.\n"
            "Run: python src/data_pipeline/ingest.py"
        )
    return pd.read_csv(path)
def save_processed(
    X: pd.DataFrame,
    y: pd.Series | None,
    output_dir: Path,
    q75_balance: float | None = None,
) -> None:
    """Persist feature matrix, labels, and column/stats manifest to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_path = output_dir / "bank_features.csv"
    X.to_csv(X_path, index=False)
    print(f"  [features] Saved features -> {X_path}")
    if y is not None:
        y_path = output_dir / "bank_labels.csv"
        y.to_csv(y_path, index=False, header=True)
        print(f"  [features] Saved labels   -> {y_path}")
    import json
    manifest = {
        "columns": list(X.columns),
        "q75_balance": float(q75_balance) if q75_balance is not None else None,
    }
    manifest_path = output_dir / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [features] Saved manifest -> {manifest_path}")
def load_processed(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load previously saved processed features and labels."""
    X = pd.read_csv(processed_dir / "bank_features.csv")
    y = pd.read_csv(processed_dir / "bank_labels.csv").squeeze()
    return X, y
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering for Bank Marketing data."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use the 500-row sample (for CI).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory to write bank_features.csv and bank_labels.csv.",
    )
    args = parser.parse_args()
    print(f"\n{'='*55}")
    print("  Feature Engineering â€” Bank Marketing")
    print(f"{'='*55}")
    print(f"  Mode: {'sample (CI)' if args.sample else 'full dataset'}")
    df = load_raw(sample=args.sample)
    print(f"  Raw shape: {df.shape}")
    q75_balance = float(df["balance"].quantile(0.75)) if "balance" in df.columns else None
    X, y = build_features(df, q75_balance=q75_balance)
    save_processed(X, y, args.output_dir, q75_balance=q75_balance)
    print(f"\n  [OK] Done. Features ready at {args.output_dir}")
    print(f"{'='*55}\n")
if __name__ == "__main__":
    main()
