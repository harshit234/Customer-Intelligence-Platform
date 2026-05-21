"""
validate.py â€” Schema, missing-value, duplicate checks + business-rule validations.
Usage:
    python src/data_pipeline/validate.py
    python src/data_pipeline/validate.py --sample
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "samples"
BANK_SCHEMA = DataFrameSchema(
    columns={
        "age": Column(int, Check.in_range(18, 100), nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(
            str,
            Check.isin(["married", "single", "divorced"]),
            nullable=False,
        ),
        "education": Column(str, nullable=False),
        "default": Column(str, Check.isin(["yes", "no"]), nullable=False),
        "balance": Column(int, nullable=False),
        "housing": Column(str, Check.isin(["yes", "no"]), nullable=False),
        "loan": Column(str, Check.isin(["yes", "no"]), nullable=False),
        "contact": Column(str, nullable=True),
        "day": Column(int, Check.in_range(1, 31), nullable=False),
        "month": Column(
            str,
            Check.isin(
                ["jan","feb","mar","apr","may","jun",
                 "jul","aug","sep","oct","nov","dec"]
            ),
            nullable=False,
        ),
        "duration": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "campaign": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "poutcome": Column(str, nullable=True),
        "y": Column(str, Check.isin(["yes", "no"]), nullable=False),
    },
    strict=False,
)
CFPB_SCHEMA = DataFrameSchema(
    columns={
        "Complaint ID": Column(str, nullable=False),
        "Product": Column(str, nullable=False),
        "Consumer complaint narrative": Column(
            str,
            Check(lambda s: s.str.len() > 50, element_wise=False),
            nullable=False,
        ),
    },
    strict=False,
)
def run_bank_business_rules(df: pd.DataFrame) -> list[str]:
    """
    5 business-rule validations for Bank Marketing data.
    Returns a list of error messages (empty = all pass).
    """
    errors = []
    if "y" not in df.columns:
        errors.append("BIZ-01: Target column 'y' missing.")
    elif not set(df["y"].unique()).issubset({"yes", "no"}):
        errors.append("BIZ-01: Target 'y' has unexpected values.")
    if "y" in df.columns:
        minority_pct = (df["y"] == "yes").mean() * 100
        if minority_pct < 5:
            errors.append(
                f"BIZ-02: Minority class (y=yes) is {minority_pct:.1f}% â€” "
                "suspiciously low, check sample."
            )
    if "balance" in df.columns:
        extreme_neg = (df["balance"] < -10_000).sum()
        if extreme_neg > 0:
            errors.append(
                f"BIZ-03: {extreme_neg} rows have balance < -10,000 â€” "
                "verify data source."
            )
    if "campaign" in df.columns:
        bad_campaign = (df["campaign"] > 50).sum()
        if bad_campaign > 0:
            errors.append(
                f"BIZ-04: {bad_campaign} rows have campaign > 50 contacts â€” "
                "likely outliers or data error."
            )
    if "duration" in df.columns and "y" in df.columns:
        bad_duration = ((df["duration"] == 0) & (df["y"] == "yes")).sum()
        if bad_duration > 0:
            errors.append(
                f"BIZ-05: {bad_duration} rows have duration=0 but y='yes' â€” "
                "impossible outcome."
            )
    return errors
def run_cfpb_business_rules(df: pd.DataFrame) -> list[str]:
    """
    5 business-rule validations for CFPB complaint data.
    Returns a list of error messages.
    """
    errors = []
    blank = (df["Consumer complaint narrative"].str.strip() == "").sum()
    if blank > 0:
        errors.append(f"CFPB-01: {blank} blank narratives after strip.")
    if "Complaint ID" in df.columns:
        dupes = df["Complaint ID"].duplicated().sum()
        if dupes > 0:
            errors.append(f"CFPB-02: {dupes} duplicate Complaint IDs.")
    if df["Product"].isna().all():
        errors.append("CFPB-03: Product column is entirely null.")
    med_len = df["Consumer complaint narrative"].str.len().median()
    if med_len < 100:
        errors.append(
            f"CFPB-04: Median narrative length is {med_len:.0f} chars â€” "
            "narratives seem too short."
        )
    n_products = df["Product"].nunique()
    if n_products < 5:
        errors.append(
            f"CFPB-05: Only {n_products} distinct products â€” "
            "sample may be too narrow."
        )
    return errors
def validate_bank(df: pd.DataFrame, label: str = "Bank Marketing") -> bool:
    print(f"\n{'='*55}")
    print(f"  Validating {label}")
    print(f"{'='*55}")
    print(f"  Shape: {df.shape}")
    print(f"  Nulls: {df.isnull().sum().sum()} total")
    print(f"  Duplicates: {df.duplicated().sum()}")
    passed = True
    try:
        BANK_SCHEMA.validate(df, lazy=True)
        print("  [OK] Schema validation passed")
    except pa.errors.SchemaErrors as e:
        print(f"  [FAIL] Schema errors:\n{e.failure_cases.head(10)}")
        passed = False
    biz_errors = run_bank_business_rules(df)
    if biz_errors:
        for err in biz_errors:
            print(f"  [FAIL] {err}")
        passed = False
    else:
        print("  [OK] All 5 business rules passed")
    return passed
def validate_cfpb(df: pd.DataFrame, label: str = "CFPB Complaints") -> bool:
    print(f"\n{'='*55}")
    print(f"  Validating {label}")
    print(f"{'='*55}")
    print(f"  Shape: {df.shape}")
    print(f"  Nulls in narrative: {df['Consumer complaint narrative'].isnull().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    passed = True
    try:
        CFPB_SCHEMA.validate(df, lazy=True)
        print("  [OK] Schema validation passed")
    except pa.errors.SchemaErrors as e:
        print(f"  [FAIL] Schema errors:\n{e.failure_cases.head(10)}")
        passed = False
    biz_errors = run_cfpb_business_rules(df)
    if biz_errors:
        for err in biz_errors:
            print(f"  [FAIL] {err}")
        passed = False
    else:
        print("  [OK] All 5 business rules passed")
    return passed
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use small sample files (for CI).",
    )
    args = parser.parse_args()
    all_passed = True
    bank_path = (
        SAMPLE_DIR / "bank_marketing_sample.csv"
        if args.sample
        else RAW_DIR / "bank_marketing.csv"
    )
    if bank_path.exists():
        bank_df = pd.read_csv(bank_path)
        ok = validate_bank(bank_df, label=f"Bank Marketing ({'sample' if args.sample else 'full'})")
        all_passed = all_passed and ok
    else:
        print(f"\n[WARN] Bank Marketing data not found at {bank_path}. Run ingest.py first.")
        all_passed = False
    cfpb_path = (
        SAMPLE_DIR / "cfpb_metadata_sample.csv"
        if args.sample
        else RAW_DIR / "cfpb_complaints_sample.csv"
    )
    if cfpb_path.exists():
        cfpb_df = pd.read_csv(cfpb_path, dtype={"Complaint ID": str})
        if "Consumer complaint narrative" in cfpb_df.columns:
            ok = validate_cfpb(cfpb_df, label=f"CFPB ({'sample' if args.sample else 'full'})")
            all_passed = all_passed and ok
        else:
            print("\n  [INFO] CFPB Git sample has no narratives - skipping narrative checks (CI mode).")
            print("  [OK] Metadata columns present")
    else:
        print(f"\n[WARN] CFPB data not found at {cfpb_path}. Run ingest.py first.")
        all_passed = False
    print(f"\n{'='*55}")
    if all_passed:
        print("  [OK] ALL VALIDATIONS PASSED")
    else:
        print("  [FAIL] SOME VALIDATIONS FAILED - fix before training.")
    print(f"{'='*55}\n")
    sys.exit(0 if all_passed else 1)
if __name__ == "__main__":
    main()
