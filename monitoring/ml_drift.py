from pathlib import Path
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "bank_features.csv"
REPORT_DIR = ROOT / "monitoring" / "reports"
REPORT_PATH = REPORT_DIR / "ml_drift_report.html"
def generate_drifted_data(df: pd.DataFrame) -> pd.DataFrame:
    df_drifted = df.copy()
    if "age" in df_drifted.columns:
        df_drifted["age"] = df_drifted["age"] + 12.0
    if "balance" in df_drifted.columns:
        df_drifted["balance"] = df_drifted["balance"] * 2.5
    if "contact_intensity" in df_drifted.columns:
        df_drifted["contact_intensity"] = (df_drifted["contact_intensity"] + 2.0).clip(upper=20)
    for col in df_drifted.select_dtypes(include=[np.number]).columns:
        std = df_drifted[col].std()
        if std > 0:
            noise = np.random.normal(0, std * 0.05, size=len(df_drifted))
            df_drifted[col] = df_drifted[col] + noise
    return df_drifted
def main() -> None:
    print(f"\n{'='*55}")
    print("  Generating Feature Drift Report â€” ML Lane")
    print(f"{'='*55}")
    if not FEATURES_PATH.exists():
        print(f"Reference features not found at {FEATURES_PATH}.")
        print("Please build features first: python src/data_pipeline/features.py")
        sample_path = ROOT / "data" / "samples" / "bank_marketing_sample.csv"
        if sample_path.exists():
            print("Falling back to raw sample for reference data...")
            from src.data_pipeline.features import build_features
            df_raw = pd.read_csv(sample_path)
            q75 = float(df_raw["balance"].quantile(0.75)) if "balance" in df_raw.columns else None
            ref_df, _ = build_features(df_raw, q75_balance=q75)
        else:
            raise FileNotFoundError("Could not find any reference feature data.")
    else:
        print(f"Loading reference features from {FEATURES_PATH} ...")
        ref_df = pd.read_csv(FEATURES_PATH)
    print(f"Reference features shape: {ref_df.shape}")
    print("Generating synthetic drifted current dataset ...")
    curr_df = generate_drifted_data(ref_df)
    print(f"Current features shape: {curr_df.shape}")
    print("Running Evidently AI DataDrift report ...")
    report = Report(metrics=[
        DataDriftPreset()
    ])
    report.run(reference_data=ref_df, current_data=curr_df)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    print(f"Saved drift report to {REPORT_PATH}")
    print(f"{'='*55}\n")
if __name__ == "__main__":
    main()
