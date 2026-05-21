from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_pipeline.features import (
    PROCESSED_DIR,
    build_features,
    load_processed,
    load_raw,
    save_processed,
)
from src.training.evaluate import (
    GateResult,
    PromotionGate,
    business_reading,
    classification_report_str,
    compute_metrics,
    threshold_analysis_table,
)
DEFAULT_EXPERIMENT = "CampaignConversion"
MODEL_NAME         = "CampaignConversion"
TEST_SIZE          = 0.20
RANDOM_STATE       = 42
CV_FOLDS           = 5
DECISION_THRESHOLD = 0.40
XGB_PARAMS: dict = {
    "n_estimators":       400,
    "max_depth":          5,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   3,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "eval_metric":        "logloss",
    "random_state":       RANDOM_STATE,
    "n_jobs":             -1,
}
LATENCY_WARMUP_N  = 50
LATENCY_MEASURE_N = 200
def get_data(sample: bool) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix and labels.
    Prefers pre-processed files in data/processed/.
    If not found, runs the feature pipeline on the fly.
    """
    features_path = PROCESSED_DIR / "bank_features.csv"
    labels_path   = PROCESSED_DIR / "bank_labels.csv"
    manifest_path = PROCESSED_DIR / "feature_manifest.json"
    if features_path.exists() and labels_path.exists() and manifest_path.exists() and not sample:
        print("  [train] Loading pre-processed features ...")
        return load_processed()
    print("  [train] Pre-processed data not found - running feature pipeline ...")
    df = load_raw(sample=sample)
    q75_balance = float(df["balance"].quantile(0.75)) if "balance" in df.columns else None
    X, y = build_features(df, q75_balance=q75_balance)
    if not sample:
        save_processed(X, y, PROCESSED_DIR, q75_balance=q75_balance)
    return X, y
def build_baseline(scale_pos_weight: float) -> Pipeline:
    """
    Baseline: LogisticRegression with class_weight='balanced' inside a
    StandardScaler pipeline.  Fast, interpretable, always deployable.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=RANDOM_STATE,
            C=0.1,
            solver="lbfgs",
        )),
    ])
def build_improved(scale_pos_weight: float) -> Pipeline:
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            f"Missing dependency: {exc}. "
            "Run: pip install xgboost imbalanced-learn"
        )
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    xgb_clf = xgb.XGBClassifier(**params)
    calibrated = CalibratedClassifierCV(xgb_clf, method="isotonic", cv=3)
    return ImbPipeline([
        ("smote",     SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE)),
        ("scaler",    StandardScaler()),
        ("clf",       calibrated),
    ])
def measure_latency_ms(model, X_sample: pd.DataFrame) -> float:
    """
    Measure median single-row inference latency in milliseconds.
    Runs LATENCY_WARMUP_N predictions to warm up JIT/cache, then
    LATENCY_MEASURE_N timed predictions on random single rows.
    Returns
    -------
    float - median latency in milliseconds.
    """
    row = X_sample.iloc[:1]
    for _ in range(LATENCY_WARMUP_N):
        model.predict_proba(row)
    times_ms = []
    for _ in range(LATENCY_MEASURE_N):
        t0 = time.perf_counter()
        model.predict_proba(row)
        times_ms.append((time.perf_counter() - t0) * 1_000)
    return float(np.median(times_ms))
def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a styled confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
def _save_feature_importance(model, feature_names: list[str], out_path: Path) -> None:
    """Extract and save feature importances (XGBoost / calibrated wrapper)."""
    raw = model
    if hasattr(raw, "named_steps"):
        raw = raw.named_steps.get("clf", raw)
    if hasattr(raw, "estimator"):
        raw = raw.estimator
    if hasattr(raw, "calibrated_classifiers_"):
        raw = raw.calibrated_classifiers_[0].estimator
    if not hasattr(raw, "feature_importances_"):
        return
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": raw.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(out_path, index=False)
def _train_one(
    model,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str,
    sample_mode: bool,
    extra_params: dict | None = None,
) -> tuple[str, dict[str, float], float]:
    import mlflow
    import mlflow.sklearn
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=f"{model_type}_{'sample' if sample_mode else 'full'}"
    ) as run:
        run_id = run.info.run_id
        print(f"\n  {'-'*52}")
        print(f"  MLflow run : {run_id[:12]}...  [{model_type}]")
        print(f"  {'-'*52}")
        params: dict = {
            "model_type":            model_type,
            "cv_folds":              CV_FOLDS,
            "test_size":             TEST_SIZE,
            "decision_threshold":    DECISION_THRESHOLD,
            "sample_mode":           sample_mode,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        print(f"  Running {CV_FOLDS}-fold stratified CV ...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_roc_auc_std",  float(cv_scores.std()))
        print(f"  CV ROC-AUC : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print("  Fitting on full train split ...")
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_proba, threshold=DECISION_THRESHOLD)
        mlflow.log_metrics(metrics)
        for name, val in metrics.items():
            print(f"  {name:<20}: {val:.4f}")
        latency_ms = measure_latency_ms(model, X_test)
        mlflow.log_metric("latency_ms_median", round(latency_ms, 3))
        print(f"  latency (median)    : {latency_ms:.2f} ms")
        tmp_dir = ROOT / "_mlflow_tmp"
        tmp_dir.mkdir(exist_ok=True)
        report = classification_report_str(
            y_test.values, y_proba, threshold=DECISION_THRESHOLD
        )
        (tmp_dir / "classification_report.txt").write_text(report)
        mlflow.log_artifact(str(tmp_dir / "classification_report.txt"))
        print(report)
        biz = business_reading(y_test.values, y_proba, threshold=DECISION_THRESHOLD)
        (tmp_dir / "business_reading.txt").write_text(biz)
        mlflow.log_artifact(str(tmp_dir / "business_reading.txt"))
        print(biz)
        tbl = threshold_analysis_table(y_test.values, y_proba)
        tbl.to_csv(tmp_dir / "threshold_analysis.csv", index=False)
        mlflow.log_artifact(str(tmp_dir / "threshold_analysis.csv"))
        print("\n  Threshold analysis:")
        print(tbl.to_string(index=False))
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
        _save_confusion_matrix(
            y_test.values, y_pred,
            tmp_dir / "confusion_matrix.png",
            title=model_type,
        )
        mlflow.log_artifact(str(tmp_dir / "confusion_matrix.png"))
        _save_feature_importance(model, list(X_train.columns), tmp_dir / "feature_importance.csv")
        fi_path = tmp_dir / "feature_importance.csv"
        if fi_path.exists():
            mlflow.log_artifact(str(fi_path))
        mlflow.sklearn.log_model(model, artifact_path="model")
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()
    return run_id, metrics, latency_ms
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Baseline (LogReg) + Improved (XGBoost+SMOTE+Calibration), "
            "then run the 3-axis PromotionGate."
        )
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use 500-row sample for CI smoke tests.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Train both models but skip gate / registration.",
    )
    args = parser.parse_args()
    print(f"\n{'='*55}")
    print("  Training - Bank Marketing Conversion Model")
    print(f"{'='*55}")
    print(f"  Experiment : {args.experiment_name}")
    print(f"  Mode       : {'sample (CI)' if args.sample else 'full dataset'}")
    X, y = get_data(sample=args.sample)
    print(f"  Dataset    : {X.shape[0]:,} rows x {X.shape[1]} features")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"  Train pos  : {pos} | neg: {neg} | scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"\n{'-'*55}")
    print("  [1/2] Training Baseline (LogisticRegression) ...")
    baseline_model = build_baseline(scale_pos_weight)
    baseline_run_id, baseline_metrics, baseline_latency = _train_one(
        model=baseline_model,
        model_type="LogisticRegression",
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        experiment_name=args.experiment_name,
        sample_mode=args.sample,
        extra_params={"class_weight_strategy": "balanced", "solver": "lbfgs"},
    )
    print(f"\n{'-'*55}")
    print("  [2/2] Training Improved (XGBoost + SMOTE + Calibration) ...")
    improved_model = build_improved(scale_pos_weight)
    improved_run_id, improved_metrics, improved_latency = _train_one(
        model=improved_model,
        model_type="XGBoost_SMOTE_Calibrated",
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        experiment_name=args.experiment_name,
        sample_mode=args.sample,
        extra_params={
            **{k: v for k, v in XGB_PARAMS.items() if k not in ("eval_metric",)},
            "smote_sampling_strategy": 0.3,
            "calibration_method": "isotonic",
            "scale_pos_weight": round(scale_pos_weight, 4),
        },
    )
    if args.no_promote:
        print("\n  [gate] Skipped (--no-promote).")
        print(f"  Baseline run : {baseline_run_id}")
        print(f"  Improved run : {improved_run_id}")
        sys.exit(0)
    print(f"\n{'='*55}")
    print("  Promotion Gate (3-axis check)")
    print(f"{'='*55}")
    gate = PromotionGate(sample_mode=args.sample)
    gate_result: GateResult = gate.compare(
        improved_metrics=improved_metrics,
        baseline_metrics=baseline_metrics,
        improved_latency_ms=improved_latency,
    )
    print(gate_result)
    import mlflow
    if gate_result.winner == "improved":
        winner_run_id  = improved_run_id
        winner_label   = "XGBoost+SMOTE+Calibration"
    else:
        winner_run_id  = baseline_run_id
        winner_label   = "LogisticRegression (baseline fallback)"
    print(f"\n  Registering winner: {winner_label}")
    gate.promote_mlflow_model(
        run_id=winner_run_id,
        model_name=MODEL_NAME,
        gate_result=gate_result,
    )
    print(f"\n{'='*55}")
    print(f"  [OK] Training complete")
    print(f"  Baseline run  : {baseline_run_id[:12]}...")
    print(f"  Improved run  : {improved_run_id[:12]}...")
    print(f"  Winner        : {winner_label}")
    print(f"  Registered as : {MODEL_NAME} -> Production")
    print(f"{'='*55}\n")
if __name__ == "__main__":
    main()
