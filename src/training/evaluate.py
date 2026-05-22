from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }
def classification_report_str(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    target_names: list[str] | None = None,
) -> str:
    """
    Return a formatted sklearn classification report string,
    suitable for logging as an MLflow text artifact.
    """
    if target_names is None:
        target_names = ["not_subscribed", "subscribed"]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_proba, threshold)
    report_lines = [
        "=" * 60,
        "  Classification Report",
        "=" * 60,
        classification_report(y_true, y_pred, target_names=target_names),
        "-" * 60,
        f"  ROC-AUC          : {metrics['roc_auc']:.4f}",
        f"  Avg Precision    : {metrics['avg_precision']:.4f}",
        f"  Brier Score      : {metrics['brier_score']:.4f}  (lower = better)",
        "=" * 60,
    ]
    return "\n".join(report_lines)
def threshold_analysis_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
    population_size: int | None = None,
) -> "pd.DataFrame":
    import pandas as pd
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    n = population_size or len(y_true)
    pos_rate = float(y_true.mean())
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, zero_division=0))
        called_pct = float(y_pred.mean()) * 100
        predicted_subs = int(round(prec * y_pred.sum()))
        rows.append({
            "threshold":              round(t, 2),
            "precision":             round(prec, 4),
            "recall":                round(rec, 4),
            "f1":                    round(f1, 4),
            "customers_contacted_%": round(called_pct, 1),
            "predicted_subscribers": predicted_subs,
        })
    return pd.DataFrame(rows)
def business_reading(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.40,
    population_size: int | None = None,
    cost_per_call: float = 5.0,
    revenue_per_subscription: float = 200.0,
) -> str:
    n = population_size or len(y_true)
    y_pred = (y_proba >= threshold).astype(int)
    n_contacted     = int(y_pred.sum())
    prec            = float(precision_score(y_true, y_pred, zero_division=0))
    rec             = float(recall_score(y_true, y_pred, zero_division=0))
    f1              = float(f1_score(y_true, y_pred, zero_division=0))
    est_subscribers = int(round(prec * n_contacted))
    total_positives = int(y_true.sum())
    missed          = total_positives - est_subscribers
    total_cost    = n_contacted * cost_per_call
    total_revenue = est_subscribers * revenue_per_subscription
    net           = total_revenue - total_cost
    lines = [
        "",
        "=" * 60,
        "  Business Reading",
        "=" * 60,
        f"  Decision threshold         : {threshold:.2f}",
        f"  Scoring population         : {n:,} customers",
        f"  Customers flagged / called : {n_contacted:,}  "
        f"({n_contacted/n*100:.1f}% of population)",
        f"  Expected subscribers gained: {est_subscribers:,}  "
        f"(precision={prec:.1%})",
        f"  Subscribers missed         : {missed:,}  "
        f"(recall={rec:.1%})",
        "-" * 60,
        "  Financial estimate (illustrative)",
        f"    Call cost  ({n_contacted:,} Ã— ${cost_per_call:.0f})  : "
        f"${total_cost:,.0f}",
        f"    Revenue    ({est_subscribers:,} Ã— ${revenue_per_subscription:.0f}): "
        f"${total_revenue:,.0f}",
        f"    Net value                : ${net:,.0f}",
        "=" * 60,
    ]
    return "\n".join(lines)
@dataclass
class GateResult:
    """Outcome of a PromotionGate evaluation."""
    promoted: bool
    reason: str
    winner: str = ""
    challenger_metrics: dict[str, float] = field(default_factory=dict)
    champion_metrics: dict[str, float] = field(default_factory=dict)
    gate_details: list[str] = field(default_factory=list)
    def __str__(self) -> str:
        status = "[PROMOTED]" if self.promoted else "[BLOCKED]"
        lines = [
            "",
            "=" * 60,
            f"  Promotion Gate â€” {status}",
            "=" * 60,
            f"  Reason : {self.reason}",
        ]
        if self.winner:
            lines.append(f"  Winner : {self.winner}")
        if self.gate_details:
            lines.append("  Details:")
            for d in self.gate_details:
                lines.append(f"    {d}")
        if self.challenger_metrics:
            lines.append("  Improved (XGBoost) metrics:")
            for k, v in self.challenger_metrics.items():
                lines.append(f"    {k:<22}: {v:.4f}")
        if self.champion_metrics:
            lines.append("  Baseline (LogReg) metrics:")
            for k, v in self.champion_metrics.items():
                lines.append(f"    {k:<22}: {v:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
class PromotionGate:
    PR_AUC_DELTA:    float = 0.03
    F1_DROP_MAX:     float = 0.02
    LATENCY_MS_MAX:  float = 200.0
    DEFAULT_THRESHOLDS: dict[str, float] = {
        "roc_auc":       0.72,
        "avg_precision": 0.45,
        "f1":            0.40,
    }
    SAMPLE_THRESHOLDS: dict[str, float] = {
        "roc_auc":       0.55,
        "avg_precision": 0.25,
        "f1":            0.20,
    }
    RELATIVE_DELTA: float = 0.01
    PRIMARY_METRIC: str = "roc_auc"
    def __init__(
        self,
        pr_auc_delta: float | None = None,
        f1_drop_max: float | None = None,
        latency_ms_max: float | None = None,
        absolute_thresholds: dict[str, float] | None = None,
        relative_delta: float | None = None,
        primary_metric: str | None = None,
        sample_mode: bool = False,
    ) -> None:
        self.pr_auc_delta   = pr_auc_delta   if pr_auc_delta   is not None else self.PR_AUC_DELTA
        self.f1_drop_max    = f1_drop_max    if f1_drop_max    is not None else self.F1_DROP_MAX
        self.latency_ms_max = latency_ms_max if latency_ms_max is not None else self.LATENCY_MS_MAX
        if sample_mode:
            self.pr_auc_delta   = 0.01
            self.f1_drop_max    = 0.05
            self.latency_ms_max = 500.0
            self.thresholds     = self.SAMPLE_THRESHOLDS
        else:
            self.thresholds = absolute_thresholds or self.DEFAULT_THRESHOLDS
        self.relative_delta = relative_delta if relative_delta is not None else self.RELATIVE_DELTA
        self.primary_metric = primary_metric or self.PRIMARY_METRIC
        self.sample_mode    = sample_mode
    def compare(
        self,
        improved_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        improved_latency_ms: float | None = None,
    ) -> GateResult:
        details: list[str] = []
        failures: list[str] = []
        imp_ap  = improved_metrics.get("avg_precision", 0.0)
        base_ap = baseline_metrics.get("avg_precision", 0.0)
        delta_ap = imp_ap - base_ap
        ok1 = delta_ap >= self.pr_auc_delta
        mark = "PASS" if ok1 else "FAIL"
        details.append(
            f"[{mark}] PR-AUC delta: improved={imp_ap:.4f}  baseline={base_ap:.4f}  "
            f"delta={delta_ap:+.4f}  required>={self.pr_auc_delta:.2f}"
        )
        if not ok1:
            failures.append(
                f"PR-AUC delta {delta_ap:+.4f} < required {self.pr_auc_delta:.2f}"
            )
        imp_f1  = improved_metrics.get("f1", 0.0)
        base_f1 = baseline_metrics.get("f1", 0.0)
        f1_drop = base_f1 - imp_f1
        ok2 = f1_drop <= self.f1_drop_max
        mark = "PASS" if ok2 else "FAIL"
        details.append(
            f"[{mark}] F1 drop:     improved={imp_f1:.4f}  baseline={base_f1:.4f}  "
            f"drop={f1_drop:+.4f}  allowed<={self.f1_drop_max:.2f}"
        )
        if not ok2:
            failures.append(
                f"F1 drop {f1_drop:+.4f} > allowed {self.f1_drop_max:.2f}"
            )
        if improved_latency_ms is not None:
            ok3 = improved_latency_ms <= self.latency_ms_max
            mark = "PASS" if ok3 else "FAIL"
            details.append(
                f"[{mark}] Latency:    {improved_latency_ms:.1f}ms  "
                f"allowed<={self.latency_ms_max:.0f}ms"
            )
            if not ok3:
                failures.append(
                    f"Latency {improved_latency_ms:.1f}ms > allowed {self.latency_ms_max:.0f}ms"
                )
        else:
            details.append("[SKIP] Latency check â€” no measurement provided")
        if failures:
            return GateResult(
                promoted=True,
                reason=f"Improved blocked ({len(failures)} check(s) failed) â€” baseline promoted.",
                winner="baseline",
                challenger_metrics=improved_metrics,
                champion_metrics=baseline_metrics,
                gate_details=details,
            )
        return GateResult(
            promoted=True,
            reason="All 3 gate checks passed â€” improved model promoted.",
            winner="improved",
            challenger_metrics=improved_metrics,
            champion_metrics=baseline_metrics,
            gate_details=details,
        )
    def evaluate(
        self,
        challenger_metrics: dict[str, float],
        champion_metrics: dict[str, float] | None = None,
    ) -> GateResult:
        details: list[str] = []
        failures: list[str] = []
        for metric, floor in self.thresholds.items():
            value = challenger_metrics.get(metric)
            if value is None:
                failures.append(f"Metric '{metric}' missing from challenger results.")
                details.append(f"[MISSING] {metric}")
                continue
            ok = value >= floor
            mark = "PASS" if ok else "FAIL"
            details.append(
                f"[{mark}] {metric:<18} challenger={value:.4f}  floor={floor:.4f}"
            )
            if not ok:
                failures.append(
                    f"Absolute gate: {metric} = {value:.4f} < floor {floor:.4f}"
                )
        if failures:
            return GateResult(
                promoted=False,
                reason=f"Absolute gate failed â€” {len(failures)} check(s) not met.",
                challenger_metrics=challenger_metrics,
                champion_metrics=champion_metrics or {},
                gate_details=details,
            )
        if champion_metrics:
            champ_val = champion_metrics.get(self.primary_metric)
            chal_val  = challenger_metrics.get(self.primary_metric)
            if champ_val is not None and chal_val is not None:
                delta = chal_val - champ_val
                required = self.relative_delta
                ok = delta >= required
                mark = "PASS" if ok else "FAIL"
                details.append(
                    f"[{mark}] relative {self.primary_metric}: "
                    f"Î”={delta:+.4f}  required>={required:.4f}"
                )
                if not ok:
                    return GateResult(
                        promoted=False,
                        reason=(
                            f"Relative gate failed: {self.primary_metric} improvement "
                            f"{delta:+.4f} < required {required:.4f}"
                        ),
                        challenger_metrics=challenger_metrics,
                        champion_metrics=champion_metrics,
                        gate_details=details,
                    )
            else:
                details.append(
                    f"[SKIP] relative gate â€” {self.primary_metric} not available in champion metrics"
                )
        else:
            details.append("[SKIP] relative gate â€” no champion metrics provided (first run)")
        return GateResult(
            promoted=True,
            reason="All gate checks passed.",
            challenger_metrics=challenger_metrics,
            champion_metrics=champion_metrics or {},
            gate_details=details,
        )
    def evaluate_mlflow_runs(
        self,
        challenger_run_id: str,
        champion_run_id: str | None = None,
    ) -> GateResult:
        """
        Load metrics from MLflow run IDs and run the two-tier gate.
        Parameters
        ----------
        challenger_run_id : MLflow run ID for the challenger model.
        champion_run_id   : MLflow run ID for the current champion. Optional.
        """
        import mlflow
        client = mlflow.tracking.MlflowClient()
        challenger_metrics = _fetch_run_metrics(client, challenger_run_id)
        champion_metrics = (
            _fetch_run_metrics(client, champion_run_id) if champion_run_id else None
        )
        return self.evaluate(challenger_metrics, champion_metrics)
    def promote_mlflow_model(
        self,
        run_id: str,
        model_name: str,
        gate_result: GateResult,
    ) -> None:
        """
        Register and stage the winning model in MLflow.
        Transitions the new version to 'Production' and any previous
        Production version to 'Archived'.
        """
        import mlflow
        from mlflow.tracking import MlflowClient
        if not gate_result.promoted:
            print("  [gate] Model NOT promoted. Skipping registration.")
            return
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        print(f"  [gate] Registered model '{model_name}' version {mv.version}")
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for pv in prod_versions:
            client.transition_model_version_stage(
                name=model_name, version=pv.version, stage="Archived"
            )
            print(f"  [gate] Archived previous Production version {pv.version}")
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Production"
        )
        print(f"  [gate] Promoted version {mv.version} -> Production [OK]")
def _fetch_run_metrics(client: Any, run_id: str) -> dict[str, float]:
    """Fetch all metrics from an MLflow run."""
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items()}
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an MLflow run and optionally promote it."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID of the challenger model.",
    )
    parser.add_argument(
        "--champion-run-id",
        default=None,
        help="MLflow run ID of the current champion (enables relative gate).",
    )
    parser.add_argument(
        "--model-name",
        default="CampaignConversion",
        help="Registered model name in MLflow (default: CampaignConversion).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="If gate passes, register and stage the model in MLflow.",
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Use relaxed thresholds for CI smoke tests on 500-row sample.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help='JSON string of custom absolute thresholds, e.g. \'{"roc_auc": 0.70}\'',
    )
    args = parser.parse_args()
    custom_thresholds = None
    if args.thresholds:
        try:
            custom_thresholds = json.loads(args.thresholds)
        except json.JSONDecodeError as exc:
            print(f"  [gate] ERROR: --thresholds is not valid JSON: {exc}")
            sys.exit(1)
    gate = PromotionGate(
        absolute_thresholds=custom_thresholds,
        sample_mode=args.sample_mode,
    )
    result = gate.evaluate_mlflow_runs(
        challenger_run_id=args.run_id,
        champion_run_id=args.champion_run_id,
    )
    print(result)
    if args.promote and result.promoted:
        gate.promote_mlflow_model(
            run_id=args.run_id,
            model_name=args.model_name,
            gate_result=result,
        )
    sys.exit(0 if result.promoted else 1)
if __name__ == "__main__":
    main()
