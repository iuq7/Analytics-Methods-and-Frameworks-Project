"""Metrics, threshold sweep, decision rule selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from .data import load_splits, PROC_DIR
from .models import load

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def risk_score(p_engaged: np.ndarray) -> np.ndarray:
    """Higher = more disengagement risk."""
    return 1.0 - p_engaged


def core_metrics(y_true, p_engaged) -> dict:
    """Engaged is positive class. PR-AUC + ROC-AUC + Brier."""
    return {
        "roc_auc": float(roc_auc_score(y_true, p_engaged)),
        "pr_auc": float(average_precision_score(y_true, p_engaged)),
        "brier": float(brier_score_loss(y_true, p_engaged)),
    }


def risk_topk(y_true, p_engaged, k_pct: float = 0.20) -> dict:
    """Treat top-k% by risk score as 'flagged for retention'. Measure how
    many of those flagged are actually disengaged (y_true == 0).
    """
    risk = risk_score(p_engaged)
    n = len(risk)
    k = max(1, int(n * k_pct))
    order = np.argsort(-risk)
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True
    disengaged = (y_true == 0).astype(int).values if hasattr(y_true, "values") else (y_true == 0).astype(int)
    tp = int(((flagged) & (disengaged == 1)).sum())
    fp = int(((flagged) & (disengaged == 0)).sum())
    fn = int(((~flagged) & (disengaged == 1)).sum())
    return {
        "k_pct": k_pct,
        "n_flagged": k,
        "precision_disengaged_at_k": tp / max(1, (tp + fp)),
        "recall_disengaged_at_k": tp / max(1, (tp + fn)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def expected_value(
    y_true,
    p_engaged,
    ltv: float = 500.0,
    offer_cost: float = 30.0,
    retain_rate: float = 0.4,
) -> pd.DataFrame:
    """Sweep threshold on disengagement risk; report EV per cohort.

    Assumes flagged customers receive offer. Retention rate applies only to
    truly disengaged (would have left). False positives waste offer_cost.
    """
    risk = risk_score(p_engaged)
    disengaged = (y_true == 0).astype(int).values if hasattr(y_true, "values") else (y_true == 0).astype(int)
    rows = []
    for thr in np.linspace(0.5, 0.99, 50):
        flag = risk >= thr
        tp = int((flag & (disengaged == 1)).sum())
        fp = int((flag & (disengaged == 0)).sum())
        retained = retain_rate * tp
        ev = retained * ltv - (tp + fp) * offer_cost
        rows.append(
            {
                "threshold": float(thr),
                "n_flagged": int(flag.sum()),
                "tp": tp,
                "fp": fp,
                "expected_value": float(ev),
                "precision": tp / max(1, tp + fp),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    _, X_test, _, y_test = load_splits()
    X_test_f = pd.read_parquet(PROC_DIR / "X_test_feat.parquet")

    results = []
    for name in ["logreg", "rf", "rf_calibrated"]:
        model = load(name)
        p = model.predict_proba(X_test_f)[:, 1]
        m = core_metrics(y_test, p)
        m["model"] = name
        m.update(risk_topk(y_test, p, 0.20))
        results.append(m)

    df = pd.DataFrame(results)[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "brier",
            "precision_disengaged_at_k",
            "recall_disengaged_at_k",
            "k_pct",
            "n_flagged",
        ]
    ]
    out = REPORTS / "results.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nresults -> {out}")

    # EV sweep on calibrated RF (best for thresholding)
    best_model = load("rf_calibrated")
    p = best_model.predict_proba(X_test_f)[:, 1]
    ev = expected_value(y_test, p)
    ev_path = REPORTS / "ev_sweep.csv"
    ev.to_csv(ev_path, index=False)
    best = ev.loc[ev["expected_value"].idxmax()]
    print(f"\nbest EV threshold: {best['threshold']:.3f}  "
          f"EV: {best['expected_value']:.0f}  "
          f"flagged: {best['n_flagged']}  "
          f"precision: {best['precision']:.3f}")
    with open(REPORTS / "best_threshold.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)

    # Budget-cap analysis: rank by risk, vary cap from 5% to 50%
    risk = risk_score(p)
    disengaged = (y_test == 0).astype(int).values
    cap_rows = []
    for k_pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        n = len(risk)
        k = int(n * k_pct)
        order = np.argsort(-risk)
        flag = np.zeros(n, dtype=bool)
        flag[order[:k]] = True
        tp = int((flag & (disengaged == 1)).sum())
        fp = int((flag & (disengaged == 0)).sum())
        cap_rows.append(
            {
                "budget_pct": k_pct,
                "n_flagged": int(flag.sum()),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, disengaged.sum()),
                "lift": (tp / max(1, tp + fp)) / max(1e-9, disengaged.mean()),
            }
        )
    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_csv(REPORTS / "budget_cap.csv", index=False)
    print("\nBudget cap analysis:")
    print(cap_df.to_string(index=False))


if __name__ == "__main__":
    main()
