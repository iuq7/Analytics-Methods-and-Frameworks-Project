"""Action Matrix: RFM segment x Risk tier. Pilot plan helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROC_DIR, load_splits
from .features import assign_segments, build_features
from .models import load

REPORTS = Path(__file__).resolve().parents[1] / "reports"

RISK_TIERS = ["Low", "Mid", "High"]
ACTIONS = {
    ("Champions", "Low"): "Loyalty interest rate",
    ("Champions", "Mid"): "Personal manager check-in",
    ("Champions", "High"): "Priority retention call (24h SLA)",
    ("Loyal", "Low"): "Cross-sell savings",
    ("Loyal", "Mid"): "Cross-sell + check-in",
    ("Loyal", "High"): "Retention offer + advisor call",
    ("At-Risk High Value", "Low"): "Monitor; re-score next month",
    ("At-Risk High Value", "Mid"): "Win-back offer (email + SMS)",
    ("At-Risk High Value", "High"): "TOP PRIORITY: manager call 24h",
    ("New Potential", "Low"): "Onboarding nudge",
    ("New Potential", "Mid"): "Upsell savings plan",
    ("New Potential", "High"): "Welcome offer + check-in",
    ("Mid-Value", "Low"): "Standard newsletter",
    ("Mid-Value", "Mid"): "Promo email",
    ("Mid-Value", "High"): "Retention email",
    ("Hibernating", "Low"): "Ignore",
    ("Hibernating", "Mid"): "Low-cost reactivation email",
    ("Hibernating", "High"): "Low-cost reactivation email",
}


def tier_risk(risk: np.ndarray, qs=(0.5, 0.8)) -> np.ndarray:
    """Bucket risk into Low/Mid/High by global quantiles."""
    lo, hi = np.quantile(risk, qs)
    out = np.full(len(risk), "Low", dtype=object)
    out[risk >= lo] = "Mid"
    out[risk >= hi] = "High"
    return out


def build_action_table(
    X_test: pd.DataFrame, p_engaged: np.ndarray
) -> pd.DataFrame:
    df = X_test.copy()
    df["risk_score"] = 1.0 - p_engaged
    df["risk_tier"] = tier_risk(df["risk_score"].values)
    df["segment"] = assign_segments(df)
    df["action"] = [
        ACTIONS.get((seg, tier), "Standard")
        for seg, tier in zip(df["segment"], df["risk_tier"])
    ]
    return df


def scenario_sensitivity(
    X_test: pd.DataFrame, y_test: pd.Series, p_engaged: np.ndarray
) -> pd.DataFrame:
    """Stress tests: feature drop + retention rate shift."""
    rows = []
    base_disengaged = (y_test == 0).astype(int).values
    risk = 1.0 - p_engaged
    n = len(risk)
    k = int(n * 0.20)
    order = np.argsort(-risk)
    flag = np.zeros(n, dtype=bool)
    flag[order[:k]] = True
    tp = int((flag & (base_disengaged == 1)).sum())
    fp = int((flag & (base_disengaged == 0)).sum())
    base_precision = tp / max(1, tp + fp)
    rows.append({"scenario": "baseline", "top20_precision": base_precision, "tp": tp, "fp": fp})

    # Cost shocks
    for ltv, cost, retain in [
        (500, 30, 0.40),
        (300, 30, 0.40),
        (500, 60, 0.40),
        (500, 30, 0.25),
        (500, 30, 0.55),
    ]:
        retained = retain * tp
        ev = retained * ltv - (tp + fp) * cost
        rows.append(
            {
                "scenario": f"LTV={ltv} cost={cost} retain={retain}",
                "top20_precision": base_precision,
                "expected_value": float(ev),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    _, X_test, _, y_test = load_splits()
    X_test_f = pd.read_parquet(PROC_DIR / "X_test_feat.parquet")
    model = load("rf_calibrated")
    p = model.predict_proba(X_test_f)[:, 1]

    action = build_action_table(X_test_f, p)
    action.to_parquet(REPORTS.parent / "data" / "processed" / "actions.parquet", index=False)

    # Crosstab: segment x risk tier
    ct = pd.crosstab(action["segment"], action["risk_tier"])
    ct.to_csv(REPORTS / "segment_x_risk.csv")
    print("Segment x Risk tier:")
    print(ct)

    # Action volume
    av = action["action"].value_counts().rename_axis("action").reset_index(name="n")
    av.to_csv(REPORTS / "action_volume.csv", index=False)
    print("\nAction volumes:")
    print(av.to_string(index=False))

    # Scenarios
    sens = scenario_sensitivity(X_test_f, y_test, p)
    sens.to_csv(REPORTS / "sensitivity.csv", index=False)
    print("\nSensitivity:")
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
