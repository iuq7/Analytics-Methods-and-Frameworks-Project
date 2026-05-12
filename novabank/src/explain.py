"""Explainability + fairness audit."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from .data import PROC_DIR, load_splits
from .models import load

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def feature_importance(model, top_n: int = 15) -> pd.DataFrame:
    """Pull importances from RF inside pipeline (or calibrated wrapper)."""
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV
        rf_pipe = model.calibrated_classifiers_[0].estimator
    else:
        rf_pipe = model
    pre = rf_pipe.named_steps["pre"]
    rf = rf_pipe.named_steps["clf"]
    if not hasattr(rf, "feature_importances_"):
        return pd.DataFrame()
    names = pre.get_feature_names_out()
    imp = pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
    return imp.sort_values("importance", ascending=False).head(top_n)


def surrogate_rules(X_feat: pd.DataFrame, p_engaged: np.ndarray) -> str:
    """Shallow decision tree trained on model's risk score for plain rules."""
    risk = 1.0 - p_engaged
    y_proxy = (risk > np.quantile(risk, 0.80)).astype(int)
    numeric_cols = X_feat.select_dtypes(include=[np.number]).columns.tolist()
    Xn = X_feat[numeric_cols].copy()
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=42)
    tree.fit(Xn, y_proxy)
    return export_text(tree, feature_names=numeric_cols, max_depth=4)


def fairness_audit(
    X_feat: pd.DataFrame, y_true: pd.Series, p_engaged: np.ndarray, attr: str
) -> pd.DataFrame:
    """Per-group selection rate at top-20% and precision/recall."""
    risk = 1.0 - p_engaged
    n = len(risk)
    k = int(n * 0.20)
    order = np.argsort(-risk)
    flag = np.zeros(n, dtype=bool)
    flag[order[:k]] = True
    disengaged = (y_true.values == 0).astype(int)

    df = X_feat[[attr]].copy()
    df["flagged"] = flag
    df["disengaged"] = disengaged
    rows = []
    for g, sub in df.groupby(attr):
        n_g = len(sub)
        flag_g = sub["flagged"].sum()
        tp = int(((sub["flagged"]) & (sub["disengaged"] == 1)).sum())
        fp = int(((sub["flagged"]) & (sub["disengaged"] == 0)).sum())
        fn = int(((~sub["flagged"]) & (sub["disengaged"] == 1)).sum())
        rows.append(
            {
                "group": g,
                "n": n_g,
                "selection_rate": flag_g / n_g,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "base_rate": sub["disengaged"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def age_band(s: pd.Series) -> pd.Series:
    return pd.cut(
        s, bins=[-1, 30, 45, 60, 120], labels=["<=30", "31-45", "46-60", "61+"]
    )


def main() -> None:
    _, X_test, _, y_test = load_splits()
    X_test_f = pd.read_parquet(PROC_DIR / "X_test_feat.parquet")
    model = load("rf_calibrated")
    p = model.predict_proba(X_test_f)[:, 1]

    # Feature importance
    imp = feature_importance(model, top_n=20)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    print("Top 10 features:")
    print(imp.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(15).sort_values("importance").plot.barh(
        x="feature", y="importance", ax=ax, legend=False
    )
    ax.set_title("Top 15 features by RF importance")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=120)
    plt.close(fig)

    # Surrogate rules
    rules = surrogate_rules(X_test_f, p)
    (REPORTS / "decision_rules.txt").write_text(rules)
    print("\nSurrogate decision rules saved.")

    # Fairness: by age band, marital, job
    audit_df = X_test_f.copy()
    audit_df["age_band"] = age_band(audit_df["age"])
    for attr in ["age_band", "marital", "job"]:
        f = fairness_audit(audit_df, y_test, p, attr)
        f.to_csv(REPORTS / f"fairness_{attr}.csv", index=False)
        print(f"\nFairness ({attr}):")
        print(f.to_string(index=False))

    # PR curve + calibration plot for memo
    from sklearn.metrics import precision_recall_curve, brier_score_loss
    from sklearn.calibration import calibration_curve

    prec, rec, _ = precision_recall_curve(y_test, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall (engaged)")
    ax.set_ylabel("Precision (engaged)")
    ax.set_title("Precision-Recall — Calibrated RF")
    fig.tight_layout()
    fig.savefig(FIG / "pr_curve.png", dpi=120)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.set_xlabel("Mean predicted p(engaged)")
    ax.set_ylabel("Observed engaged fraction")
    ax.set_title("Calibration — Calibrated RF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
