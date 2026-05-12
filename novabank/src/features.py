"""Feature engineering: RFM proxies, encoders, synthetic signals.

Synthetic features follow Scenario 3 (Engagement Analysis). They are
deterministic functions of real fields plus seeded noise so results are
reproducible. Each rule is documented; document these in the executive memo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
NUMERIC_BASE = ["age", "balance", "day_of_week", "campaign", "pdays", "previous"]


def add_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Recency/Frequency/Monetary proxies from bank marketing fields."""
    df = df.copy()
    df["recency"] = df["pdays"].replace(-1, df["pdays"].max() + 1)
    df["frequency"] = df["previous"] + df["campaign"]
    df["monetary"] = df["balance"].clip(lower=0)

    df["r_score"] = pd.qcut(-df["recency"], q=5, labels=False, duplicates="drop") + 1
    df["f_score"] = pd.qcut(
        df["frequency"].rank(method="first"), q=5, labels=False, duplicates="drop"
    ) + 1
    df["m_score"] = pd.qcut(
        df["monetary"].rank(method="first"), q=5, labels=False, duplicates="drop"
    ) + 1
    df["rfm_score"] = df["r_score"] + df["f_score"] + df["m_score"]
    return df


def add_synthetic(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Synthetic features per Scenario 3. Deterministic given seed.

    Assumptions documented inline. Adjust mapping in memo appendix.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # overdraft_count_6m: more when balance negative
    base = (df["balance"] < 0).astype(int) * 4
    df["overdraft_count_6m"] = (base + rng.poisson(0.3, n)).astype(int)

    # product_count: 1 + housing + loan + default
    df["product_count"] = (
        1
        + (df["housing"].astype(str) == "yes").astype(int)
        + (df["loan"].astype(str) == "yes").astype(int)
        + (df["default"].astype(str) == "yes").astype(int)
    )

    # digital_user: bias toward cellular contact + younger age
    p_digital = 0.2 + 0.4 * (df["contact"].astype(str) == "cellular") + 0.2 * (df["age"] < 40)
    df["digital_user"] = (rng.random(n) < p_digital.clip(0, 0.95)).astype(int)

    # direct_deposit_active: balance > median AND productive job
    productive = df["job"].astype(str).isin(
        ["admin.", "technician", "management", "blue-collar", "services"]
    )
    df["direct_deposit_active"] = (
        ((df["balance"] > df["balance"].median()) & productive).astype(int)
    )

    # unresolved_complaint: small fraction with poutcome == failure
    fail_mask = df["poutcome"].astype(str) == "failure"
    df["unresolved_complaint"] = (
        fail_mask & (rng.random(n) < 0.15)
    ).astype(int)

    # transaction_velocity: ratio in [0.1, 1.5], lower when pdays high
    pdays_adj = df["pdays"].replace(-1, 999)
    velocity = 1.0 - (pdays_adj / pdays_adj.max()).clip(0, 1) * 0.6
    df["transaction_velocity"] = (velocity + rng.normal(0, 0.05, n)).clip(0.1, 1.5)

    # fee_ratio: higher when balance low/negative
    safe_bal = df["balance"].abs().clip(lower=1)
    df["fee_ratio"] = (10 / safe_bal).clip(0, 1) + rng.normal(0, 0.02, n)
    df["fee_ratio"] = df["fee_ratio"].clip(lower=0)

    return df


SYNTHETIC = [
    "overdraft_count_6m",
    "product_count",
    "digital_user",
    "direct_deposit_active",
    "unresolved_complaint",
    "transaction_velocity",
    "fee_ratio",
]
RFM_NUM = ["recency", "frequency", "monetary", "r_score", "f_score", "m_score", "rfm_score"]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals: NaN -> 'unknown' (kept as signal). Numerics: median."""
    df = df.copy()
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("unknown")
    for c in NUMERIC_BASE:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_missing(df)
    df = add_rfm(df)
    df = add_synthetic(df)
    return df


def make_preprocessor() -> ColumnTransformer:
    """Encoder for downstream sklearn models. Trees can also use raw."""
    numeric = NUMERIC_BASE + RFM_NUM + SYNTHETIC
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )


def rfm_segment(row: pd.Series) -> str:
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if m >= 4 and r <= 2:
        return "At-Risk High Value"
    if f >= 4 and m >= 3:
        return "Loyal"
    if r >= 4 and m <= 2:
        return "New Potential"
    if r <= 2 and m <= 2:
        return "Hibernating"
    return "Mid-Value"


def assign_segments(df: pd.DataFrame) -> pd.Series:
    return df.apply(rfm_segment, axis=1)
