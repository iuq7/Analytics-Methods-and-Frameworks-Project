"""Load UCI Bank Marketing dataset, profile, split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = RAW_DIR / "bank_marketing.parquet"
SEED = 42


def fetch() -> pd.DataFrame:
    """Fetch from UCI repo, cache as parquet."""
    if RAW_PATH.exists():
        return pd.read_parquet(RAW_PATH)
    from ucimlrepo import fetch_ucirepo

    bm = fetch_ucirepo(id=222)
    X = bm.data.features.copy()
    y = bm.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip().lower() for c in df.columns]
    df.to_parquet(RAW_PATH, index=False)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Scenario 3: model engagement (rare class) directly.

    `engaged = 1` when customer accepted offer (y == 'yes'). Downstream we
    define risk_score = 1 - p_engaged so predictions still drive retention
    targeting (low engagement probability = high disengagement risk = proxy
    for churn risk). Keeps PR-AUC meaningful (positive baseline ~11%).
    """
    target_col = "y" if "y" in df.columns else df.columns[-1]
    df = df.copy()
    df["engaged"] = (df[target_col].astype(str).str.lower() == "yes").astype(int)
    return df.drop(columns=[target_col])


def split(df: pd.DataFrame, test_size: float = 0.3):
    """Stratified train/test split. Drops `duration` (post-call leakage)."""
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    y = df["engaged"]
    X = df.drop(columns=["engaged"])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile: dtype, missing, n_unique, sample."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "unknown_pct": float(
                    (s.astype(str).str.lower() == "unknown").mean()
                    if s.dtype == object
                    else 0.0
                ),
                "n_unique": int(s.nunique(dropna=True)),
                "example": s.dropna().iloc[0] if s.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def save_splits(X_train, X_test, y_train, y_test) -> None:
    X_train.to_parquet(PROC_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(PROC_DIR / "X_test.parquet", index=False)
    y_train.to_frame("engaged").to_parquet(PROC_DIR / "y_train.parquet", index=False)
    y_test.to_frame("engaged").to_parquet(PROC_DIR / "y_test.parquet", index=False)


def load_splits():
    return (
        pd.read_parquet(PROC_DIR / "X_train.parquet"),
        pd.read_parquet(PROC_DIR / "X_test.parquet"),
        pd.read_parquet(PROC_DIR / "y_train.parquet")["engaged"],
        pd.read_parquet(PROC_DIR / "y_test.parquet")["engaged"],
    )


def main() -> None:
    df = fetch()
    print(f"raw shape: {df.shape}")
    df = build_target(df)
    prof = profile(df)
    prof_path = Path(__file__).resolve().parents[1] / "reports" / "profile.csv"
    prof.to_csv(prof_path, index=False)
    print(f"profile -> {prof_path}")
    print(f"target rate (engaged=1): {df['engaged'].mean():.4f}")
    X_train, X_test, y_train, y_test = split(df)
    save_splits(X_train, X_test, y_train, y_test)
    print(f"train: {X_train.shape}  test: {X_test.shape}")


if __name__ == "__main__":
    main()
