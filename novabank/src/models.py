"""Train baseline + improved models. Persist artifacts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_splits, PROC_DIR
from .features import build_features, make_preprocessor

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("pre", make_preprocessor()),
            ("clf", estimator),
        ]
    )


def train_logreg(X_train, y_train) -> Pipeline:
    pipe = build_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            solver="liblinear",
            random_state=42,
        )
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_rf(X_train, y_train) -> Pipeline:
    pipe = build_pipeline(
        RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_rf_calibrated(X_train, y_train) -> CalibratedClassifierCV:
    """Isotonic calibration on top of RF via 5-fold CV."""
    base = build_pipeline(
        RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    )
    calib = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calib.fit(X_train, y_train)
    return calib


def save(model, name: str) -> Path:
    path = ARTIFACT_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load(name: str):
    with open(ARTIFACT_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main() -> None:
    X_train, X_test, y_train, y_test = load_splits()
    X_train_f = build_features(X_train)
    X_test_f = build_features(X_test)

    print("Training logreg...")
    logreg = train_logreg(X_train_f, y_train)
    save(logreg, "logreg")

    print("Training random forest...")
    rf = train_rf(X_train_f, y_train)
    save(rf, "rf")

    print("Training calibrated RF...")
    rf_cal = train_rf_calibrated(X_train_f, y_train)
    save(rf_cal, "rf_calibrated")

    # persist featurized test set for evaluate.py
    X_test_f.to_parquet(PROC_DIR / "X_test_feat.parquet", index=False)
    X_train_f.to_parquet(PROC_DIR / "X_train_feat.parquet", index=False)
    print(f"artifacts -> {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
