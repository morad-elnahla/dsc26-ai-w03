"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "failed"

# Known when the parcel is created. Left out: parcel_id (identifier), courier_notes and
# delivered_at (written after the attempt), failed (the label).
FEATURE_COLUMNS: list[str] = [
    "customer_id", "merchant_id", "area", "created_at", "promised_slot",
    "cod_amount", "n_items", "address_text",
]
HISTORY_COLUMNS = ["customer_prior_parcels", "customer_prior_fail_rate"]
CAT_COLS = ["area", "merchant_id", "promised_slot"]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time customer history: only parcels strictly earlier than each row."""
    out = df.copy()
    key = pd.DataFrame({
        "c": df["customer_id"].to_numpy(),
        "t": df["created_at"].to_numpy(),
        "f": df["failed"].to_numpy(dtype=float),
    })
    # one line per (customer, timestamp) so equal timestamps never see each other
    g = key.groupby(["c", "t"], sort=True).agg(n=("f", "size"), fails=("f", "sum")).reset_index()
    g["prior_n"] = g.groupby("c")["n"].cumsum() - g["n"]
    g["prior_f"] = g.groupby("c")["fails"].cumsum() - g["fails"]
    merged = key[["c", "t"]].merge(g[["c", "t", "prior_n", "prior_f"]], on=["c", "t"], how="left")
    prior_n = merged["prior_n"].to_numpy()
    prior_f = merged["prior_f"].to_numpy()
    rate = np.full(len(df), np.nan)
    np.divide(prior_f, prior_n, out=rate, where=prior_n > 0)
    out["customer_prior_parcels"] = prior_n.astype(int)
    out["customer_prior_fail_rate"] = rate
    return out


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be strictly between 0 and 1")
    n_test = math.ceil(len(df) * test_fraction)
    ordered = df.sort_values(time_col, kind="stable")
    boundary = ordered[time_col].iloc[len(df) - n_test]
    return df[df[time_col] < boundary], df[df[time_col] >= boundary]


def _cats(X):
    X = pd.DataFrame(X).copy()
    for c in X.columns:
        X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
    return X


def _address(X):
    s = pd.DataFrame(X).iloc[:, 0]
    return s.astype(object).where(s.notna(), "").astype(str).str.lower().to_numpy()


def _numeric(X):
    X = pd.DataFrame(X)
    out = pd.DataFrame(index=X.index)
    out["log_cod"] = np.log1p(pd.to_numeric(X["cod_amount"], errors="coerce").astype(float))
    out["n_items"] = pd.to_numeric(X["n_items"], errors="coerce").astype(float)
    out["prior_parcels"] = np.log1p(
        pd.to_numeric(X["customer_prior_parcels"], errors="coerce").astype(float))
    out["prior_fail_rate"] = pd.to_numeric(
        X["customer_prior_fail_rate"], errors="coerce").astype(float)
    addr = X["address_text"]
    out["address_empty"] = (addr.isna() | (addr.astype(str).str.strip() == "")).astype(float)
    return out.to_numpy(dtype=float)


def build_pipeline(seed: int = 0) -> Pipeline:
    num_cols = ["cod_amount", "n_items", "address_text"] + HISTORY_COLUMNS
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("make", FunctionTransformer(_numeric)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("clean", FunctionTransformer(_cats)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_COLS),
        ("addr", Pipeline([
            ("clean", FunctionTransformer(_address)),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5)),
        ]), ["address_text"]),
    ], remainder="drop")
    clf = LogisticRegression(C=0.3, max_iter=2000, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    y = test_df[TARGET]
    p = model.predict_proba(test_df)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
        "n": int(len(test_df)),
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    model = build_pipeline(seed).fit(train_df, train_df[TARGET])
    dummy = DummyClassifier(strategy="prior").fit(train_df, train_df[TARGET])
    m, d = evaluate(model, test_df), evaluate(dummy, test_df)
    return {"model": m, "dummy": d, "lift_pr_auc": m["pr_auc"] - d["pr_auc"]}