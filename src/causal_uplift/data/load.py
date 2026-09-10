"""Dataset loading and preprocessing for uplift modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

FEATURE_COLS = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "zip_code_Rural",
    "zip_code_Surburban",
    "zip_code_Urban",
    "channel_Multichannel",
    "channel_Phone",
    "channel_Web",
    "history_segment_0",
    "history_segment_1",
    "history_segment_2",
    "history_segment_3",
    "history_segment_4",
    "history_segment_5",
    "history_segment_6",
]


def _one_hot(df: pd.DataFrame, col: str, prefix: Optional[str] = None) -> pd.DataFrame:
    prefix = prefix or col
    dummies = pd.get_dummies(df[col], prefix=prefix, dtype=int)
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


def prepare_hillstrom(
    df: pd.DataFrame,
    treatment_arm: str = "Mens E-Mail",
    outcome: str = "visit",
) -> pd.DataFrame:
    """
    Restrict to one treatment arm vs control and build model matrix.

    Classic uplift framing: Mens E-Mail vs No E-Mail, outcome = visit (or conversion).
    """
    keep = df["segment"].isin([treatment_arm, "No E-Mail"]).copy()
    out = df.loc[keep].copy()
    out["treatment"] = (out["segment"] == treatment_arm).astype(int)
    out["outcome"] = out[outcome].astype(int)

    out = _one_hot(out, "zip_code", "zip_code")
    out = _one_hot(out, "channel", "channel")
    hs = out["history_segment"].astype(str)
    out["history_segment_idx"] = hs.str.extract(r"^(\d+)").astype(float).fillna(0).astype(int)
    for i in range(7):
        out[f"history_segment_{i}"] = (out["history_segment_idx"] == i).astype(int)
    out = out.drop(columns=["history_segment", "history_segment_idx"])

    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0

    cols = FEATURE_COLS + ["treatment", "outcome"]
    if "true_cate" in out.columns:
        cols = cols + ["true_cate"]
    return out[cols].reset_index(drop=True)


def load_hillstrom_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def generate_synthetic_rct(
    n: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic RCT with known ground-truth CATE for sanity checks."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 13, size=n)
    history = rng.lognormal(mean=4.5, sigma=0.8, size=n).clip(0, 1000)
    mens = rng.integers(0, 2, size=n)
    womens = rng.integers(0, 2, size=n)
    newbie = rng.integers(0, 2, size=n)
    zip_code = rng.choice(["Rural", "Surburban", "Urban"], size=n, p=[0.2, 0.4, 0.4])
    channel = rng.choice(["Phone", "Web", "Multichannel"], size=n, p=[0.3, 0.5, 0.2])
    history_segment = np.digitize(history, bins=[0, 50, 100, 200, 350, 500, 750])

    base_logit = (
        -1.2
        - 0.08 * recency
        + 0.002 * history
        + 0.3 * mens
        + 0.15 * womens
        - 0.4 * newbie
    )
    true_cate = (
        0.05
        + 0.12 * newbie
        + 0.08 * mens
        + 0.00015 * history
        - 0.01 * recency
        + 0.04 * (zip_code == "Urban").astype(float)
    )
    true_cate = np.clip(true_cate, -0.05, 0.35)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    p0 = sigmoid(base_logit)
    p1 = np.clip(p0 + true_cate, 0.01, 0.99)

    treatment = rng.integers(0, 2, size=n)
    y0 = rng.binomial(1, p0)
    y1 = rng.binomial(1, p1)
    outcome = np.where(treatment == 1, y1, y0)

    df = pd.DataFrame(
        {
            "recency": recency,
            "history_segment": [f"{i}) bin" for i in history_segment],
            "history": history,
            "mens": mens,
            "womens": womens,
            "zip_code": zip_code,
            "newbie": newbie,
            "channel": channel,
            "segment": np.where(treatment == 1, "Mens E-Mail", "No E-Mail"),
            "visit": outcome,
            "conversion": rng.binomial(1, np.clip(outcome * 0.15, 0, 1)),
            "spend": np.where(outcome == 1, rng.lognormal(3.0, 0.5, n), 0.0),
            "true_cate": true_cate,
        }
    )
    return df


def load_dataset(
    source: str = "hillstrom",
    data_dir: Optional[Path] = None,
    treatment_arm: str = "Mens E-Mail",
    outcome: str = "visit",
    synthetic_n: int = 20000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """Load prepared uplift table."""
    data_dir = Path(data_dir) if data_dir else RAW_DIR
    meta = {"source": source, "treatment_arm": treatment_arm, "outcome": outcome}

    if source == "synthetic":
        raw = generate_synthetic_rct(n=synthetic_n, seed=seed)
        prepared = prepare_hillstrom(raw, treatment_arm=treatment_arm, outcome=outcome)
        meta["rct"] = True
        meta["ground_truth_cate"] = True
        meta["n"] = len(prepared)
        return prepared, meta

    candidates = [
        data_dir / "hillstrom.csv",
        SAMPLE_DIR / "hillstrom_sample.csv",
        PROJECT_ROOT / "data" / "sample" / "hillstrom_sample.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Hillstrom CSV not found. Run scripts/download_data.py first, "
            "or use --source synthetic."
        )
    raw = load_hillstrom_csv(path)
    prepared = prepare_hillstrom(raw, treatment_arm=treatment_arm, outcome=outcome)
    meta["path"] = str(path)
    meta["rct"] = True
    meta["ground_truth_cate"] = False
    meta["n"] = len(prepared)
    return prepared, meta


def train_test_split_xy(
    df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
):
    """Stratified split on treatment for stable propensities in both folds."""
    from sklearn.model_selection import train_test_split

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df["outcome"].values
    t = df["treatment"].values
    extras = {}
    if "true_cate" in df.columns:
        extras["true_cate"] = df["true_cate"].values

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=t
    )
    result = {
        "X_train": X.iloc[train_idx].reset_index(drop=True),
        "X_test": X.iloc[test_idx].reset_index(drop=True),
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "t_train": t[train_idx],
        "t_test": t[test_idx],
        "feature_cols": feature_cols,
    }
    if extras:
        result["true_cate_train"] = extras["true_cate"][train_idx]
        result["true_cate_test"] = extras["true_cate"][test_idx]
    return result
