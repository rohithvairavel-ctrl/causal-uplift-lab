"""Budget-constrained treatment policy simulation on RCT holdout."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def incremental_conversions(
    y: np.ndarray,
    t: np.ndarray,
    treat_mask: np.ndarray,
) -> dict:
    """
    RCT estimate of incremental conversions if we apply policy `treat_mask`.

    Among customers selected by the policy, estimate uplift:
      (mean Y | T=1, selected) - (mean Y | T=0, selected)
    then scale by number selected (as if we could treat all selected).
    """
    y = np.asarray(y).astype(float)
    t = np.asarray(t).astype(int)
    treat_mask = np.asarray(treat_mask).astype(bool)
    selected = treat_mask
    n_sel = int(selected.sum())
    if n_sel == 0:
        return {
            "n_selected": 0,
            "uplift_in_selected": 0.0,
            "incremental_conversions": 0.0,
            "rate_treated": None,
            "rate_control": None,
        }
    yt = y[selected & (t == 1)]
    yc = y[selected & (t == 0)]
    rate_t = float(yt.mean()) if len(yt) else np.nan
    rate_c = float(yc.mean()) if len(yc) else np.nan
    uplift = rate_t - rate_c if (len(yt) and len(yc)) else 0.0
    return {
        "n_selected": n_sel,
        "n_treated_obs": int(len(yt)),
        "n_control_obs": int(len(yc)),
        "rate_treated": rate_t,
        "rate_control": rate_c,
        "uplift_in_selected": float(uplift),
        "incremental_conversions": float(uplift * n_sel),
    }


def treat_top_k(score: np.ndarray, k_frac: float) -> np.ndarray:
    """Select top k_frac by score."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-score)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def treat_everyone(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def treat_none(n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool)


def roi_style(
    incremental_conversions: float,
    n_treated: int,
    value_per_conversion: float = 50.0,
    cost_per_treatment: float = 5.0,
) -> dict:
    revenue = incremental_conversions * value_per_conversion
    cost = n_treated * cost_per_treatment
    return {
        "value_per_conversion": value_per_conversion,
        "cost_per_treatment": cost_per_treatment,
        "expected_incremental_revenue": float(revenue),
        "treatment_cost": float(cost),
        "net_value": float(revenue - cost),
        "roi": float((revenue - cost) / cost) if cost > 0 else None,
    }


def compare_policies(
    y: np.ndarray,
    t: np.ndarray,
    uplift_score: np.ndarray,
    risk_score: np.ndarray,
    k_frac: float = 0.3,
    value_per_conversion: float = 50.0,
    cost_per_treatment: float = 5.0,
) -> pd.DataFrame:
    """
    Compare:
      - treat top-k% by uplift (CATE)
      - treat top-k% by outcome risk (high predicted visit / churn-risk analogue)
      - treat everyone
      - treat nobody
    """
    n = len(y)
    policies = {
        "uplift_top_k": treat_top_k(uplift_score, k_frac),
        "risk_top_k": treat_top_k(risk_score, k_frac),
        "treat_everyone": treat_everyone(n),
        "treat_none": treat_none(n),
    }
    rows = []
    for name, mask in policies.items():
        inc = incremental_conversions(y, t, mask)
        n_treat = int(mask.sum())
        econ = roi_style(
            inc["incremental_conversions"],
            n_treat,
            value_per_conversion=value_per_conversion,
            cost_per_treatment=cost_per_treatment,
        )
        rows.append({"policy": name, "k_frac": k_frac if "top_k" in name else (1.0 if name == "treat_everyone" else 0.0), **inc, **econ})
    return pd.DataFrame(rows)
