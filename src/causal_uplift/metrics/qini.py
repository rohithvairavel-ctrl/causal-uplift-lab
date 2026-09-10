"""Qini curve, AUUC, and uplift-by-decile metrics for RCT evaluation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _order_by_score(score: np.ndarray, descending: bool = True) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    # Higher score = treat first for uplift; for risk baseline also descending
    return np.argsort(-score if descending else score)


def qini_curve(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    n_bins: int = 100,
) -> pd.DataFrame:
    """
    Cumulative incremental outcomes when treating top-k by score (RCT estimator).

    At fraction p of population (ordered by score desc):
      Q(p) = (n_t / N_t) * sum Y among treated in top-p
           - (n_c / N_c) * sum Y among control in top-p
    We use the running difference of conversion rates scaled by group size in bin.
    Classic formulation (cumulative incremental gain):
      uplift_cum = Y_treated_cum / n_treated_total * n_in_segment_treated_cum style...

    We report:
      fraction, cum_gain, random_gain, perfect placeholder None
    where cum_gain is the cumulative incremental conversions if we only
    look at the top fraction (RCT difference-in-conversion * size).
    """
    y = np.asarray(y).astype(float)
    t = np.asarray(t).astype(int)
    score = np.asarray(score, dtype=float)
    order = _order_by_score(score)
    y, t = y[order], t[order]
    n = len(y)
    # Running sums
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    cum_nt = np.cumsum(t)
    cum_nc = np.cumsum(1 - t)

    # Incremental conversions if treated everyone in top-k (scale control rate)
    # Q(k) = cum_yt - cum_yc * (cum_nt / cum_nc)   when cum_nc>0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt - np.where(cum_nc > 0, cum_yc * (cum_nt / np.maximum(cum_nc, 1)), 0.0)

    # subsample to n_bins points
    idx = np.linspace(0, n - 1, num=min(n_bins, n), dtype=int)
    frac = (idx + 1) / n
    # Random baseline: overall ATE * fraction * n_treated_share approx
    ate = y[t == 1].mean() - y[t == 0].mean() if (t == 1).any() and (t == 0).any() else 0.0
    # Random Q(p) ≈ ATE * (# treated in top p) ≈ ATE * p * n_t_total / something
    # Standard: diagonal from 0 to Q(1)
    q_end = float(q[-1])
    random_gain = frac * q_end

    return pd.DataFrame(
        {
            "fraction": frac,
            "qini": q[idx],
            "random": random_gain,
            "cum_treated": cum_nt[idx],
            "cum_control": cum_nc[idx],
        }
    )


def auuc(curve: pd.DataFrame) -> float:
    """Area under the Qini curve (trapezoidal), higher is better."""
    return float(np.trapezoid(curve["qini"].values, curve["fraction"].values))


def qini_coefficient(curve: pd.DataFrame) -> float:
    """
    Qini coefficient ≈ AUUC_model - AUUC_random.
    Random is the diagonal to Q(1).
    """
    auuc_model = auuc(curve)
    auuc_random = float(np.trapezoid(curve["random"].values, curve["fraction"].values))
    return auuc_model - auuc_random


def uplift_by_decile(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Per-decile observed uplift (conversion_treated - conversion_control)."""
    y = np.asarray(y).astype(float)
    t = np.asarray(t).astype(int)
    score = np.asarray(score, dtype=float)
    # Higher score -> decile 10
    order = _order_by_score(score)
    y, t, score = y[order], t[order], score[order]
    n = len(y)
    edges = np.linspace(0, n, n_deciles + 1, dtype=int)
    rows = []
    for d in range(n_deciles):
        sl = slice(edges[d], edges[d + 1])
        yd, td = y[sl], t[sl]
        nt, nc = (td == 1).sum(), (td == 0).sum()
        yt = yd[td == 1].mean() if nt else np.nan
        yc = yd[td == 0].mean() if nc else np.nan
        rows.append(
            {
                "decile": n_deciles - d,  # 10 = highest score
                "n": int(edges[d + 1] - edges[d]),
                "n_treated": int(nt),
                "n_control": int(nc),
                "rate_treated": float(yt) if yt == yt else None,
                "rate_control": float(yc) if yc == yc else None,
                "uplift": float(yt - yc) if (yt == yt and yc == yc) else None,
                "mean_score": float(score[sl].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("decile", ascending=False).reset_index(drop=True)


def evaluate_scores(
    y: np.ndarray,
    t: np.ndarray,
    scores: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """Compute AUUC / Qini coeff / decile table summary for each score vector."""
    out = {}
    for name, s in scores.items():
        curve = qini_curve(y, t, s)
        dec = uplift_by_decile(y, t, s)
        out[name] = {
            "auuc": auuc(curve),
            "qini_coefficient": qini_coefficient(curve),
            "top_decile_uplift": float(dec.iloc[0]["uplift"]) if len(dec) else None,
            "bottom_decile_uplift": float(dec.iloc[-1]["uplift"]) if len(dec) else None,
            "curve": curve,
            "deciles": dec,
        }
    return out
