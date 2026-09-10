"""SVG figure writers (matplotlib) for Qini, deciles, policy comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_qini_svg(
    curves: Dict[str, pd.DataFrame],
    path: Path,
    title: str = "Qini curves (holdout RCT)",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in curves.items():
        ax.plot(curve["fraction"], curve["qini"], label=name, linewidth=2)
    first = next(iter(curves.values()))
    ax.plot(first["fraction"], first["random"], "--", color="gray", label="random")
    ax.set_xlabel("Fraction targeted (highest score first)")
    ax.set_ylabel("Cumulative incremental conversions (Qini)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_decile_svg(deciles: pd.DataFrame, path: Path, title: str = "Uplift by score decile"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = deciles.sort_values("decile")
    colors = ["#2ca02c" if (u or 0) > 0 else "#d62728" for u in d["uplift"]]
    ax.bar(d["decile"].astype(str), d["uplift"].fillna(0), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decile (10 = highest predicted uplift)")
    ax.set_ylabel("Observed uplift (visit rate T - C)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_policy_svg(policy_df: pd.DataFrame, path: Path, title: str = "Policy net value"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df = policy_df.copy()
    ax.bar(df["policy"], df["net_value"], color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net value ($)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
