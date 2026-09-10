#!/usr/bin/env python3
"""Train S/T/X metalearners, evaluate Qini/AUUC, simulate policies, save artifacts."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_uplift.data.load import load_dataset, train_test_split_xy  # noqa: E402
from causal_uplift.learners import SLearner, TLearner, XLearner  # noqa: E402
from causal_uplift.metrics import evaluate_scores  # noqa: E402
from causal_uplift.policy import compare_policies  # noqa: E402
from causal_uplift.plots import save_decile_svg, save_policy_svg, save_qini_svg  # noqa: E402


def make_base_model(name: str, seed: int = 42):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(max_iter=1000, solver="lbfgs")
    if name == "gbm":
        return GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08, random_state=seed
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=4, max_iter=60, learning_rate=0.1, random_state=seed
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
            n_jobs=2,
        )
    raise ValueError(f"Unknown model {name}")


def save_model_b64(obj, path: Path):
    """Save joblib bytes as base64 text for GitHub text-file push."""
    import io
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    raw = buf.getvalue()
    path.write_text(base64.b64encode(raw).decode("ascii"))
    if str(path).endswith(".joblib.b64"):
        bin_path = Path(str(path)[: -len(".b64")])
        bin_path.write_bytes(raw)
    return path


def load_model_b64(path: Path):
    import io
    raw = base64.b64decode(Path(path).read_text())
    return joblib.load(io.BytesIO(raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hillstrom", "synthetic"], default="hillstrom")
    ap.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    ap.add_argument("--model", default="hgb", help="logreg|gbm|hgb|xgboost|lightgbm")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-frac", type=float, default=0.3)
    ap.add_argument("--value", type=float, default=40.0, help="value per incremental conversion")
    ap.add_argument("--cost", type=float, default=1.0, help="cost per treatment")
    ap.add_argument("--synthetic-n", type=int, default=20000)
    args = ap.parse_args()

    reports = ROOT / "reports"
    figures = reports / "figures"
    models = ROOT / "models"
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    print(f"Loading source={args.source} outcome={args.outcome} ...")
    try:
        df, meta = load_dataset(
            source=args.source,
            outcome=args.outcome,
            synthetic_n=args.synthetic_n,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(e)
        print("Falling back to synthetic RCT.")
        df, meta = load_dataset(
            source="synthetic",
            outcome=args.outcome,
            synthetic_n=args.synthetic_n,
            seed=args.seed,
        )
        args.source = "synthetic"

    split = train_test_split_xy(df, test_size=args.test_size, seed=args.seed)
    X_tr, X_te = split["X_train"], split["X_test"]
    y_tr, y_te = split["y_train"], split["y_test"]
    t_tr, t_te = split["t_train"], split["t_test"]

    print(
        f"n_train={len(X_tr)} n_test={len(X_te)} "
        f"treat_rate={t_tr.mean():.3f} outcome_rate={y_tr.mean():.3f}"
    )

    learners = {
        "S-learner": SLearner(make_base_model(args.model, seed=args.seed)),
        "T-learner": TLearner(make_base_model(args.model, seed=args.seed + 1)),
        "X-learner": XLearner(make_base_model(args.model, seed=args.seed + 2)),
    }

    cate = {}
    for name, learner in learners.items():
        print(f"Fitting {name} ...")
        learner.fit(X_tr, y_tr, t_tr)
        cate[name] = learner.predict_cate(X_te)
        if not __import__("os").environ.get("SKIP_MODEL_SAVE"):
            save_model_b64(learner, models / f"{name.lower().replace('-', '_')}.joblib.b64")

    risk_model = make_base_model(args.model, seed=args.seed + 9)
    risk_model.fit(X_tr, y_tr)
    if hasattr(risk_model, "predict_proba"):
        risk_score = risk_model.predict_proba(X_te)[:, 1]
    else:
        risk_score = risk_model.predict(X_te)
    if not __import__("os").environ.get("SKIP_MODEL_SAVE"):
        save_model_b64(risk_model, models / "outcome_risk.joblib.b64")

    scores = {**cate, "outcome_risk": risk_score}
    rng = np.random.default_rng(args.seed)
    scores["random"] = rng.random(len(y_te))

    evals = evaluate_scores(y_te, t_te, scores)

    uplift_names = ["S-learner", "T-learner", "X-learner"]
    best_name = max(uplift_names, key=lambda n: evals[n]["qini_coefficient"])
    best_cate = cate[best_name]
    print(f"Best uplift model by Qini coeff: {best_name}")

    policy_df = compare_policies(
        y_te,
        t_te,
        uplift_score=best_cate,
        risk_score=risk_score,
        k_frac=args.k_frac,
        value_per_conversion=args.value,
        cost_per_treatment=args.cost,
    )

    curves = {n: evals[n]["curve"] for n in ["T-learner", "X-learner", "S-learner", "outcome_risk"]}
    save_qini_svg(curves, figures / "qini_curves.svg")
    save_decile_svg(evals[best_name]["deciles"], figures / "uplift_by_decile.svg", title=f"Uplift by decile — {best_name}")
    save_policy_svg(policy_df, figures / "policy_net_value.svg")

    metrics = {
        "meta": {
            **meta,
            "model": args.model,
            "test_size": args.test_size,
            "seed": args.seed,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "best_uplift_model": best_name,
            "k_frac": args.k_frac,
            "value_per_conversion": args.value,
            "cost_per_treatment": args.cost,
            "feature_cols": split["feature_cols"],
        },
        "ate_test": float(y_te[t_te == 1].mean() - y_te[t_te == 0].mean()),
        "learners": {
            name: {
                "auuc": round(evals[name]["auuc"], 6),
                "qini_coefficient": round(evals[name]["qini_coefficient"], 6),
                "top_decile_uplift": evals[name]["top_decile_uplift"],
                "bottom_decile_uplift": evals[name]["bottom_decile_uplift"],
                "mean_cate": float(np.mean(cate[name])) if name in cate else None,
            }
            for name in scores
        },
        "policy": policy_df.to_dict(orient="records"),
    }

    if "true_cate_test" in split:
        pehe = {}
        true = split["true_cate_test"]
        for name in uplift_names:
            pehe[name] = float(np.sqrt(np.mean((cate[name] - true) ** 2)))
        metrics["pehe"] = pehe
        print("PEHE:", pehe)

    metrics_path = reports / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics["learners"][k] for k in metrics["learners"]}, indent=2))
    print("Policy:")
    print(policy_df.to_string(index=False))
    print(f"Wrote {metrics_path}")
    print(f"Figures in {figures}")
    print(f"Models in {models}")


if __name__ == "__main__":
    main()
