# Causal Uplift Lab

> **Prediction** answers *who will churn / visit?*  
> **Uplift (CATE)** answers *who should we treat?* — retention offer, coupon, or email.

Resume-ready causal ML portfolio project: metalearners (S / T / X), Qini / AUUC evaluation, budget-constrained policy simulation, and a Streamlit demo. Built as the **innovative sequel to vanilla churn prediction**.

**Repo:** https://github.com/rohithvairavel-ctrl/causal-uplift-lab

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate or conda activate uplift
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py --source hillstrom --model gbm
streamlit run app/streamlit_app.py
```

Optional: `python scripts/train.py --source synthetic --model gbm` (known CATE + PEHE).

> Committed `models/*.joblib.b64` are logistic-regression metalearners (small/app-ready). Metrics/figures below are from `--model gbm`.

## Why this exists

Prediction ranks **outcome risk** P(Y=1|X). Uplift ranks **incremental effect of treatment** CATE = E[Y(1)-Y(0)|X]. Target persuadables; avoid sure things, lost causes, and sleeping dogs.

## Dataset

1. **Hillstrom MineThatData** — public email marketing RCT (~64k). Default: Men's email vs control, outcome=`visit`. Download via `scripts/download_data.py`; sample at `data/sample/hillstrom_sample.csv`.
2. **Synthetic RCT** — known heterogeneous CATE for PEHE and clear uplift-vs-risk demos.

## Methods

S / T / X metalearners in `src/causal_uplift/learners/` (sklearn; optional xgboost/lightgbm). Baseline: outcome-risk model (treat high-risk).

## Evaluation (real run, seed=42, gbm, $40/visit, $1/email)

### Hillstrom (ATE ≈ +7.35 pp visit)

| Model | AUUC | Qini coeff | Top-decile uplift |
|---|---:|---:|---:|
| S-learner | 246.55 | 11.55 | 0.115 |
| **T-learner** | **247.76** | **12.76** | 0.105 |
| X-learner | 241.37 | 6.37 | 0.117 |
| Outcome-risk | 255.28 | 20.28 | 0.100 |

Policy @ top 30%: uplift ROI **2.41**, risk ROI **2.45**, treat-everyone ROI **1.94**. Effects are fairly homogeneous — honest finding.

### Synthetic (heterogeneous CATE)

| Model | AUUC | Qini | PEHE |
|---|---:|---:|---:|
| **S-learner** | **259.9** | **76.2** | **0.044** |
| T-learner | 250.6 | 66.9 | 0.060 |
| X-learner | 249.3 | 65.6 | 0.057 |
| Outcome-risk | 240.5 | 56.8 | — |

Policy @ top 30%: uplift ROI **7.45** vs risk **6.92** vs treat-everyone **3.91**.

See `reports/metrics.json`, `reports/metrics_synthetic.json`, and `reports/figures/*.svg`.

## Layout

`src/causal_uplift/{data,learners,metrics,policy,plots}`, `scripts/`, `app/streamlit_app.py`, `notebooks/`, `models/`, `reports/`.

## Interview talking points

1. Potential outcomes / RCT identification  
2. T-learner vs X-learner  
3. Qini/AUUC ≠ ROC-AUC  
4. Budget policy: uplift-top-k vs risk-top-k vs treat-everyone  
5. Homogeneous vs heterogeneous effects (Hillstrom vs synthetic)

## Caveats

Unconfoundedness (Hillstrom is RCT); short outcome window; noisy CATE; unit economics are knobs; causalml/econml optional — S/T/X reimplemented for clean installs.

## License

MIT for code. Hillstrom data: original MineThatData challenge terms.
