"""X-learner (Künzel et al.): imputed treatment effects + propensity weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .base import BaseMetalearner


class XLearner(BaseMetalearner):
    name = "X-learner"

    def __init__(self, model, propensity_model=None, effect_model=None):
        self.model = model
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, solver="lbfgs"
        )
        # Continuous CATE stage — regressor by default
        self.effect_model = effect_model or HistGradientBoostingRegressor(
            max_depth=4, max_iter=100, learning_rate=0.08
        )
        self.mu0_ = None
        self.mu1_ = None
        self.tau0_ = None
        self.tau1_ = None
        self.g_ = None

    def fit(self, X, y, t):
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        t = np.asarray(t).astype(int)

        self.mu0_ = clone(self.model)
        self.mu1_ = clone(self.model)
        self.mu0_.fit(X.iloc[t == 0], y[t == 0])
        self.mu1_.fit(X.iloc[t == 1], y[t == 1])

        def pred(m, Xx):
            if hasattr(m, "predict_proba"):
                return m.predict_proba(Xx)[:, 1]
            return np.asarray(m.predict(Xx), dtype=float)

        d1 = y[t == 1] - pred(self.mu0_, X.iloc[t == 1])
        d0 = pred(self.mu1_, X.iloc[t == 0]) - y[t == 0]

        self.tau1_ = clone(self.effect_model)
        self.tau0_ = clone(self.effect_model)
        self.tau1_.fit(X.iloc[t == 1], d1)
        self.tau0_.fit(X.iloc[t == 0], d0)

        self.g_ = clone(self.propensity_model)
        self.g_.fit(X, t)
        return self

    def predict_cate(self, X) -> np.ndarray:
        X = pd.DataFrame(X)
        tau0 = np.asarray(self.tau0_.predict(X), dtype=float)
        tau1 = np.asarray(self.tau1_.predict(X), dtype=float)
        g = self.g_.predict_proba(X)[:, 1]
        return g * tau0 + (1.0 - g) * tau1
