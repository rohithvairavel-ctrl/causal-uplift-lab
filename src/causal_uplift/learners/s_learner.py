"""S-learner: single model with treatment as a feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseMetalearner


class SLearner(BaseMetalearner):
    name = "S-learner"

    def __init__(self, model):
        self.model = model
        self.model_ = None

    def fit(self, X, y, t):
        X = pd.DataFrame(X).copy()
        X["_t"] = np.asarray(t).astype(float)
        self.model_ = clone(self.model)
        self.model_.fit(X, y)
        self.feature_cols_ = [c for c in X.columns if c != "_t"]
        return self

    def _mu(self, X, t_val: float) -> np.ndarray:
        X = pd.DataFrame(X).copy()
        X["_t"] = t_val
        # Align columns
        if hasattr(self.model_, "feature_names_in_"):
            cols = list(self.model_.feature_names_in_)
            for c in cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[cols]
        proba = self.model_.predict_proba(X)[:, 1] if hasattr(self.model_, "predict_proba") else self.model_.predict(X)
        return np.asarray(proba, dtype=float)

    def predict_mu(self, X):
        return self._mu(X, 0.0), self._mu(X, 1.0)

    def predict_cate(self, X) -> np.ndarray:
        mu0, mu1 = self.predict_mu(X)
        return mu1 - mu0
