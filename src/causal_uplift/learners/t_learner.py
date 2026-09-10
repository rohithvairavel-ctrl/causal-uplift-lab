"""T-learner: separate outcome models for treated and control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseMetalearner


class TLearner(BaseMetalearner):
    name = "T-learner"

    def __init__(self, model):
        self.model = model
        self.model0_ = None
        self.model1_ = None

    def fit(self, X, y, t):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        t = np.asarray(t).astype(int)
        self.model0_ = clone(self.model)
        self.model1_ = clone(self.model)
        self.model0_.fit(X.iloc[t == 0], y[t == 0])
        self.model1_.fit(X.iloc[t == 1], y[t == 1])
        return self

    def _predict(self, model, X) -> np.ndarray:
        X = pd.DataFrame(X)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return np.asarray(model.predict(X), dtype=float)

    def predict_mu(self, X):
        return self._predict(self.model0_, X), self._predict(self.model1_, X)

    def predict_cate(self, X) -> np.ndarray:
        mu0, mu1 = self.predict_mu(X)
        return mu1 - mu0
