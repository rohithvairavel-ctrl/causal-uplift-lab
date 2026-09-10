"""Base interface for metalearners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseMetalearner(ABC):
    """Predict CATE = E[Y|X,T=1] - E[Y|X,T=0]."""

    name: str = "base"

    @abstractmethod
    def fit(self, X, y, t) -> "BaseMetalearner":
        ...

    @abstractmethod
    def predict_cate(self, X) -> np.ndarray:
        ...

    def predict_mu(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Optional: return (mu0, mu1). Default raises."""
        raise NotImplementedError
