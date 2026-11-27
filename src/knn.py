"""k-nearest-neighbour regression that does not depend on row order.

Daily yield changes are whole basis points, so many training days sit at exactly the
same distance from a query. Library kNN then picks among tied days by row order or
search-tree layout, which changed our results between library versions.

Here every day strictly closer than the k-th distance gets weight 1, and the days tied
at the k-th distance share the remaining weight equally. This equals the average
prediction over every way of breaking the ties, so it is deterministic and order-free.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TieAwareKNN(RegressorMixin, BaseEstimator):
    def __init__(self, k: int = 5, rtol: float = 1e-9, atol: float = 1e-9):
        # Distinct distances between whole-bp inputs differ by far more than these
        # tolerances; float noise in the inputs (about 1e-14) is well inside them.
        self.k = k
        self.rtol = rtol
        self.atol = atol

    def fit(self, X, y):
        X = np.asarray(X, float)
        # Standardize with the training window only (same as StandardScaler, ddof=0).
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.X_ = X
        self.y_ = np.asarray(y, float)
        if not 1 <= self.k <= len(self.y_):
            raise ValueError(f'k={self.k} with {len(self.y_)} training days')
        return self

    def predict(self, X):
        Q = np.asarray(X, float)
        w = 1.0 / self.scale_**2
        # Squared distances from raw differences, so days at the same integer offsets
        # from a query get bit-identical distances (no rounding from centring first).
        d = np.einsum('qij,j->qi', (Q[:, None, :] - self.X_[None, :, :]) ** 2, w)
        k = self.k
        dk = np.partition(d, k - 1, axis=1)[:, k - 1:k]
        tol = self.rtol * dk + self.atol
        inside = d < dk - tol
        tied = np.abs(d - dk) <= tol
        n_in = inside.sum(axis=1)
        n_tied = tied.sum(axis=1)
        share = (k - n_in) / n_tied
        return (inside @ self.y_ + share * (tied @ self.y_)) / k
