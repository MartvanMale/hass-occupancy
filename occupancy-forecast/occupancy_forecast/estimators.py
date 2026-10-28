"""Pipeline steps shared by the model families.

A column with no values in a fit's rows makes scikit-learn 1.9 raise; this fills it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class UnobservedToConstant(TransformerMixin, BaseEstimator):
    """Fills columns that had no values at fit with 0.0, at fit and at predict."""

    def fit(self, X, y=None):
        # Column at a time, because `pd.isna` on the whole frame allocates a
        # mask as big as the frame.
        width = X.shape[1]
        column = (lambda i: X.iloc[:, i]) if hasattr(X, "iloc") else (lambda i: X[:, i])
        self.unobserved_ = np.fromiter(
            (bool(pd.isna(column(i)).all()) for i in range(width)), bool, width)
        return self

    def transform(self, X):
        cols = np.flatnonzero(self.unobserved_)
        if not len(cols):
            return X
        X = X.copy()
        if hasattr(X, "iloc"):
            X.iloc[:, cols] = 0.0
        else:
            X[:, cols] = 0.0
        return X
