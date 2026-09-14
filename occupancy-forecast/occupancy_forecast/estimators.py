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
        self.unobserved_ = np.asarray(pd.isna(X)).all(axis=0)
        return self

    def transform(self, X):
        X = X.copy()
        cols = np.flatnonzero(self.unobserved_)
        if hasattr(X, "iloc"):
            X.iloc[:, cols] = 0.0
        else:
            X[:, cols] = 0.0
        return X
