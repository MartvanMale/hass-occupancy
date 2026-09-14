"""The step in front of every model fit that fills columns with no values."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from occupancy_forecast.estimators import UnobservedToConstant


def _frame(rows: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((rows, 3)), columns=["a", "b", "c"]), rng.random(rows)


def test_a_column_with_no_values_does_not_stop_the_fit():
    """scikit-learn 1.9 raises on such a column; this is #23."""
    X, y = _frame()
    X["b"] = np.nan
    model = Pipeline([("unobserved", UnobservedToConstant()),
                      ("model", HistGradientBoostingRegressor(max_iter=5))])
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()


def test_the_columns_to_fill_are_decided_at_fit():
    X, _ = _frame()
    X["b"] = np.nan
    step = UnobservedToConstant().fit(X)
    serving = X.copy()
    serving["b"] = 0.5
    serving["c"] = np.nan

    out = step.transform(serving)
    assert (out["b"] == 0.0).all(), "blank at fit: filled at serve"
    assert out["c"].isna().all(), "observed at fit: a gap at serve stays a gap"

    array = step.transform(serving.to_numpy())
    assert (array[:, 1] == 0.0).all() and np.isnan(array[:, 2]).all()


def test_a_frame_with_no_blank_column_passes_through_unchanged():
    X, _ = _frame()
    X.iloc[::7, 1] = np.nan
    pd.testing.assert_frame_equal(UnobservedToConstant().fit_transform(X), X)
