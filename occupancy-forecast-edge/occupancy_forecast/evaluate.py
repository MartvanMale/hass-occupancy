"""Honest evaluation: the folds, and the one implementation of the metrics.

Expanding calendar folds, as adjacent slots are near duplicates, embargoed in
TIME, as the table holds several subjects per timestamp. Brier and log-loss are
primary; MAE rewards a confident 0/1 and must never choose between settings.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config

# 7-day windows, not 14: at 14 too few folds exist for a sign test to pass
# short of a clean sweep. Never re-pick the width after seeing which one
# passes; that is how a validation harness becomes decoration.
TEST_DAYS = 7
MIN_TRAIN_DAYS = 45
MIN_TEST_ROWS = 200

# Below this span `fold_geometry` tapers, so a fresh install need not wait seven
# weeks; the ship gate is unchanged and still demands a real effect size.
FULL_GEOMETRY_DAYS = MIN_TRAIN_DAYS + TEST_DAYS
MIN_TRAINABLE_DAYS = 10

# Probabilities are clipped before log-loss: a confident baseline that says 0.0
# and is wrong once would otherwise score infinity and swamp every average.
EPS = 1e-6

# Above this fraction of the slot spent home, the slot counts as "home" for the
# proper scoring rules. The fractional target is kept alongside.
HOME_THRESHOLD = 0.5


@dataclass
class Scores:
    n: int
    base_rate: float
    brier: float
    log_loss: float
    auc: float
    mae_frac: float

    def skill_vs(self, reference: "Scores") -> float:
        """Brier skill score against a reference, in percent."""
        if reference.brier <= 0:
            return float("nan")
        return 100.0 * (1.0 - self.brier / reference.brier)


def score(y_frac: np.ndarray, p: np.ndarray) -> Scores:
    """Score predicted P(home) against the observed fraction-of-slot-at-home."""
    y_frac = np.asarray(y_frac, dtype=float)
    p = np.asarray(p, dtype=float)
    keep = ~(np.isnan(y_frac) | np.isnan(p))
    y_frac, p = y_frac[keep], p[keep]
    if len(y_frac) == 0:
        return Scores(0, float("nan"), float("nan"), float("nan"),
                      float("nan"), float("nan"))

    y = (y_frac >= HOME_THRESHOLD).astype(float)
    q = np.clip(p, EPS, 1 - EPS)

    brier = float(np.mean((q - y) ** 2))
    logloss = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return Scores(
        n=int(len(y)),
        base_rate=float(y.mean()),
        brier=brier,
        log_loss=logloss,
        auc=_auc(y, q),
        mae_frac=float(np.mean(np.abs(p - y_frac))),
    )


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via the rank identity; ties get average ranks."""
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def reliability(y_frac: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Calibration curve. HA acts on the probability, so ranking is not enough:
    a model can have excellent AUC and still say 0.9 when it means 0.6.
    """
    y_frac = np.asarray(y_frac, dtype=float)
    p = np.asarray(p, dtype=float)
    keep = ~(np.isnan(y_frac) | np.isnan(p))
    y = (y_frac[keep] >= HOME_THRESHOLD).astype(float)
    p = p[keep]

    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1],
                     "n": int(mask.sum()),
                     "predicted": float(p[mask].mean()),
                     "observed": float(y[mask].mean())})
    return pd.DataFrame(rows)


@dataclass
class Fold:
    index: int
    test_start: pd.Timestamp
    test_stop: pd.Timestamp
    train_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)


def fold_geometry(times: pd.Series, n_subjects: int | None = None) -> dict:
    """Fold widths the history supports, tapering below `FULL_GEOMETRY_DAYS`.
    Counted in ORIGINS, never rows: one origin is 48 rows on the long table, so
    callers pass `times` reduced to one entry per origin.
    """
    times = pd.to_datetime(pd.Series(times), utc=True)
    span_days = max(0.0, (times.max() - times.min()) / pd.Timedelta(days=1))
    n_subjects = max(1, n_subjects if n_subjects is not None else len(config.SUBJECTS))

    rows_per_day = (1440 // config.GRID_MINUTES) * n_subjects
    if span_days >= FULL_GEOMETRY_DAYS:
        return {"test_days": TEST_DAYS, "min_train_days": MIN_TRAIN_DAYS,
                "min_test_rows": MIN_TEST_ROWS}

    test_days = max(2, int(span_days // 5))
    return {"test_days": test_days,
            "min_train_days": max(5, int(span_days) - 3 * test_days),
            "min_test_rows": max(1, int(1.5 * rows_per_day))}


def calendar_folds(times: pd.Series, *, embargo: pd.Timedelta | pd.Series,
                   test_days: int = TEST_DAYS, min_train_days: int = MIN_TRAIN_DAYS,
                   min_test_rows: int = MIN_TEST_ROWS) -> list[Fold]:
    """Expanding calendar folds; `min_train_days` is 45 to cover the four-week
    climatology. `embargo` may be per row, as the long table needs, and either
    way the row's TARGET must land before `test_start`.
    """
    times = pd.to_datetime(pd.Series(times).reset_index(drop=True), utc=True)
    if isinstance(embargo, pd.Series):
        embargo = pd.Series(embargo).reset_index(drop=True)
    begin, end = times.min(), times.max()
    step = pd.Timedelta(days=test_days)
    test_start = begin + pd.Timedelta(days=min_train_days)

    folds: list[Fold] = []
    while test_start < end:
        test_stop = test_start + step
        train_idx = np.flatnonzero(((times + embargo) < test_start).to_numpy())
        test_idx = np.flatnonzero(((times >= test_start) & (times < test_stop)).to_numpy())
        if len(test_idx) >= min_test_rows and len(train_idx) >= min_test_rows:
            folds.append(Fold(len(folds), test_start, test_stop, train_idx, test_idx))
        test_start = test_stop

    return folds


def embargo_for_rows(horizons: pd.Series) -> pd.Series:
    """`embargo_for`, vectorised over a long table's `horizon_h` column."""
    return (pd.to_timedelta(pd.Series(horizons).to_numpy(), unit="h")
            + pd.Timedelta(minutes=config.GRID_MINUTES))


def sign_test(wins: int, trials: int) -> float:
    """Two-sided p for `wins` of `trials` under a fair coin. The fold counts are
    small and the intuition bad: at 8 folds, 8/8 is p=0.008 but 6/8 is p=0.29.
    """
    if trials == 0:
        return float("nan")
    tail = sum(math.comb(trials, k) for k in range(wins, trials + 1)) / 2 ** trials
    return float(min(1.0, 2 * tail))


def embargo_for(horizon: int) -> pd.Timedelta:
    """The minimum honest gap between train and test for a given horizon: one
    slot beyond it, so the last training row's target falls strictly before the
    first test row.
    """
    return pd.Timedelta(hours=horizon) + pd.Timedelta(minutes=config.GRID_MINUTES)


def summarize(scores: list[Scores]) -> dict:
    """Pool per-fold scores, keeping the spread rather than only the mean."""
    if not scores:
        return {}
    weights = np.array([s.n for s in scores], dtype=float)
    def pooled(attr: str) -> float:
        values = np.array([getattr(s, attr) for s in scores], dtype=float)
        keep = ~np.isnan(values)
        if not keep.any():
            return float("nan")
        return float(np.average(values[keep], weights=weights[keep]))

    brier = [s.brier for s in scores]
    out = {name: pooled(name) for name in
           ("base_rate", "brier", "log_loss", "auc", "mae_frac")}
    out["n"] = int(weights.sum())
    out["n_folds"] = len(scores)
    out["brier_fold_min"] = float(np.nanmin(brier))
    out["brier_fold_max"] = float(np.nanmax(brier))
    out["per_fold"] = [asdict(s) for s in scores]
    return out
