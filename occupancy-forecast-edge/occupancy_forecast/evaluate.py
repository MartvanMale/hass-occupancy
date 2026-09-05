"""Honest evaluation: the folds, and the one implementation of the metrics.

Two things live here, and both are load-bearing.

**The folds.** Expanding-window, calendar-anchored, with a *time-based* embargo.
Occupancy is strongly autocorrelated and adjacent 30-minute slots are near
duplicates, so a random split does not measure generalisation at all -- it
measures how well the model memorised the neighbouring slot. The same leak
measured on an adjacent forecasting problem was worth +0.208 R2 to
HistGradientBoosting against +0.002 to a linear model, which is the trap: a
simple model shows almost no difference,
so the split looks fine right up until you use a model with capacity.

The embargo has to be measured in *time*, not rows. The table is sorted by
(subject, time) and holds three subjects, so consecutive index positions are
usually the same timestamp for different people; a row-count embargo would
protect nothing. It must be at least the horizon, because a row timestamped just
before a test window has its target *inside* that window.

**The metrics.** Brier and log-loss on the binarised outcome, primary. MAE on
the fraction is reported but must never be used to choose between settings:
measured 2026-08-31, ranking the baselines by MAE on the fractional target
inverts the Brier ranking and declares persistence the winner at every horizon,
because MAE rewards a confident 0/1 guess and punishes a calibrated 0.65. That
would have sent the whole project in the wrong direction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config

# 7-day test windows, not the 14 this started with.
#
# This was 14 first, and it was a mistake that had nothing to do with the
# results: 173 days of history minus a 27-day outage and a 45-day warm-up
# leaves only 8 windows, and a sign test over 8 folds has essentially no power.
# 6/8 is p=0.29 and 7/8 is p=0.07, so *nothing short of a clean sweep could ever
# clear a 5% bar* -- the gate would have been rejecting on fold count rather
# than on evidence. 7-day windows give 15 folds, still ~1000 rows each, where
# 12/15 is p=0.035.
#
# Both widths were run and the direction agrees at every horizon (see the README
# table); only the significance changes. Recorded here rather than quietly
# switched, because choosing a fold width after seeing which one passes is
# exactly how a validation harness becomes decoration.
TEST_DAYS = 7
MIN_TRAIN_DAYS = 45
MIN_TEST_ROWS = 200

# The geometry above needs 52 days (45 + 7) before a single fold exists, and for
# a long-running install it is the right one -- the numbers in baseline.py and
# the README were measured with it and must not move.
#
# But a fresh install has none of that, and making somebody wait seven weeks to
# see anything at all is its own kind of wrong. Below the full span the geometry
# shrinks to whatever the history can support, and `fold_geometry` is where that
# tapering lives. Two things keep it honest rather than merely encouraging:
#
#   * The ship gate (train.py) is unchanged, and it demands a real effect size,
#     not just a majority of folds. A model that cannot beat its own baseline is
#     not served -- predict.py falls back per horizon. So an early model can add
#     skill or sit out; it cannot make the forecast worse.
#   * `min_test_rows` scales with the household. 200 was silently a
#     three-subject assumption: at 30-minute slots one subject yields 48 rows a
#     day, so a lone person could never have cleared a 200-row window no matter
#     how long they waited.
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
    """Calibration curve. HA acts on the probability, so ranking is not enough.

    A model can have excellent AUC and still say 0.9 when it means 0.6, which
    for a "pre-heat if they will be home" rule is the difference between a warm
    house and a wasted hour of gas.
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
    """Fold widths the available history can actually support.

    At or above `FULL_GEOMETRY_DAYS` this returns the constants unchanged, so a
    mature install evaluates exactly as it always has. Below that it tapers,
    targeting two or three folds rather than none: the alternative is
    `calendar_folds` returning `[]` and training failing with "no folds", which
    is what made a fresh install wait seven weeks.

    `min_test_rows` is derived from the subject count, not fixed. One subject
    produces `1440 / GRID_MINUTES` rows a day, and a window has to hold at least
    a day and a half of them to say anything.

    Counted in ORIGINS -- distinct (subject, slot) -- never in rows. On the long
    table one origin carries 48 rows, one per horizon, so a row count would
    inflate 48x and quietly turn the fresh-install taper into a no-op: a two-day
    window for one subject would "hold" 4,608 rows and clear any floor set here.
    Callers pass `times` already reduced to one entry per origin.
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
    """Expanding-window folds anchored to the calendar.

    Train is everything up to `test_start - embargo`; test is the `test_days`
    window starting at `test_start`. Windows advance by `test_days`, so every
    row after the warm-up is tested exactly once.

    `min_train_days` is 45 rather than the more usual 30 because the longest
    in-table feature is a four-week same-weekday climatology: with a shorter
    warm-up the first fold trains on rows whose climatology column is NaN for
    structural reasons and reads as though the feature were useless.

    `embargo` may be **per row**, and on the long table it must be: one scalar
    would have to cover the worst case (48 h) and would then throw away 47 hours
    of perfectly legal training data from every 1 h row, in every fold. The
    invariant is the same either way and is about the row's TARGET, not its
    origin -- `time + embargo <= test_start` -- which is why the module docstring
    insists the embargo is measured in time rather than in rows.
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
    """Two-sided p for `wins` of `trials` under a fair coin.

    Kept close at hand because the fold counts here are small and the intuition
    is bad: at 8 folds, 8/8 is p=0.008 but 6/8 is p=0.29. Six out of eight is
    not a result, however good the pooled mean looks.
    """
    if trials == 0:
        return float("nan")
    tail = sum(math.comb(trials, k) for k in range(wins, trials + 1)) / 2 ** trials
    return float(min(1.0, 2 * tail))


def embargo_for(horizon: int) -> pd.Timedelta:
    """The minimum honest gap between train and test for a given horizon.

    One extra slot beyond the horizon: the target of the last training row must
    fall strictly before the first test row.
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
