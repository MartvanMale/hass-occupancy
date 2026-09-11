"""Fit two families of model, let the gate pick per horizon, and score honestly.

Direct multi-horizon, never recursive: the target is always the real slot at
`t + h`. A dedicated fit per horizon and one pooled fit over all 48 are both
trained on the same folds; `choose` picks per horizon and the crossover is
measured, not hardcoded. The independent unit is the person-day, not the
melted row -- see MIN_SAMPLES_LEAF.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import baseline, config, evaluate, features, log

_log = log.get(__name__)

MODEL_VERSION = "0.4.1"

# Minimum Brier skill over the best baseline before a horizon is published at
# all; below it `predict` publishes nothing rather than a number no better than
# arithmetic.
MIN_SHIP_SKILL_PCT = 5.0

# 5% is only meaningful with folds behind it: on two or three the gate reduces
# to a coin flip, so demand a bigger effect when there is less evidence.
FEW_FOLDS = 4
FEW_FOLDS_SKILL_PCT = 15.0


class Phases:
    """Elapsed seconds per named stretch of a train. An accumulator, not a log
    line per phase -- a phase that runs 48 times wants to be one total;
    `.line()` is logged once."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = (self.seconds.get(name, 0.0)
                                  + time.perf_counter() - started)

    def as_dict(self) -> dict:
        return {name: round(value, 1) for name, value in self.seconds.items()}

    def line(self) -> str:
        return ", ".join(f"{name} {value:.1f}s"
                         for name, value in self.as_dict().items())


def fold_record_allows(beat: int, n_folds: int) -> bool:
    """Whether the per-fold record permits shipping: the sign test gates only
    in the REJECTING direction.

    A model that wins 4 of 19 folds is refused; one that wins 9 of 19 has
    proven nothing, and the skill bar decides instead of a coin. The asymmetry
    is deliberate: nothing serves a refused horizon, so refusing too readily
    costs it outright.
    """
    if n_folds == 0:
        return False
    losses = n_folds - beat
    return not (beat * 2 < n_folds
                and evaluate.sign_test(losses, n_folds) < 0.05)


def min_ship_skill_pct(n_folds: int) -> float:
    """The skill bar for this many folds. Constant once the evidence is there."""
    return FEW_FOLDS_SKILL_PCT if n_folds <= FEW_FOLDS else MIN_SHIP_SKILL_PCT
EVALUATION = "rolling-origin-embargoed"

MODELS_DIR = config.MODELS_DIR
FEATURES_PATH = config.FEATURES_PATH

CATEGORICAL_FEATURES = ["subject"]

# How many of an origin's 48 horizon-rows a FIT sees. The melt adds rows, not
# information, and one full fit costs minutes on the box that runs the house.
# Drawn per row rather than per origin; TEST rows are never subsampled -- the
# gate sees all 48.
TRAIN_HORIZONS_PER_ORIGIN = 12
TRAIN_SAMPLE_RATE = TRAIN_HORIZONS_PER_ORIGIN / len(config.HORIZONS_H)

# The leaf floor scaled to what a fit holds: after the melt and the subsample
# an origin carries TRAIN_HORIZONS_PER_ORIGIN rows, so this keeps the floor
# meaning "about fifty origins". Leaves and iterations go the other way,
# because one model now does what 48 did.
MIN_SAMPLES_LEAF = 100
MAX_LEAF_NODES = 63


def subsample(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A quarter of the rows, deterministically. See TRAIN_HORIZONS_PER_ORIGIN."""
    if TRAIN_SAMPLE_RATE >= 1.0 or frame.empty:
        return frame
    rng = np.random.default_rng(seed)
    keep = rng.random(len(frame)) < TRAIN_SAMPLE_RATE
    if not keep.any():
        return frame
    return frame.iloc[np.flatnonzero(keep)]


def origin_features() -> list[str]:
    """Features available at the origin, for every horizon. A function,
    because which people exist is discovered."""
    return [
        "state_now",
        "minutes_in_state",
        "coverage",
        *features.CALENDAR_COLUMNS,
        *features.extra_origin_columns(),
        *features.zone_columns(),
        *(f"other_{slug}" for slug in config.all_slugs()),
        *features.PROXIMITY_COLUMNS,
    ]


def may_be_nan() -> set[str]:
    """Columns a row may be missing without being dropped.

    DERIVED from config, never a literal: a literal that misses a person's
    `other_*` column silently drops every row for them, failing as "no folds".
    """
    return {
        "minutes_in_state",
        *(f"other_{slug}" for slug in config.all_slugs()),
        *features.zone_columns(),
        *features.PROXIMITY_COLUMNS,
    }


def base_features() -> list[str]:
    """The feature list. One, not forty-eight: `horizon_h` is an ordinary
    feature, which is the point of the pooled fit. The daily-lag gate is
    applied upstream, in `features.long_frame`.
    """
    return [
        *origin_features(),
        *features.long_shipped_columns(),
        features.HORIZON_COLUMN,
        *CATEGORICAL_FEATURES,
    ]


def features_for(horizon: int) -> list[str]:
    """The DEDICATED family's feature list: one horizon, wide columns.

    The lag gate is enforced by NOT NAMING the leaky column, which is why it can
    sit in the parquet harmlessly. Two mechanisms, one rule.
    """
    lag_columns = [f"tgt{horizon}h_lag{days}d" for days in features.safe_daily_lags(horizon)]
    return [
        *origin_features(),
        *features.target_calendar_columns(horizon),
        *features.extra_target_columns(horizon),
        *lag_columns,
        *features.cross_subject_lag_columns(horizon),
        features.climatology_column(horizon),
        features.slot_climatology_column(horizon),
        *CATEGORICAL_FEATURES,
    ]


def nan_allowed_for(horizon: int) -> set[str]:
    """Everything target-relative for one horizon, served extras included: a
    candidate that is served is NaN through its own warm-up, and requiring it
    drops those rows."""
    return {
        *may_be_nan(),
        *features.extra_target_columns(horizon),
        *(f"tgt{horizon}h_lag{days}d" for days in features.safe_daily_lags(horizon)),
        *features.cross_subject_lag_columns(horizon),
        features.climatology_column(horizon),
        features.slot_climatology_column(horizon),
    }


def required_origin_columns() -> list[str]:
    """The origin columns a row must carry to be fitted at all. One set for
    both families and for the baseline ladder, so the gate compares two means
    over the same denominator."""
    return [c for c in origin_features() if c not in may_be_nan()]


def columns_for(horizon: int) -> list[str]:
    """Every column one dedicated run touches. The parquet holds a thousand
    columns and each of 48 runs would otherwise load all of them to use
    forty."""
    return sorted({*features_for(horizon), "time", "subject",
                   f"y_{horizon}h", RESIDUAL_BASE})


def load_for(path: Path, horizon: int) -> pd.DataFrame:
    """The dedicated family's frame for one horizon."""
    available = set(pq.read_schema(path).names)
    wanted = features_for(horizon)
    absent = [c for c in wanted if c not in available]
    if absent:
        raise ValueError(
            f"{path} is missing {len(absent)} of the {len(wanted)} features for "
            f"{horizon}h: {absent[:5]}{' ...' if len(absent) > 5 else ''}. "
            f"The feature list has moved ahead of the table -- rebuild it with "
            f"`python -m occupancy_forecast.features --out {path}`.")

    table = pd.read_parquet(path, columns=columns_for(horizon))
    required = [c for c in wanted if c not in nan_allowed_for(horizon)]
    return (table
            .dropna(subset=[*required, f"y_{horizon}h"])
            .sort_values("time")
            .reset_index(drop=True))


def nan_allowed() -> set[str]:
    """Everything target-relative, NaN through the warm-up by design. On the
    long table this must cover the gated lags: `lag1d` is NaN above +24 h, so
    requiring it would drop half the table."""
    # `long_shipped_columns`, not `long_columns`: a served candidate is as
    # target-relative as the rest, and requiring it silently drops its warm-up.
    return {*may_be_nan(), *features.long_shipped_columns()}


# The model predicts the CHANGE from `state_now`: a tree can only step the
# identity, so hand it the identity for free and ask only for the correction.
# `state_now` at every horizon, because a daily lag reads 0.0 or 1.0 and the
# tree would have to correct that everywhere.
RESIDUAL_BASE = "state_now"


def residual_base(horizon: int) -> str:
    """The column this horizon's fit is a residual off. Takes a horizon and
    ignores it -- a function because the panel asks per horizon; see the note
    on RESIDUAL_BASE before making it vary."""
    return RESIDUAL_BASE


def _encoder() -> ColumnTransformer:
    """One-hot the subject, pass everything else through. Shared by both families."""
    return ColumnTransformer([
        ("subject", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ], remainder="passthrough")


def _dedicated_estimator() -> Pipeline:
    """One horizon's model: gradient boosting on the residual off `state_now`.

    A regressor on `home_frac`, not a classifier, so partial slots survive;
    HistGradientBoosting for native NaN handling. Capacity is deliberately low
    -- the independent unit is the person-day.
    """
    return Pipeline([
        ("encode", _encoder()),
        ("model", HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0, random_state=0,
        )),
    ])


def _pooled_estimator() -> Pipeline:
    """Every horizon at once, with `horizon_h` a feature. Bigger than its
    dedicated sibling because one model holds the horizon axis too; tree
    building dominates the fit, not row count."""
    return Pipeline([
        ("encode", _encoder()),
        ("model", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=MAX_LEAF_NODES,
            min_samples_leaf=MIN_SAMPLES_LEAF, l2_regularization=1.0,
            random_state=0,
        )),
    ])


def horizon_weights(frame: pd.DataFrame, residual: np.ndarray) -> np.ndarray:
    """Weight each row by 1 / the mean squared residual at its horizon.

    Without this the short horizons are ruined: the residual grows with the
    horizon, so pooled, 47 of every 48 rows pull the splits toward
    long-horizon variance. Normalised to mean 1 so this changes the balance,
    not the learning rate.
    """
    horizons = frame[features.HORIZON_COLUMN].to_numpy()
    scale = pd.Series(residual ** 2).groupby(horizons).transform("mean").to_numpy()
    weights = 1.0 / np.maximum(scale, 1e-4)
    return weights / weights.mean()


@dataclass
class Metrics:
    horizon_h: int
    evaluation: str
    n_folds: int
    n_scored: int
    n_train_final: int
    base_rate: float
    brier: float
    log_loss: float
    auc: float
    mae_frac: float
    brier_fold_min: float
    brier_fold_max: float
    best_baseline: str
    best_baseline_brier: float
    skill_vs_best_baseline_pct: float
    folds_beating_best_baseline: int
    sign_test_p: float
    ships: bool
    # Which family this is and which one it beat. `kind` is None when neither
    # cleared the ladder.
    kind: str | None = None
    rival_brier: float | None = None
    rival_kind: str | None = None
    # Brier per subject, out of fold: the headline pools them, which left "is
    # the house model better than the house baselines" unanswerable.
    brier_by_subject: dict = field(default_factory=dict)
    fallback: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    per_fold: list = field(default_factory=list)
    reliability: list = field(default_factory=list)


def read_wide(path: Path, subjects: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The built table, checked against the feature list first so a stale
    table gets the explanation rather than a pyarrow error. Read ONCE per train
    and passed around -- if you add a third caller, hand it the frame."""
    available = set(pq.read_schema(path).names)
    wanted = [c for c in origin_features() if c not in ("subject",)]
    absent = [c for c in wanted if c not in available]
    if absent:
        raise ValueError(
            f"{path} is missing {len(absent)} of the {len(wanted)} origin "
            f"features: {absent[:5]}{' ...' if len(absent) > 5 else ''}. "
            f"The feature list has moved ahead of the table -- rebuild it with "
            f"`python -m occupancy_forecast.features --out {path}`.")

    table = pd.read_parquet(path)
    if subjects is not None:
        table = table[table["subject"].isin(subjects)]
    return table.sort_values("time").reset_index(drop=True)


def to_long(wide: pd.DataFrame, subjects: tuple[str, ...] | None = None,
            horizons=None) -> pd.DataFrame:
    """Melt and drop only what genuinely cannot be used."""
    frame = features.long_frame(wide, horizons=horizons, subjects=subjects)
    required = [c for c in base_features() if c not in nan_allowed()]
    return (frame
            .dropna(subset=[*required, features.TARGET_COLUMN])
            .sort_values(["time", features.HORIZON_COLUMN])
            .reset_index(drop=True))


def fit_pooled(estimator: Pipeline, frame: pd.DataFrame) -> Pipeline:
    """Fit on the residual off `state_now`, weighted per horizon; see
    `horizon_weights`."""
    residual = (frame[features.TARGET_COLUMN] - frame[RESIDUAL_BASE]).to_numpy()
    estimator.fit(frame[base_features()], residual,
                  model__sample_weight=horizon_weights(frame, residual))
    return estimator


def predict_pooled(estimator: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Add the residual back onto `state_now` and read it as a probability."""
    residual = estimator.predict(frame[base_features()])
    return np.clip(residual + frame[RESIDUAL_BASE].to_numpy(), 0.0, 1.0)


def fit_dedicated(estimator: Pipeline, frame: pd.DataFrame, horizon: int) -> Pipeline:
    """One horizon's fit. No horizon weighting: there is only one horizon."""
    residual = (frame[f"y_{horizon}h"] - frame[RESIDUAL_BASE]).to_numpy()
    estimator.fit(frame[features_for(horizon)], residual)
    return estimator


def predict_dedicated(estimator: Pipeline, frame: pd.DataFrame,
                      horizon: int) -> np.ndarray:
    residual = estimator.predict(frame[features_for(horizon)])
    return np.clip(residual + frame[RESIDUAL_BASE].to_numpy(), 0.0, 1.0)


def _scores_by_fold(scored: pd.DataFrame, target: str, n_folds: int) -> list:
    """Per-fold `Scores` indexed by fold NUMBER: `ships` walks this list
    positionally against the ladder, so every fold gets an entry even when
    empty."""
    by_fold = dict(iter(scored.groupby("fold", sort=True)))
    return [
        evaluate.score(g[target].to_numpy(), g["p"].to_numpy())
        if (g := by_fold.get(index)) is not None
        else evaluate.score(np.array([]), np.array([]))
        for index in range(n_folds)
    ]


def _candidate(horizon: int, kind: str, scored: pd.DataFrame, target: str,
               n_folds: int, n_train: int, rungs: dict, wide: pd.DataFrame) -> Metrics:
    """Score one candidate family at one horizon against the baseline ladder."""
    fold_scores = _scores_by_fold(scored, target, n_folds)
    pooled = evaluate.summarize(fold_scores)

    best_name, best = min(
        ((name, stats) for name, stats in rungs.items() if stats),
        key=lambda item: item[1]["brier"], default=("none", {"brier": float("nan")}))

    beat = sum(
        1 for i, score in enumerate(fold_scores)
        if not np.isnan(score.brier)
        and score.brier < best.get("per_fold", [{}] * len(fold_scores))[i].get("brier", np.inf))
    # Trials are the folds the model actually SCORED: counting a padded fold as
    # a loss biased the sign test toward refusal where history is thinnest.
    trials = sum(1 for score in fold_scores if not np.isnan(score.brier))
    p_value = evaluate.sign_test(beat, trials)

    # Ship only where the model earns its place: a real effect, and a fold
    # record that has not been shown to be worse than a coin flip.
    ships = bool(
        pooled["brier"] < best["brier"]
        and fold_record_allows(beat, trials)
        and 100.0 * (1.0 - pooled["brier"] / best["brier"])
        >= min_ship_skill_pct(trials)
    )

    # How the ladder's winner was calibrated. EVIDENCE, not a serving path --
    # kept in the artifact because removing a field would cost a MODEL_VERSION
    # bump. The name stored is the wide one, whichever family this is.
    lags = features.safe_daily_lags(horizon)
    if best_name == "persistence" or not lags:
        wide_column = RESIDUAL_BASE
    else:
        wide_column = f"tgt{horizon}h_lag{min(lags)}d"
    fallback_column = wide_column
    # Two columns, then the dropna: dropping on the whole table copies eleven
    # hundred unread columns per surviving row.
    shrink_frame = wide[[f"y_{horizon}h", wide_column]].dropna()
    weight, base = baseline._fit_shrink(
        shrink_frame, shrink_frame[wide_column].to_numpy(), horizon)

    # Out of fold, so the same rows the headline Brier was computed on.
    squared = (scored["p"] - scored[target]) ** 2
    by_subject = {str(name): round(float(part.mean()), 6)
                  for name, part in squared.groupby(scored["subject"])
                  if part.notna().any()}

    return Metrics(
        horizon_h=horizon,
        evaluation=EVALUATION,
        kind=kind,
        brier_by_subject=by_subject,
        # Scored folds, so "won 3 of 3" is what the sign test saw; `per_fold`
        # still carries one entry per window, padded, for the positional walk.
        n_folds=trials,
        n_scored=pooled["n"],
        n_train_final=n_train,
        base_rate=pooled["base_rate"],
        brier=pooled["brier"],
        log_loss=pooled["log_loss"],
        auc=pooled["auc"],
        mae_frac=pooled["mae_frac"],
        brier_fold_min=pooled["brier_fold_min"],
        brier_fold_max=pooled["brier_fold_max"],
        best_baseline=best_name,
        best_baseline_brier=best["brier"],
        skill_vs_best_baseline_pct=100.0 * (1.0 - pooled["brier"] / best["brier"]),
        folds_beating_best_baseline=beat,
        sign_test_p=p_value,
        ships=ships,
        fallback={"which": best_name, "column": fallback_column,
                  "weight": weight, "base": base},
        baselines={name: stats.get("brier") for name, stats in rungs.items()},
        per_fold=pooled["per_fold"],
        reliability=evaluate.reliability(
            scored[target].to_numpy(), scored["p"].to_numpy()).to_dict("records"),
    )


def choose(dedicated: Metrics | None, pooled: Metrics | None) -> Metrics:
    """Which family serves this horizon.

    The crossover is a property of this household at this much history and is
    deliberately not hardcoded. The bar is absolute: beat the best baseline and
    the fold record, or ship nothing -- beating the rival family while losing
    to persistence is not winning.
    """
    runners = [m for m in (dedicated, pooled) if m is not None]
    if not runners:
        raise ValueError("no candidate produced metrics")
    shipping = [m for m in runners if m.ships]
    winner = (min(shipping, key=lambda m: m.brier) if shipping
              else min(runners, key=lambda m: m.brier))
    # The loser's number, so the crossover is visible on the status page.
    other = [m for m in runners if m is not winner]
    winner.rival_brier = other[0].brier if other else None
    winner.rival_kind = other[0].kind if other else None

    if not shipping:
        # Nothing beat the ladder, so `kind` has no answer and neither does the
        # comparison hanging off it.
        winner.ships = False
        winner.kind = winner.rival_kind = None
        winner.rival_brier = None
    return winner


# ---------------------------------------------------------------------------
# The two training paths. Both families are cut on ONE set of fold windows,
# computed once and handed to everything -- that is what makes `choose` a
# comparison.
# ---------------------------------------------------------------------------

def shared_windows(path: Path | pd.DataFrame) -> tuple[list, dict]:
    """The fold windows every candidate is scored on, and their geometry.

    Cut from the ORIGINS, because one origin is 48 pooled rows and counting
    rows would inflate the geometry by 48. The embargo used here is the worst
    case.
    """
    wide = read_wide(path) if isinstance(path, (str, Path)) else path
    origins = (wide[["subject", "time"]].drop_duplicates()
               .sort_values("time").reset_index(drop=True))
    geometry = evaluate.fold_geometry(origins["time"])
    folds = evaluate.calendar_folds(
        origins["time"], embargo=evaluate.embargo_for(max(config.HORIZONS_H)),
        **geometry)
    if not folds:
        raise ValueError(f"no folds from {len(origins)} origins")
    return [(f.test_start, f.test_stop) for f in folds], geometry


def _one_ladder(frame: pd.DataFrame, horizon: int, geometry: dict, windows: list,
                settings):
    """One horizon's baseline ladder in a worker. `config.configure` first:
    the climatology rungs group on the LOCAL calendar, and an unconfigured
    worker scores them in UTC."""
    if settings is not None:
        config.configure(settings)
    started = time.perf_counter()
    rungs = baseline.run(frame, horizon, geometry=geometry, windows=windows,
                         required=required_origin_columns())
    return horizon, rungs, time.perf_counter() - started


def train_dedicated(path: Path, horizon: int, windows: list) -> tuple[Pipeline, pd.DataFrame, int]:
    """One horizon's own model. Returns the estimator and its out-of-fold runs."""
    frame = load_for(path, horizon)
    target = f"y_{horizon}h"
    embargo = evaluate.embargo_for(horizon)
    times = frame["time"]

    collected = []
    for index, (start, stop) in enumerate(windows):
        train_idx = np.flatnonzero(((times + embargo) < start).to_numpy())
        test_idx = np.flatnonzero(((times >= start) & (times < stop)).to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        estimator = fit_dedicated(_dedicated_estimator(), frame.iloc[train_idx], horizon)
        test = frame.iloc[test_idx]
        part = test[["subject", "time", target]].copy()
        part["p"] = predict_dedicated(estimator, test, horizon)
        part["fold"] = index
        collected.append(part)
    if not collected:
        raise ValueError(f"every fold was empty for {horizon}h")

    return (fit_dedicated(_dedicated_estimator(), frame, horizon),
            pd.concat(collected, ignore_index=True), len(frame))


def _pooled_fold(frame: pd.DataFrame, index: int, start, stop, settings,
                 extras: tuple[str, ...] = ()):
    """One pooled fold in a worker.

    `config.configure` FIRST and it is not optional: a fresh interpreter has
    `TIMEZONE="UTC"` and `PEOPLE=()`, and training in that state does not fail
    -- it produces plausible, wrong models. Predictions travel back, never the
    estimator.
    """
    if settings is not None:
        config.configure(settings)
    # A module global is not inherited by a fresh interpreter: anything that
    # changes what a fit reads has to travel as an argument, or two probe arms
    # come back byte-identical and look like an honest null.
    features.SHIPPED_EXTRAS = extras
    # The embargo is applied PER ROW: one scalar would cover the worst case and
    # throw 47 hours of legal training data away from every short-horizon row.
    times = frame["time"]
    targets = times + evaluate.embargo_for_rows(frame[features.HORIZON_COLUMN])
    train_idx = np.flatnonzero((targets < start).to_numpy())
    test_idx = np.flatnonzero(((times >= start) & (times < stop)).to_numpy())
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    estimator = fit_pooled(_pooled_estimator(), subsample(frame.iloc[train_idx], index))
    test = frame.iloc[test_idx]
    part = test[["subject", "time", features.HORIZON_COLUMN,
                 features.TARGET_COLUMN]].copy()
    part["p"] = predict_pooled(estimator, test)
    part["fold"] = index
    return part


def _pooled_final(frame: pd.DataFrame, seed: int, settings,
                  extras: tuple[str, ...] = ()) -> Pipeline:
    """The shipped pooled model, in a worker, on the whole history at the same
    subsample rate the folds used. The one place a fitted Pipeline crosses the
    process boundary -- it is the largest fit and hiding it in the fan-out
    keeps the workers busy."""
    if settings is not None:
        config.configure(settings)
    features.SHIPPED_EXTRAS = extras
    return fit_pooled(_pooled_estimator(), subsample(frame, seed=seed))


def train_pooled(path: Path | pd.DataFrame, windows: list, horizons=None,
                 n_jobs: int | None = None,
                 phases: Phases | None = None) -> tuple[Pipeline, pd.DataFrame, int]:
    """One model over every horizon. Folds are farmed out longest-first with
    the final refit first in the queue -- the windows expand, so submitting in
    order strands the most expensive task alone. Takes a path or an
    already-read frame."""
    phases = phases if phases is not None else Phases()
    horizons = config.HORIZONS_H if horizons is None else tuple(horizons)
    with phases("pooled melt"):
        wide = read_wide(path) if isinstance(path, (str, Path)) else path
        frame = to_long(wide, horizons=horizons)
    if frame.empty:
        raise ValueError("no usable rows in the feature table after the filter")

    settings = config.SETTINGS
    extras = features.SHIPPED_EXTRAS
    order = range(len(windows) - 1, -1, -1)
    with phases("pooled fits"):
        answers = Parallel(n_jobs=worker_count() if n_jobs is None else n_jobs,
                           backend="loky")([
            delayed(_pooled_final)(frame, len(windows), settings, extras),
            *(delayed(_pooled_fold)(frame, index, *windows[index], settings, extras)
              for index in order),
        ])
    estimator, parts = answers[0], answers[1:]
    collected = [p for p in parts if p is not None]
    if not collected:
        raise ValueError("every pooled fold was empty after the per-row embargo")

    # Back into fold order: `per_fold` lists are read positionally elsewhere.
    collected.sort(key=lambda part: int(part["fold"].iloc[0]))
    return estimator, pd.concat(collected, ignore_index=True), len(frame)


DEDICATED_NAME = "occupancy_{horizon}h.pkl"
POOLED_NAME = "occupancy_pooled.pkl"


def save(estimator: Pipeline, metrics, models_dir: Path = MODELS_DIR,
         name: str = POOLED_NAME, feature_names: list[str] | None = None,
         kind: str = "pooled") -> Path:
    """Persist a model and its verdicts, via a temp file and an atomic rename
    -- /predict may be reading concurrently. `kind` travels in the artifact
    because the two families are served differently."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / name
    tmp = path.with_suffix(".pkl.tmp")
    payload = ({h: asdict(m) for h, m in metrics.items()}
               if isinstance(metrics, dict) else asdict(metrics))
    with tmp.open("wb") as fh:
        # The feature list travels WITH the model, so a feature added here
        # cannot silently desynchronise from what is served.
        pickle.dump({"model": estimator, "version": MODEL_VERSION, "kind": kind,
                     "metrics": payload, "features": feature_names}, fh)
    tmp.replace(path)
    return path


def worker_count() -> int:
    """Cores less one, so Home Assistant keeps a core while a background job
    runs. PROCESSES, not threads: the models are too small for scikit-learn's
    own parallelism, and joblib pins each worker to one thread so the two
    cannot multiply."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:      # not Linux
        cores = os.cpu_count() or 1
    return max(1, cores - 1)


def summary_path(models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / "metrics.json"


def write_summary(horizons: dict, models_dir: Path = MODELS_DIR,
                  failed: dict | None = None,
                  duration_s: float | None = None,
                  phases: dict | None = None) -> Path:
    """The verdicts and what the run cost. `phases` sits beside `duration_s`:
    the total is what a person waiting experiences, the breakdown says what to
    attack."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = summary_path(models_dir)
    path.write_text(json.dumps({
        "model_version": MODEL_VERSION,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "duration_s": duration_s,
        "phases": phases or {},
        "evaluation": EVALUATION,
        "horizons": horizons,
        "failed": failed or {},
    }, indent=2))
    return path


def stamp_duration(seconds: float, models_dir: Path = MODELS_DIR) -> None:
    """Record the whole train's duration after the fact: the feature table and
    the ETA models are built either side of `train_all`."""
    path = summary_path(models_dir)
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    summary["duration_s"] = round(seconds, 1)
    path.write_text(json.dumps(summary, indent=2))


def last_summary(models_dir: Path = MODELS_DIR) -> dict | None:
    """When the models on disk were trained; read at start-up because the
    answer outlives the process. A corrupt file is no answer, not a reason not
    to start."""
    path = summary_path(models_dir)
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    return {"trained_at": summary.get("trained_at"),
            "duration_s": summary.get("duration_s"),
            "failed": summary.get("failed") or {}}


def _dedicated_and_save(path: Path, horizon: int, windows: list, models_dir: Path,
                        settings, extras: tuple[str, ...] = ()) -> tuple:
    """One dedicated horizon end to end in a worker; saving happens here so no
    fitted Pipeline crosses the process boundary."""
    if settings is not None:
        config.configure(settings)
    features.SHIPPED_EXTRAS = extras          # see `_pooled_fold`
    started = time.perf_counter()
    try:
        estimator, scored, n_train = train_dedicated(path, horizon, windows)
    except Exception as err:  # noqa: BLE001
        _log.warning("+%sh dedicated: skipped -- %s", horizon, err)
        return horizon, None, None, str(err), time.perf_counter() - started
    save(estimator, {}, models_dir, DEDICATED_NAME.format(horizon=horizon),
         features_for(horizon), kind="dedicated")
    return horizon, scored, n_train, None, time.perf_counter() - started


def _run_gate(horizons, dedicated: dict, pooled_scored, pooled_rows: int,
              windows: list, rungs: dict, wide: pd.DataFrame,
              failed: dict) -> tuple[dict, dict]:
    """Score both families at every horizon and let `choose` pick."""
    summary, chosen = {}, {}
    for horizon in horizons:
        one = two = None
        if horizon in dedicated:
            scored, n_train = dedicated[horizon]
            one = _candidate(horizon, "dedicated", scored, f"y_{horizon}h",
                             len(windows), n_train, rungs[horizon], wide)
        if pooled_scored is not None:
            part = pooled_scored[
                pooled_scored[features.HORIZON_COLUMN] == float(horizon)]
            if not part.empty:
                two = _candidate(horizon, "pooled", part, features.TARGET_COLUMN,
                                 len(windows), pooled_rows, rungs[horizon], wide)
        if one is None and two is None:
            failed.setdefault(str(horizon), "no candidate produced metrics")
            continue
        winner = choose(one, two)
        summary[str(horizon)] = asdict(winner)
        chosen[horizon] = winner
    return summary, chosen


def train_all(path: Path = FEATURES_PATH, models_dir: Path = MODELS_DIR,
              horizons: tuple[int, ...] = config.HORIZONS_H,
              n_jobs: int | None = None) -> dict:
    """Train both families on the same windows and the same ladder, then let
    `choose` pick a winner per horizon."""
    phases = Phases()
    with phases("read"):
        wide = read_wide(path)
        windows, geometry = shared_windows(wide)
    settings = config.SETTINGS
    extras = features.SHIPPED_EXTRAS

    # The ladder runs once per horizon, shared by both candidates and fanned out
    # in the SAME pool as the dedicated fits -- two blocks meant two barriers.
    # Dedicated tasks go first because they are the longer ones.
    workers = worker_count() if n_jobs is None else n_jobs
    with phases("ladder+dedicated"):
        answers = Parallel(n_jobs=workers, backend="loky")([
            *(delayed(_dedicated_and_save)(path, h, windows, models_dir,
                                           settings, extras)
              for h in horizons),
            *(delayed(_one_ladder)(
                wide[sorted({*baseline.columns_for(h), *required_origin_columns()})],
                h, geometry, windows, settings)
              for h in horizons),
        ])
    results, ladders = answers[:len(horizons)], answers[len(horizons):]
    rungs = {h: r for h, r, _ in ladders}
    dedicated = {h: (scored, n_train) for h, scored, n_train, err, _ in results
                 if err is None}
    failed = {f"{h}h dedicated": err for h, _, _, err, _ in results
              if err is not None}
    # Worker-seconds, not wall clock: the two share a fan-out, so wall time
    # cannot say which to attack.
    phases.seconds["ladder(worker)"] = sum(secs for _, _, secs in ladders)
    phases.seconds["dedicated(worker)"] = sum(secs for *_, secs in results)

    pooled_scored, pooled_rows = None, 0
    try:
        estimator, pooled_scored, pooled_rows = train_pooled(
            wide, windows, horizons=horizons, n_jobs=n_jobs, phases=phases)
    except Exception as err:  # noqa: BLE001
        _log.warning("pooled fit: skipped -- %s", err)
        failed["pooled"] = str(err)

    with phases("gate"):
        summary, chosen = _run_gate(horizons, dedicated, pooled_scored,
                                    pooled_rows, windows, rungs, wide, failed)

    if not summary:
        raise RuntimeError(
            f"every horizon failed to train. First error: "
            f"{next(iter(failed.values()), 'unknown')}")

    # EVERY horizon's verdict goes into every artifact that could serve it, or
    # a horizon reads differently depending on which file loaded first.
    # Dedicated files this run did not write are deleted: they would pass the
    # version check and be served against a metrics.json that says otherwise.
    with phases("write"):
        if pooled_scored is not None:
            save(estimator, chosen, models_dir, POOLED_NAME,
                 base_features(), kind="pooled")
        for horizon in horizons:
            name = DEDICATED_NAME.format(horizon=horizon)
            path_h = models_dir / name
            if horizon not in dedicated or horizon not in chosen:
                if path_h.exists():
                    path_h.unlink()
                    _log.warning("+%sh dedicated: removed a stale artifact from an "
                                 "earlier train; this run produced none", horizon)
                continue
            with path_h.open("rb") as fh:
                artifact = pickle.load(fh)
            save(artifact["model"], {horizon: chosen[horizon]}, models_dir, name,
                 artifact["features"], kind="dedicated")

    _log.info("train_all: %s", phases.line())
    write_summary(summary, models_dir, failed, phases=phases.as_dict())

    # Close the pool rather than parking it: loky prints a wall of
    # leaked-semlock warnings at every container stop otherwise.
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception as err:  # noqa: BLE001
        _log.debug("could not close the worker pool: %s", err)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the occupancy models")
    parser.add_argument("--features", type=Path, default=config.FEATURES_PATH)
    parser.add_argument("--models", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--horizons", type=int, nargs="*", default=list(config.HORIZONS_H))
    args = parser.parse_args(argv)

    from . import runtime
    runtime.bootstrap()

    summary = train_all(args.features, args.models, tuple(args.horizons))

    print(f"{'horizon':>8} {'serves':>10} {'brier':>7} {'rival':>7} "
          f"{'best baseline':>28} {'skill':>8} {'folds':>7}  ships")
    for horizon in args.horizons:
        m = summary.get(str(horizon))
        if m is None:
            continue
        rival = f"{m['rival_brier']:.3f}" if m.get("rival_brier") is not None else "-"
        print(f"{horizon:>7}h {(m['kind'] or 'baseline'):>10} {m['brier']:>7.3f} "
              f"{rival:>7} "
              f"{m['best_baseline'][:20] + ' ' + format(m['best_baseline_brier'], '.3f'):>28} "
              f"{m['skill_vs_best_baseline_pct']:>7.1f}% "
              f"{m['folds_beating_best_baseline']:>3}/{m['n_folds']:<3} "
              f"  {'YES' if m['ships'] else 'no'}")


if __name__ == "__main__":
    main()
