"""Fit two families of model, let the gate pick per horizon, and score honestly.

Direct multi-horizon, never recursive: the target is always the real slot at
`t + h`. Recursion would compound its own error over 96 slots and there is no
way to calibrate that.

**Two families, because one was measured to be wrong at both ends.** Each
horizon used to get its own fit. Collapsing all 48 into one POOLED model, with
`horizon_h` an ordinary numeric feature, fixed the far end emphatically and
broke the near end just as emphatically. MEASURED on 173 days, Brier:

    h        +1      +6     +15   |   +24     +36     +48
    dedicated 0.0437  0.1090  0.1316 | 0.1558  0.1995  0.2124
    pooled    0.0749  0.1162  0.1367 | 0.1454  0.1597  0.1739

They cross at +16 h, monotonically, and it is a bias-variance split. Near the
origin the residual off `state_now` is small and the signal is strong enough to
carry a model of its own, so pooling only dilutes it -- 47 of every 48 rows pull
the splits toward long-horizon variance. Far out the per-horizon fits were
starved, each overfitting the same ~500 person-days, and sharing across horizons
is worth more than anything horizon-specific.

So both are fitted and `choose` picks per horizon. **The crossover is not
hardcoded, deliberately**: +16 h belongs to this household at this much history
and will move as the archive grows. The gate already decided model-versus-
baseline by measurement; this is one more candidate in the same comparison.

The house is a training subject like the people. An intermediate design made it
a learned combination over the people's forecasts instead; it shipped 0 of 48
and CHANGELOG.md records the numbers.

The estimator is deliberately small. The melted table has ~1.2M rows and the
independent unit is still a *person-day*, of which there are about 500 across
173 days. Sizing the model for the row count would be sizing it for 2000x the
information that is there -- see TRAIN_HORIZONS_PER_ORIGIN and MIN_SAMPLES_LEAF,
which exist to keep that from happening quietly.
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

MODEL_VERSION = "0.4.0"

# Minimum Brier skill over the best baseline for a horizon to be published at
# all. Below this the model is not adding anything worth the extra moving
# parts, and `predict` publishes nothing for that horizon rather than a number
# no better than arithmetic.
MIN_SHIP_SKILL_PCT = 5.0

# ...but 5% is only meaningful with enough folds behind it. A fresh install
# evaluates on two or three (see evaluate.fold_geometry), where the gate's
# "beat the baseline more often than not" reduces to 2-of-2 -- a coin flip away
# from shipping on luck. Demand a bigger effect when there is less evidence, so
# an early model has to have found something real rather than merely won twice.
FEW_FOLDS = 4
FEW_FOLDS_SKILL_PCT = 15.0


class Phases:
    """Elapsed seconds per named stretch of a train.

    A train is the one thing here that takes minutes, and until this existed
    there was exactly one number for the whole of it -- so every claim about
    where the time went was an inference from a comment written during some
    earlier optimisation. Making that measurable is the prerequisite for
    changing it.

    Deliberately an accumulator rather than a log line per phase: `log.py` keeps
    the add-on's output thin on purpose, and a phase that runs 48 times wants to
    be one total, not 48 lines. `.line()` is what gets logged, once.
    """

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
    """Whether the per-fold record permits shipping: was it PROVEN a minority?

    The sign test is deliberately not a hard gate in the shipping direction. At
    15 folds only 12h and 36h ever cleared p<0.05, while 6h had the single
    largest effect in the table (39% skill, 11/15 folds, p=0.118) -- refusing
    that would be pretending the test is more informative than it is on this
    much history.

    It IS a gate in the rejecting direction, and that asymmetry is the design.
    This used to demand a strict majority, and MEASURED on 2026-09-02 that was
    the binding constraint rather than skill: 13 of 48 horizons were refused,
    six of them showing +11..+15% Brier skill and failing at 9 of 19 folds
    where 10 were needed, with sign-test p = 1.000 -- a coin flip deciding the
    outcome. Scoring the SERVED curve prequentially -- decide on folds [0,k),
    score on the held-out fold k, pool over k -- the strict majority cost
    **5.07% Brier overall and 9.0% across +30..48 h**, and on the cells where
    the two rules disagree the model beat the baseline in 61 of 80
    (horizon, fold) cells. Those cells are correlated within a fold, so read
    the direction rather than the p; aggregated per fold it is 6 of 8, which is
    not a result on its own. The near horizons did not move (0.1001 -> 0.0998),
    and shipping went 35/48 -> 42/48.

    So: a model that wins 4 of 19 folds is refused -- that is the
    one-good-fortnight case the majority rule existed to catch, and this still
    catches it. One that wins 9 of 19 has proven nothing either way, and the
    skill bar decides instead of a coin.

    The asymmetry is safe because the failure modes are not symmetric, and
    since `predict` stopped serving baselines the argument has got STRONGER,
    not weaker. Shipping too readily costs a model that is merely not clearly
    better than persistence. Refusing too readily now costs the horizon
    outright -- nothing is published for it and the sensor reads `unknown` --
    which is what it was doing at exactly the horizons where a baseline is
    weakest, the far ones, where `same_slot_yesterday` is reading two days back
    and cannot see a weekday. If this gate is ever revisited, the pressure is
    toward the loose side it is already on.

    (The prequential measurement above was made when a refused horizon still
    served its baseline, so those Brier numbers compare two dense curves. The
    direction and the 35/48 -> 42/48 count stand; the cost of the strict rule
    is now larger than 5.07%, because the cells it refuses are not served at
    all rather than served by the baseline.)
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

# How many of an origin's 48 horizon-rows a FIT sees.
#
# The melt turns 25k rows into 1.2M, and adds no information whatsoever doing
# it: the 48 rows of one origin share every origin feature exactly and differ
# only in the target-relative block. The module docstring's unit -- about 500
# person-days -- is unchanged by the melt.
#
# It changes the arithmetic a great deal. MEASURED in the container, one fit on
# 1.1M x 40:
#
#     200 iter / 15 leaves    51 s   -> 13.6 min across 16 folds
#     300 iter / 31 leaves    93 s   -> 24.7 min
#
# against 102 s for the whole 48-model train this replaces -- on the box that
# runs the house, weekly. `worker_count` used to leave a core free for exactly
# this reason; one big fit has no such dial.
#
# So the fit takes a quarter of each origin's rows, drawn per row rather than
# per origin because a Bernoulli draw is one vector op on a million rows and an
# exact-N groupby is not. Every horizon still appears in every fold, tens of
# thousands of times. TEST rows are never subsampled -- the evaluation and the
# ship gate see all 48 horizons of every origin, which is the number that has
# to be honest.
TRAIN_HORIZONS_PER_ORIGIN = 12
TRAIN_SAMPLE_RATE = TRAIN_HORIZONS_PER_ORIGIN / len(config.HORIZONS_H)

# The leaf floor, scaled to what a fit actually holds.
#
# 50 was chosen when a row was one (subject, slot) and it meant "fifty slots".
# After the melt and the subsample an origin carries TRAIN_HORIZONS_PER_ORIGIN
# rows, so this keeps the floor meaning "about fifty origins" rather than
# quietly becoming "about four".
#
# `max_leaf_nodes` and `max_iter` go the other way and are raised, because one
# model now does what 48 did and has to hold the horizon axis as well.
# MEASURED on the real 173-day history, Brier at a few horizons, 6 folds:
#
#     msl=600 leaves=31   h1 0.0957  h6 0.1221  h24 0.1656  h48 0.2174
#     msl=100 leaves=63   h1 0.0826  h6 0.1093  h24 0.1509  h48 0.2017
#
# Better everywhere, and barely more expensive -- tree building dominates the
# fit, not row count, which is also why the subsample above buys so little.
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
    """Features available at the origin, for every horizon.

    A function, not a constant, because which people exist is now discovered
    rather than hardcoded.
    """
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

    DERIVED, and that matters. This used to be a literal set naming
    `other_mart`, `other_tessa`, `office_mart` and so on, while the feature list
    two lines above derived the same names from config. Rename a person and the
    two disagreed: their `other_*` column became *required*, and since
    `other_<self>` is NaN on every one of that person's own rows by
    construction, every row for them was silently dropped. The frame emptied and
    it failed with "no folds" -- a message about fold geometry when the fault
    was in the configuration.

    Everything here is genuinely optional: `minutes_in_state` is NaN before the
    first observed transition, proximity is NaN when nobody has moved, and the
    cross-subject and zone columns are NaN whenever that source is absent --
    which, on a house with no zones ticked and no Proximity integration, is
    always. HistGradientBoosting handles the NaN natively.
    """
    return {
        "minutes_in_state",
        *(f"other_{slug}" for slug in config.all_slugs()),
        *features.zone_columns(),
        *features.PROXIMITY_COLUMNS,
    }


def base_features() -> list[str]:
    """The feature list. One, not forty-eight.

    `horizon_h` is an ordinary numeric feature, which is the whole point of the
    pooled fit: the h<=24 / h>=25 lag regime and the daily periodicity of the
    target are structure the tree can now split on ONCE and share, instead of
    48 independent fits each rediscovering it from the same ~500 person-days.

    The daily-lag gate has not gone anywhere, it has moved upstream.
    `features.long_frame` copies `tgt{h}h_lag{k}d` into `lag{k}d` only for the
    lags `safe_daily_lags` allows, so a lag that reaches past the origin arrives
    NaN by omission. That is deliberately a positive selection rather than a
    mask: a mask is something a future edit can forget, and the model would
    train beautifully on a value it cannot be served.
    """
    return [
        *origin_features(),
        *features.long_shipped_columns(),
        features.HORIZON_COLUMN,
        *CATEGORICAL_FEATURES,
    ]


def features_for(horizon: int) -> list[str]:
    """The DEDICATED family's feature list: one horizon, wide columns.

    The daily-lag gate is the important line. `tgt{h}h_lag{k}d` is `home_frac`
    at `t + h - 24k`, which is only knowable at prediction time when
    `24k >= h` -- at 36 h the target's "yesterday" is twelve hours into the
    future. features.safe_daily_lags owns that rule and test_features asserts it.

    Here the gate is enforced by NOT NAMING the column, which is why the leaky
    ones can sit in the parquet harmlessly. The pooled family cannot do that --
    it has one feature list for every horizon -- so `features.long_frame`
    enforces the same rule by omission instead. Two mechanisms, one rule.
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
    """Everything target-relative for one horizon. The dedicated family's."""
    return {
        *may_be_nan(),
        *(f"tgt{horizon}h_lag{days}d" for days in features.safe_daily_lags(horizon)),
        *features.cross_subject_lag_columns(horizon),
        features.climatology_column(horizon),
        features.slot_climatology_column(horizon),
    }


def columns_for(horizon: int) -> list[str]:
    """Every column one dedicated horizon's run touches -- and only those.

    `features_for` is what the model reads. The four added here are what
    everything around it reads: `time` cuts the folds and rebuilds the target's
    calendar, `subject` groups the climatology rungs, the target is the answer,
    and RESIDUAL_BASE is both what the fit is a residual off and what the
    persistence rung predicts.

    Worth naming separately because reading the rest is not free. The parquet
    holds every horizon's targets and lags side by side -- over a thousand
    columns -- and each of the 48 runs would otherwise load all of it to use
    about forty. Doing that once per horizon was three quarters of the training
    time.
    """
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
    """Everything target-relative, which is NaN through the warm-up by design.

    All of it is an average or a lookup of days that may not exist yet, and
    HistGradientBoosting reads NaN natively. Requiring any of them would drop
    the early rows of every fold rather than letting the model see less.

    On the long table this must also cover the gated lags: `lag1d` is NaN on
    every row above +24 h by construction, so requiring it would drop half the
    table -- the exact shape of the bug `may_be_nan` documents, one axis over.
    """
    # `long_shipped_columns`, not `long_columns`: a candidate that IS being
    # served is as target-relative as the rest and is NaN through its own
    # warm-up. Requiring it silently drops those rows instead -- which cost a
    # measured arm 5% of its rows and made its comparison against the control
    # a comparison of two different row sets.
    return {*may_be_nan(), *features.long_shipped_columns()}


# The model predicts the CHANGE from the current state, not the state itself.
#
# MEASURED 2026-08-31, and it is not a small effect -- Brier, pooled over 8
# folds, direct target against residual target:
#
#        h     persistence    direct    residual
#        1h        0.054       0.051      0.051
#        6h        0.171       0.112      0.108
#       12h        0.233       0.141      0.131
#       24h        0.193       0.201      0.181
#       36h        0.241       0.241      0.208
#       48h        0.202       0.246      0.217
#
# The direct model is *worse than persistence* at 24 h and beyond; the residual
# model is better at every horizon up to 36 h. The reason is structural. At
# multiples of 24 h the strongest baseline is essentially the identity function
# on `state_now` -- daily periodicity means "what you were doing at this time
# yesterday" is most of the answer -- and a regression tree approximates the
# identity badly, since all it can do is step it. Handing the model the identity
# for free and asking only for the correction removes that handicap entirely.
#
# Same reasoning as a thermostat model predicting dT rather than T.
RESIDUAL_BASE = "state_now"

# AND IT STAYS `state_now` AT EVERY HORIZON. Tried and rejected, 2026-09-01:
# making the anchor horizon-dependent -- the nearest legal daily lag from +25 h,
# which is what the winning `same_slot_yesterday` baseline predicts -- on the
# theory that the argument above is really about anchoring on the BEST baseline,
# and that past 6 h this is no longer persistence.
#
# MEASURED over the same 173 days and 15 folds, Brier skill against each
# horizon's best baseline, on one identical feature table:
#
#        anchor             25h     26h     28h     30h     36h   ships
#        state_now         +9.8%   +8.6%   +4.9%   +0.7%  -14.9%  26/48
#        raw daily lag     +4.5%   +4.9%   -2.9%   -8.4%  -22.3%  24/48
#        slot climatology  +3.4%   +3.2%   +0.6%   -8.5%  -11.9%  24/48
#
# It lost the two horizons it was meant to win, and lost them exactly where it
# switches on. The reason is in `baseline._fit_shrink`: a daily lag is a SINGLE
# observation, so as a probability it says 0.0 or 1.0 and means "one sample said
# so". Anchoring on it adds a +/-1 term to every prediction which the tree then
# has to correct everywhere, and it cannot -- shrinking that column toward the
# base rate is worth more (0.202 -> 0.168 at 48 h) than anything the model does
# with it. `state_now` is SMOOTH, and that rather than its skill is what makes
# it a good thing to add a residual to.
#
# A smooth anchor was tried too, in the third row, and is no better. If this is
# revisited, the thing to test is a per-fold SHRUNK anchor -- which stops the
# anchor being a pure function of the horizon and makes it a fitted parameter
# that has to travel in the artifact. A much larger change than it looks.


def residual_base(horizon: int) -> str:
    """The column this horizon's fit is a residual off.

    Takes a horizon and ignores it. A function rather than a bare constant
    because the panel asks per horizon and would otherwise have to know that
    the answer is the same every time -- and because whether it *should* vary
    is a live question that has now been measured once. See the note above
    before making it vary again.
    """
    return RESIDUAL_BASE


def _encoder() -> ColumnTransformer:
    """One-hot the subject, pass everything else through. Shared by both families."""
    return ColumnTransformer([
        ("subject", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ], remainder="passthrough")


def _dedicated_estimator() -> Pipeline:
    """One horizon's model. Gradient boosting on the residual off `state_now`.

    A regressor on `home_frac in [0, 1]` rather than a classifier on the
    binarised outcome: the fraction carries the partial slots (someone who left
    at 08:40 is not the same as someone who left at 08:05). Predictions are
    added back to `state_now`, clipped to [0, 1] and read as probabilities.

    HistGradientBoosting rather than XGBoost because it handles the NaNs that
    survive the feature build natively, and one fewer dependency.

    Capacity is deliberately low -- see the module docstring. `max_iter` 200 with
    15 leaves and a 50-sample leaf floor is roughly half of what suits a table
    with 50x more independent observations. **Unchanged from the design that
    measured 0.0437 at +1 h**, which is the number this family exists to keep.
    """
    return Pipeline([
        ("encode", _encoder()),
        ("model", HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0, random_state=0,
        )),
    ])


def _pooled_estimator() -> Pipeline:
    """Every horizon at once, with `horizon_h` a feature.

    Bigger than its dedicated sibling and measurably needing to be: one model
    holds the horizon axis as well as everything else. MEASURED on the real
    history, Brier, 6 folds:

        msl=600 leaves=31   h1 0.0957  h6 0.1221  h24 0.1656  h48 0.2174
        msl=100 leaves=63   h1 0.0826  h6 0.1093  h24 0.1509  h48 0.2017

    Better everywhere, and barely more expensive -- tree building dominates the
    fit, not row count, which is also why `subsample` buys so little time.
    """
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

    **Without this the short horizons are ruined.** Squared error weights every
    row equally, and the residual off `state_now` grows with the horizon: at
    +1 h it is nearly zero, at +48 h it is most of the target. Pooled, 47 of
    every 48 rows pull the splits toward long-horizon variance and h=1 is fitted
    almost incidentally.

    MEASURED on the real history, Brier at +1 h: 0.0826 unweighted against
    0.0733 weighted; at +2 h, 0.0889 against 0.0819, which is the difference
    between losing to persistence and beating it. The long horizons improve too
    (+48 h: 0.2017 -> 0.1973), so this is not a trade.

    Normalised to mean 1 so the weights change the balance between horizons and
    not the effective learning rate.
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
    # Which family this is, and which one it beat. `kind` is None when neither
    # cleared the ladder and nothing is published for the horizon. `rival_*` is
    # here so the
    # crossover between the two families is visible on the Data tab rather than
    # being something only a plan document knows.
    kind: str | None = None
    rival_brier: float | None = None
    rival_kind: str | None = None
    # Brier per SUBJECT, out of fold. The headline `brier` pools the subjects,
    # which meant "is the house model better than the house baselines" was a
    # question nobody could answer -- and it is the question a house-specific
    # design has to beat. Three floats a horizon; detail only, not a scalar the
    # list needs.
    brier_by_subject: dict = field(default_factory=dict)
    fallback: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    per_fold: list = field(default_factory=list)
    reliability: list = field(default_factory=list)


def read_wide(path: Path, subjects: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The built table, checked against the feature list before it is read.

    The schema first, so a table that has fallen behind still gets the
    explanation below rather than whatever pyarrow says about a column it was
    asked for and could not find.

    The whole table is read here, where `columns_for` used to slim each of 48
    reads down to about thirty columns. That optimisation existed because the
    same thousand-column file was read 48 times; the wide read happens once per
    train now, so the saving it bought has been kept by deleting the reason for
    it.

    "Once" is load-bearing and was not true for a while: `shared_windows` and
    `train_pooled` each read it again, so a quarter-gigabyte table was
    materialised three times in the parent. Both take a frame now. If you add a
    third caller, pass it the frame.
    """
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
    """Fit on the residual off `state_now`, weighted per horizon.

    See RESIDUAL_BASE for the anchor and `horizon_weights` for why the weights
    are not optional.
    """
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
    """Per-fold `Scores`, indexed by fold NUMBER rather than by what produced rows.

    `ships` counts folds a candidate won by walking this list POSITIONALLY
    against the ladder's `per_fold`, so a fold that is empty here and non-empty
    there would silently shift every later comparison by one. With three
    candidates being compared that stops being a latent bug and becomes a wrong
    answer, which is why every fold gets an entry even when it is empty.
    """
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
    p_value = evaluate.sign_test(beat, len(fold_scores))

    # Ship only where the model earns its place: a real effect, and a fold
    # record that has not been shown to be worse than a coin flip.
    ships = bool(
        pooled["brier"] < best["brier"]
        and fold_record_allows(beat, len(fold_scores))
        and 100.0 * (1.0 - pooled["brier"] / best["brier"])
        >= min_ship_skill_pct(len(fold_scores))
    )

    # How the ladder's winner was calibrated: which column it reads and the
    # shrink fitted on the whole history. EVIDENCE, not a serving path --
    # `predict` no longer evaluates a baseline at all, because a horizon that
    # does not ship is not published. Kept in the artifact anyway: it is what
    # makes the bake-off on the Data tab checkable, and removing a field from
    # the pickle would cost a `MODEL_VERSION` bump and invalidate every
    # artifact on disk to save a few bytes.
    #
    # The name stored is the WIDE one, whichever family this candidate is, so
    # it is readable against a feature row rather than against the melt
    # `_model_curve` does for the pooled model.
    lags = features.safe_daily_lags(horizon)
    if best_name == "persistence" or not lags:
        wide_column = RESIDUAL_BASE
    else:
        wide_column = f"tgt{horizon}h_lag{min(lags)}d"
    fallback_column = wide_column
    # The two columns, then the dropna. `_fit_shrink` reads the target and the
    # array handed to it and nothing else, so dropping on the whole table copied
    # eleven hundred unread columns of every surviving row -- a quarter of a
    # gigabyte, ninety-six times, serially in the parent.
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
        n_folds=n_folds,
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
    """Which family actually serves this horizon.

    MEASURED on 173 days, the two families cross cleanly at h=16: a dedicated
    fit wins h=1..15 (+71% at h=1, where the residual off `state_now` is small
    and pooling only dilutes it) and the pooled fit wins h=16..48 (-18% at h=48,
    where 48 independent fits were each overfitting the same ~500 person-days).

    That crossover is NOT hardcoded, and deliberately. It is a property of this
    household at this much history, and it will move as the archive grows. The
    gate already decides model-vs-baseline by measurement; this is one more
    candidate in the same comparison.

    The bar is unchanged and absolute: a candidate must beat the best BASELINE
    by `min_ship_skill_pct` and win a per-fold majority, or it does not ship at
    all. Only among those that clear it does the lower Brier win -- beating the
    rival family while losing to persistence is not winning.
    """
    runners = [m for m in (dedicated, pooled) if m is not None]
    if not runners:
        raise ValueError("no candidate produced metrics")
    shipping = [m for m in runners if m.ships]
    winner = (min(shipping, key=lambda m: m.brier) if shipping
              else min(runners, key=lambda m: m.brier))
    # The loser's number, so the crossover is visible on the status page rather
    # than being something only a plan document knows.
    other = [m for m in runners if m is not winner]
    winner.rival_brier = other[0].brier if other else None
    winner.rival_kind = other[0].kind if other else None

    if not shipping:
        # Nothing beat the ladder. `kind` is "which family serves", so it has no
        # answer here and neither does the comparison hanging off it: naming a
        # losing family beside `kind: null` reads as "no family served, and it
        # was the pooled one". `brier` still carries the better of the two and
        # `ships` still says a baseline won, which is the whole story for a
        # horizon where both families lost.
        winner.ships = False
        winner.kind = winner.rival_kind = None
        winner.rival_brier = None
    return winner


# ---------------------------------------------------------------------------
# The two training paths
#
# Both are cut on ONE set of fold windows, computed once from the origins and
# handed to the dedicated fits, the pooled fits and the baseline ladder alike.
# That is what makes `choose` a comparison rather than a coincidence.
# ---------------------------------------------------------------------------

def shared_windows(path: Path | pd.DataFrame) -> tuple[list, dict]:
    """The fold windows every candidate is scored on, and the geometry behind them.

    Cut from the ORIGINS -- distinct (subject, slot) -- because one origin
    carries 48 rows in the pooled frame and counting rows would inflate every
    number `fold_geometry` reasons about by 48. The embargo used here is the
    worst case (+48 h) so the windows are honest about how much training history
    a fold really has; each family then applies its own, never tighter.

    Takes either the parquet path or an already-read wide frame, so `train_all`
    can read the table once and hand the same object to everything that needs
    it.
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
    """One horizon's baseline ladder, inside whichever process picks it up.

    `config.configure` first, for the reason spelled out on `_pooled_fold`, and
    it bites differently here: the climatology rungs group on the LOCAL target
    calendar, so an unconfigured worker scores them in UTC and hands back a
    ladder that is subtly too easy in one direction and too hard in the other.
    """
    if settings is not None:
        config.configure(settings)
    started = time.perf_counter()
    rungs = baseline.run(frame, horizon, geometry=geometry, windows=windows)
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
    """One pooled fold, inside whichever process picks it up.

    **`config.configure` first, and it is not optional.** A worker is a fresh
    interpreter: it imports `config` with its module defaults, which are
    `TIMEZONE = "UTC"` and `PEOPLE = ()`. Training in that state does not fail --
    it quietly asks `base_features` for a feature list without this household's
    people in it. The models come out plausible and wrong.

    Predictions travel back, never the estimator: a fitted Pipeline would have
    to be pickled across the process boundary for nothing, since the shipped
    model is refitted on everything afterwards.
    """
    if settings is not None:
        config.configure(settings)
    # And the SAME hazard for the feature switch, found the same way: a probe
    # set `features.SHIPPED_EXTRAS` in the parent, the parent reported the wider
    # feature list, and every worker fitted the narrow one -- so two arms came
    # back byte-identical and looked like an honest null. A module global is not
    # inherited by a fresh interpreter. Anything that changes what a fit reads
    # has to travel as an argument.
    features.SHIPPED_EXTRAS = extras
    # The embargo is applied PER ROW. A 1 h row and a 48 h row from the same
    # origin are honest at different distances from the test window; one scalar
    # would have to cover the worst case and would throw 47 hours of legal
    # training data away from every short-horizon row, in every fold.
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
    """The shipped pooled model, inside whichever process takes it.

    The whole history -- the folds have already given their honest number --
    subsampled at the same rate the evaluated fits were, so what ships is the
    thing that was measured.

    `config.configure` and the feature switch first, for the reason spelled out
    on `_pooled_fold`.

    This is the ONE place a fitted Pipeline crosses the process boundary, and
    the exception is bought with a measurement: it is the largest single fit in
    the run, and fitting it in the parent after the folds left every worker
    idle for the duration. Pickling it back costs a few hundred kilobytes.
    """
    if settings is not None:
        config.configure(settings)
    features.SHIPPED_EXTRAS = extras
    return fit_pooled(_pooled_estimator(), subsample(frame, seed=seed))


def train_pooled(path: Path | pd.DataFrame, windows: list, horizons=None,
                 n_jobs: int | None = None,
                 phases: Phases | None = None) -> tuple[Pipeline, pd.DataFrame, int]:
    """One model over every horizon, `horizon_h` a feature.

    The folds are farmed out rather than the horizons -- there is only one
    horizon loop left, and the fold-fits are independent. `worker_count`
    still leaves Home Assistant a core, for the reason it always did.

    The final refit rides in the SAME fan-out, first in the queue: it is the
    biggest fit here and the folds are the only work available to hide it
    behind. The folds follow it longest-first, because the windows expand --
    the last fold trains on nearly the whole history and the first on 45 days
    of it, so submitting them in order strands the most expensive task alone in
    the final wave.

    Takes either the parquet path or an already-read wide frame; the caller has
    usually read it once already and the table is a quarter of a gigabyte.
    """
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

    # Back into fold order, so the concatenated frame is what it was before the
    # queue was reordered. `_scores_by_fold` groups on the `fold` column and
    # would not notice, but `per_fold` lists read positionally elsewhere would.
    collected.sort(key=lambda part: int(part["fold"].iloc[0]))
    return estimator, pd.concat(collected, ignore_index=True), len(frame)


DEDICATED_NAME = "occupancy_{horizon}h.pkl"
POOLED_NAME = "occupancy_pooled.pkl"


def save(estimator: Pipeline, metrics, models_dir: Path = MODELS_DIR,
         name: str = POOLED_NAME, feature_names: list[str] | None = None,
         kind: str = "pooled") -> Path:
    """Persist a model and the verdicts that go with it.

    Written via a temp file and an atomic rename: /predict may be reading these
    concurrently, and a half-written pickle would fail to unpickle rather than
    merely being stale.

    `kind` travels in the artifact because the two families are served
    differently -- a dedicated model reads a wide row for one horizon, a pooled
    one reads a melted row carrying `horizon_h` -- and `predict` has to know
    which without guessing from the filename.
    """
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
    """How much of the box training may take.

    Cores less one, so that Home Assistant keeps a core while this runs. The
    add-on is advisory and its training is a background job on a box whose day
    job is the house; taking every core for the duration is how it stops being
    unnoticed. On a single-core machine that floor of 1 means serial, which is
    the correct answer there.

    PROCESSES, not threads, and the reason is measured: one small fit takes 9.1s
    pinned to a single OpenMP thread and 6.8s given six, so scikit-learn's own
    parallelism buys 1.34x out of a possible 6 -- the models are too small for
    it. Six processes buy most of six. joblib pins each worker to one thread so
    the two cannot multiply into 36 threads fighting over 6 cores.
    """
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
    """The verdicts, and what the run cost.

    `phases` is the breakdown inside `train_all` -- seconds per named stretch,
    from `Phases`. It sits beside `duration_s` rather than replacing it: the
    total is what a person waiting on the button experiences and spans work
    either side of `train_all`, the breakdown is what says which part to attack.
    """
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
    """Record how long the whole train took, after the fact.

    Separate from `write_summary` because the number is not knowable when that
    runs: fitting the models is only part of it, and the feature table and the
    ETA models are built either side. The caller times the lot and stamps it
    here, so `metrics.json` stays the one file that says when a train happened
    and what it cost.
    """
    path = summary_path(models_dir)
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    summary["duration_s"] = round(seconds, 1)
    path.write_text(json.dumps(summary, indent=2))


def last_summary(models_dir: Path = MODELS_DIR) -> dict | None:
    """When the models on disk were trained, and what it cost.

    Read at startup because the answer outlives the process. `last_train` used to
    be in-memory only, so every restart said the add-on had never trained -- on
    an installation whose models were sitting right there, with the timestamp
    inside them. A corrupt or half-written file is treated as no answer rather
    than as a reason not to start.
    """
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
    """One dedicated horizon, start to finish, inside whichever process takes it.

    `config.configure` first, for the reason spelled out on `_pooled_fold`. The
    saving happens here rather than in the parent so a fitted Pipeline never
    crosses the process boundary; only the out-of-fold runs travel back, and
    each horizon writes its own filename through an atomic rename.
    """
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
    """Score both families at every horizon and let `choose` pick.

    Its own function because it is the serial tail of `train_all` and wants to
    be timeable as one phase without wrapping a fifteen-line loop body in a
    `with`.
    """
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
    """Train both families, then let the gate pick a winner per horizon.

    Both are cut on the SAME windows and scored against the SAME baseline
    ladder, which is what makes `choose` a comparison. See its docstring for why
    the crossover is measured rather than hardcoded.
    """
    phases = Phases()
    with phases("read"):
        wide = read_wide(path)
        windows, geometry = shared_windows(wide)
    settings = config.SETTINGS
    extras = features.SHIPPED_EXTRAS

    # The ladder once per horizon, shared by both candidates -- it was being run
    # by each path independently -- and fanned out, because it was then the
    # single largest line in the run: MEASURED at 112.6 s of a 237.6 s train,
    # serial in the parent with every worker idle. Each task carries only
    # `baseline.columns_for`, so what crosses the process boundary is five
    # columns rather than the thousand-column table, 48 times over.
    #
    # It shares ONE fan-out with the dedicated family, because nothing needs a
    # ladder until the gate runs and two blocks meant two barriers: every worker
    # waiting on the slowest ladder before the first dedicated fit could start,
    # and again at the end. The dedicated tasks go first -- they are the longer
    # ones, and a queue that ends on its longest task strands it alone in the
    # final wave.
    workers = worker_count() if n_jobs is None else n_jobs
    with phases("ladder+dedicated"):
        answers = Parallel(n_jobs=workers, backend="loky")([
            *(delayed(_dedicated_and_save)(path, h, windows, models_dir,
                                           settings, extras)
              for h in horizons),
            *(delayed(_one_ladder)(wide[baseline.columns_for(h)], h, geometry,
                                   windows, settings)
              for h in horizons),
        ])
    results, ladders = answers[:len(horizons)], answers[len(horizons):]
    rungs = {h: r for h, r, _ in ladders}
    dedicated = {h: (scored, n_train) for h, scored, n_train, err, _ in results
                 if err is None}
    failed = {f"{h}h dedicated": err for h, _, _, err, _ in results
              if err is not None}
    # Worker-seconds, not wall clock: the two share a fan-out, so their wall
    # time is one number and cannot say which of them to attack next. Each task
    # reports its own, and these sum across workers -- a phase totalling five
    # times its own wall time is one that saturated the pool.
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

    # EVERY horizon's verdict is written to every artifact that could serve it,
    # not just the winner's. `predict.load_models` reads `ships` and `kind` per
    # horizon out of whichever file it opens first, and `server._status` reads
    # `best_baseline` the same way -- so a horizon whose verdict is missing
    # from one file would be reported differently depending on which artifact
    # happened to load, which is a silent inconsistency rather than an error.
    #
    # The workers wrote the dedicated pickles with empty metrics before the gate
    # had spoken; they are rewritten here.
    with phases("write"):
        if pooled_scored is not None:
            save(estimator, chosen, models_dir, POOLED_NAME,
                 base_features(), kind="pooled")
        for horizon, winner in chosen.items():
            name = DEDICATED_NAME.format(horizon=horizon)
            if not (models_dir / name).exists():
                continue
            with (models_dir / name).open("rb") as fh:
                artifact = pickle.load(fh)
            save(artifact["model"], {horizon: winner}, models_dir, name,
                 artifact["features"], kind="dedicated")

    _log.info("train_all: %s", phases.line())
    write_summary(summary, models_dir, failed, phases=phases.as_dict())

    # Close the worker pool rather than leaving it parked for reuse. Left
    # running, loky's resource tracker prints a wall of "leaked semlock
    # objects" warnings at every container stop -- roughly ten lines per
    # restart, in a log whose whole problem is that the signal is thin. The
    # cost is one pool start-up on the next train, which is a weekly job.
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
