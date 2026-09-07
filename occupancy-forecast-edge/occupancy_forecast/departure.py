"""When does this person leave the house, and will they leave at all today?

**A different question from the occupancy curve, and a different shape.** The
forecaster in `train.py` answers "what is P(home) at t+h" for 48 horizons, and it
gets the LEVEL right and the TIMING wrong: a person who leaves for work between
07:00 and 08:00 every Thursday is forecast at 0.93 at 07:14 and only reaches her
true level around 11:00. Three feature-level fixes were built and measured and
all three nulled -- see the tables recorded in `features.py`.

Smearing is that model's honest answer. One number per hour has to cover "leaves
at 7", "leaves at 8" and "doesn't leave", and the average of those is a slope.
This module splits the question instead:

    A. will this person leave the house today?      -- a day-level probability
    B. how late will they leave, GIVEN they leave?  -- a time of day

Two answers imply a curve, `P(home at t) = 1 - P(leave) * P(dep <= t)`, whose
sharpness is the departure-time spread rather than an artifact of averaging. On
this household that spread is measured at 1.24 h for one person's Thursdays and
3.60 h for her Wednesdays -- so the useful property is not a uniformly sharper
answer but one that is CONFIDENT WHEN THE ROUTINE IS REGULAR AND VAGUE WHEN IT
IS NOT, which a single curve cannot express at all.

**`eta.py` is the template**: per-person, conditional, published as its own
sensors and paired with the occupancy probability by whoever consumes it, rather
than blended into the curve. Blending was considered and rejected for now --
`predict._crossing` is deliberately blind to provenance, so a reconstructed band
sitting at a different level would create a spurious crossing that its
`min_hours` guard cannot defend against.

**Labels come from the feature parquet, not the raw store.** That is the one
place this departs from `eta.py`, which reads the proximity trace directly
because quantising a twelve-minute drive to half an hour destroys it. The answer
here is an hour of day and the bar is a 2.60 h MAE, so a 30-minute grid is eight
times finer than it needs to be -- while `features.build` has already done the
dangerous part: `observability` blanks slots inside a silence longer than
`config.MAX_SILENCE_H`, the mask that exists BECAUSE an early build reported
three straight weeks of `home_frac == 1.00` across a 653-hour recorder outage.
Calling that outage "did not leave today" would be the same bug in a new costume.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import baseline, config, evaluate, features, log

_log = log.get(__name__)

# When the question is asked. Local, and the same instant for training and
# serving, which is what keeps `left_today` free of cross-midnight ambiguity:
# the whole label window lies after the origin, so at no training origin is the
# question already answered. It is also before every departure observed in this
# household, and at 04:00 people are asleep at home -- so the origin state is
# near-constant and cannot smuggle the answer in.
#
# The cost is that the estimate is STATIC across the day: computed on the first
# cycle after 04:00 and republished until the next one. A version that sharpens
# through the morning is the obvious follow-on and is deliberately not this,
# because adding an origin axis before the single-origin version has cleared a
# gate is building on nothing.
ORIGIN_HOUR = 4

# The MAX_LEAD_MIN analogue, and it is load-bearing for the same reason.
#
# `eta.py` records that without a bounded label window its model learned "at work
# => home in four hours", a calendar fact wearing a travel time's clothes.
# Unbounded here, "first sustained absence of the day" for a mostly-home person
# becomes whatever eventually happens: a Sunday evening walk to the shop teaches
# "Sunday => leaves at 19:00", which is a fact about shop opening hours. So
# `left_today` means LEFT BEFORE 22:00 LOCAL, and the sensor says so.
LAST_DEPARTURE_HOUR = 22

# How long an absence has to last before it is a departure rather than a blip.
#
# Not an arbitrary dwell threshold: it is the line `config.CROSSING_MIN_HOURS`
# already draws for the published "hours until away", for the reason recorded on
# `predict._crossing` -- the target is a fraction of a 30-minute slot, so an
# absence shorter than about an hour has no representation in the data at all.
MIN_AWAY_SLOTS = 2

# A day has to be watched before "they did not leave" means anything.
#
# MEASURED on this archive: 18% of person-days are under half observed, and 99
# contain a slot with zero coverage. A mostly-unobserved day scored as "did not
# leave" is a false negative that teaches the model the person stays home, so
# such a day is NOT A CANDIDATE -- it is dropped, never labelled False.
MIN_OBSERVED_SLOTS = 36

# And the observed slots have to cover the part of the day the label lives in.
# A three-hour hole at 07:00 is exactly where a departure would be.
MAX_MORNING_GAP_SLOTS = 2


def origin_slot() -> int:
    return ORIGIN_HOUR * features.slots_per_hour()


def last_departure_slot() -> int:
    return LAST_DEPARTURE_HOUR * features.slots_per_hour()


def label_embargo() -> pd.Timedelta:
    """How far a fold has to stand back from the day it scores.

    The label spans from the 04:00 origin to the 22:00 cap, so a day's outcome
    is not settled until 18 hours after its own origin. DERIVED from the two
    constants rather than hand-set, so that moving either moves this -- the
    discipline `predict.LOOKBACK_DAYS` follows for the same class of mistake.
    """
    return pd.Timedelta(hours=LAST_DEPARTURE_HOUR - ORIGIN_HOUR) + pd.Timedelta(days=1)


def day_grid(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, local date), one column per slot of that day.

    `home_frac` is the observation: the fraction of the slot spent at home, NaN
    where `features.observability` blanked it. `state_now` is its origin-side
    alias and is identical in the parquet; this uses the target name because
    what is wanted here is what happened, not what a model was fed.
    """
    local = table["time"].dt.tz_convert(config.tzinfo())
    keyed = table[["subject", "home_frac"]].assign(
        _date=local.dt.date,
        _slot=features.slot_of_day(local),
    )
    # `mean`, not `first`: on the autumn transition day two UTC slots land on
    # one local slot, and `first` silently threw the second observation away.
    # The mean of two real observations of the same wall-clock half hour is a
    # fair value for it; NaN is skipped, so one blanked slot does not blank both.
    grid = keyed.pivot_table(index=["subject", "_date"], columns="_slot",
                             values="home_frac", aggfunc="mean")
    return grid.reindex(columns=range(config.SLOTS_PER_DAY))


def _longest_nan_run(values: np.ndarray) -> int:
    longest = run = 0
    for value in values:
        run = run + 1 if np.isnan(value) else 0
        longest = max(longest, run)
    return longest


def label_days(table: pd.DataFrame) -> pd.DataFrame:
    """`(subject, date) -> candidate, left_today, departure_hour`.

    A day is a CANDIDATE only where "will they leave today" is a question with an
    answer: they were home when it was asked, the day was watched, and the part
    of it the label lives in has no hole big enough to hide a departure. A day
    that fails any of those is dropped -- never recorded as "did not leave",
    which is the failure mode that would teach the model the opposite of the
    truth.
    """
    origin, cap = origin_slot(), last_departure_slot()
    threshold = evaluate.HOME_THRESHOLD
    rows = []
    for (subject, date), day in day_grid(table).iterrows():
        values = day.to_numpy(dtype=float)
        observed = int(np.count_nonzero(~np.isnan(values)))
        at_origin = values[origin]

        # Home when asked. A person already away had the question answered
        # before it was put, and `current_row` refuses the same state at serving
        # so the two sides cannot drift -- one rule, asserted in one test.
        home_at_origin = (not np.isnan(at_origin)) and at_origin >= threshold
        watched = observed >= MIN_OBSERVED_SLOTS
        gap = _longest_nan_run(values[origin:cap + 1])
        candidate = bool(home_at_origin and watched and gap <= MAX_MORNING_GAP_SLOTS)

        departure = None
        if candidate:
            for d in range(origin + 1, cap + 1):
                window = values[d:d + MIN_AWAY_SLOTS]
                if len(window) < MIN_AWAY_SLOTS or np.isnan(values[d - 1]):
                    continue
                if np.isnan(window).any():
                    continue                     # a hole breaks the away run
                if values[d - 1] >= threshold and (window < threshold).all():
                    departure = d
                    break

        rows.append({
            "subject": subject,
            "date": pd.Timestamp(date),
            "dow": pd.Timestamp(date).dayofweek,
            "observed_slots": observed,
            "candidate": candidate,
            "left_today": bool(candidate and departure is not None),
            "departure_slot": departure,
            "departure_hour": (None if departure is None
                               else departure / features.slots_per_hour()),
        })
    return pd.DataFrame(rows).sort_values(["subject", "date"]).reset_index(drop=True)


# How many earlier same-weekdays before that weekday's history means anything.
# Below this the lookup is NaN, which HistGradientBoosting reads natively --
# better than a number backed by one observation.
MIN_WEEKDAY_SAMPLES = 3

# The trailing window for "lately", beside the per-weekday view. Four weeks, so
# it holds four of each weekday and moves with a change of routine faster than
# the expanding view does.
RECENT_DAYS = 28


def _causal(days: pd.DataFrame) -> pd.DataFrame:
    """That person's own history, as of the morning of each row. Never later.

    **The leak this guards is the tempting one.** A `groupby(dow).median()` over
    the whole frame gives a beautiful number and is fiction: it has read the day
    it is predicting. Everything here is EXPANDING and SHIFTED -- each row sees
    strictly earlier days of the same person, and the shift is what makes
    "strictly" true.

    It leaks twice if it leaks once, because the weekday median is both a
    feature here and the baseline model B has to beat.
    """
    days = days.sort_values(["subject", "date"]).copy()
    # Departure hour only on days there was one; expanding().median() skips the
    # NaNs, so this is "the median of the departures, not of the days".
    hour = days["departure_hour"].where(days["left_today"])

    by_weekday = days.groupby(["subject", "dow"], sort=False)
    days["wday_rate"] = by_weekday["left_today"].transform(
        lambda s: s.shift().expanding().mean())
    days["wday_n"] = by_weekday["left_today"].transform(
        lambda s: s.shift().expanding().count())
    days["wday_hour"] = hour.groupby(
        [days["subject"], days["dow"]], sort=False).transform(
        lambda s: s.shift().expanding().median())
    thin = days["wday_n"] < MIN_WEEKDAY_SAMPLES
    days.loc[thin, ["wday_rate", "wday_hour"]] = np.nan

    # And the same two over a trailing window rather than all of history, on a
    # TIME index -- days are dropped when nobody watched them, so a fixed row
    # count would silently reach further back on a gappy stretch.
    recent_rate, recent_hour = [], []
    for _subject, part in days.groupby("subject", sort=False):
        indexed = part.set_index("date")
        window = f"{RECENT_DAYS}D"
        recent_rate.append(
            indexed["left_today"].shift().rolling(window, min_periods=3).mean())
        recent_hour.append(
            indexed["departure_hour"].where(indexed["left_today"])
            .shift().rolling(window, min_periods=3).median())
    days["recent_rate"] = pd.concat(recent_rate).to_numpy()
    days["recent_hour"] = pd.concat(recent_hour).to_numpy()

    # And the whole of that person's past, which is what the ladder falls back
    # to on a weekday it has not seen three times yet.
    by_subject = days.groupby("subject", sort=False)
    days["all_rate"] = by_subject["left_today"].transform(
        lambda s: s.shift().expanding().mean())
    days["all_hour"] = hour.groupby(days["subject"], sort=False).transform(
        lambda s: s.shift().expanding().median())
    return days


def _partner(days: pd.DataFrame) -> pd.DataFrame:
    """What the OTHER people's weekdays look like, as of the same morning.

    In a couple a departure is not an independent event -- one person leaving is
    evidence about the other. Built from each partner's own causal columns, so
    it inherits their causality rather than needing its own argument. Their
    TODAY is never read: that would be the answer arriving by a side door.
    """
    out = days.copy()
    for slug in config.all_slugs():
        column = f"other_{slug}_wday_hour"
        theirs = days.loc[days["subject"] == slug, ["date", "wday_hour"]]
        merged = out[["date"]].merge(theirs, on="date", how="left")
        out[column] = merged["wday_hour"].to_numpy()
        # A subject never mirrors itself -- that column is `wday_hour`, and two
        # names for one number is how a tree gets talked into splitting twice.
        out.loc[out["subject"] == slug, column] = np.nan
    return out


def anchor_hour(days: pd.DataFrame) -> pd.Series:
    """The hour the model is a CORRECTION to: the causal weekday median.

    A ladder, because the top rung is NaN until a weekday has been seen three
    times: weekday median, then the trailing window, then that person's own
    median over everything. NaN only when they have no history at all.
    """
    return (days["wday_hour"]
            .fillna(days["recent_hour"])
            .fillna(days["all_hour"]))


def feature_columns() -> list[str]:
    """What both halves read. Derived, never spelled at a use site."""
    return [
        # The weekday as a PLAIN INTEGER. The sin/cos pair is right about the
        # wrap and wrong about the edge, and here the weekday is the whole
        # signal: `dow == 3` is one split, isolating Thursday from a circle is a
        # conjunction of four.
        "dow", "is_weekend", "is_holiday",
        "wday_rate", "wday_hour", "wday_n",
        "recent_rate", "recent_hour",
        *(f"other_{slug}_wday_hour" for slug in config.all_slugs()),
    ]


def feature_frame(days: pd.DataFrame) -> pd.DataFrame:
    """Labelled candidate days plus the features, one row per person-day.

    Shared by training and serving on purpose -- `eta.py` makes the same point
    about `feature_frame`, and for the same reason: if the two computed the
    weekday median even slightly differently the model would be served a number
    it was never fitted on, and nothing would say so.

    `state_now` at the origin is deliberately NOT a feature. Candidacy already
    requires being home then, so on every row it lands in [0.5, 1.0] -- a column
    that is constant by construction, which a tree cannot use and a reader would
    have to think about.
    """
    days = _partner(_causal(days))
    days["baseline_hour"] = anchor_hour(days)
    days["is_weekend"] = features.is_weekend(days["dow"])
    days["is_holiday"] = features.holiday_flags(pd.DatetimeIndex(days["date"]))
    return days


# ---------------------------------------------------------------------------
# The two halves -- BUILT, MEASURED, NOT SHIPPED. Kept on purpose.
#
# Everything below this line is the model half of the departure question:
# estimator A ("will they leave today"), estimator B ("how late, given they
# leave"), the prequential harness and the ship gate for both. None of it is
# wired into the add-on and none of it is imported anywhere else. It stays
# because it is the record of what was tried, and because the prequential
# harness is the reusable piece of it.
#
# What it measured, on the real archive and on the synthetic household
# (`tests/synthetic.py`), 2026-09-02/03:
#
#   * A as a correction on a causal weekday-median lookup: -3.5% to -2.3% in
#     the control world (the lookup is provably optimal there, so losing is the
#     leak check passing) and +3.1% / +4.2% / +3.6% at 180 / 365 / 730 days in
#     the realistic world. A plateau at 3-4% against a 15% ship bar; more
#     history does not rescue it.
#   * A lambda sweep on the correction went monotonically to +0.0% at
#     lambda=0 with within-1h best there too: the correction is noise. The
#     failure is the FEATURE SET -- every feature is derived from the same
#     presence history that builds the baseline it must beat -- not the model
#     class, and no estimator or loss fixes that.
#   * The day-level ship gate itself is not safe at these sample sizes: a
#     permutation null fired it on 4-12% of RANDOM label sets.
#
# What shipped instead is `outing.py`: the per-weekday rate and median as plain
# arithmetic, which scores +29%/+32% on "is today an office day" and 0.45-0.54 h
# MAE on the hour. If this half is ever revisited, the one lead the notes leave
# open is a signal that knows about TOMORROW rather than typical weekdays --
# `next_alarm_h` -- which had no history to price when this was measured.
# ---------------------------------------------------------------------------

# Days of history before the first prediction is attempted. Below this a fit has
# fewer same-weekdays than `MIN_WEEKDAY_SAMPLES` needs and is answering from
# almost nothing.
MIN_TRAIN_DAYS = 60

# Both gates, and 15 rather than the occupancy family's 5 for eta.py's stated
# reason: the geometry here is looser, so a majority of buckets is weaker
# evidence and the gate has to demand an effect size instead.
MIN_SKILL_PCT_A = 15.0
MIN_SKILL_PCT_B = 15.0

# The bucket width for the fold record. Not a second cut of the data -- the
# prequential predictions are made once and then grouped, so there is no fold
# width to choose after seeing the answer.
BUCKET_DAYS = 14


def _estimator_a():
    """Will they leave today.

    Deliberately tiny: the independent unit is ~150 person-days, not 150 rows of
    something. NaN is read natively, which matters because the weekday lookup is
    NaN until three same-weekdays exist.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=15, l2_regularization=1.0, random_state=0)


def _estimator_b():
    """How late, given they leave.

    **`absolute_error`, and that is load-bearing.** One person's Thursdays are a
    07:00 departure and her Wednesdays a 17:00 one, so per weekday the target is
    bimodal. Squared error fits the conditional MEAN, which for that mixture sits
    near midday -- an hour she never leaves. Absolute error fits the conditional
    median, which is the right point estimate for a multimodal target scored by
    MAE and is the same quantity the baseline reports.

    The same shape of argument as `eta.py` fitting log-minutes rather than
    minutes: choose the loss the question is asked in.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=150, learning_rate=0.05,
        max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1.0,
        random_state=0)


def _fit_rate_shrink(train: pd.DataFrame) -> tuple[float, float]:
    """How far to pull the weekday rate toward that person's overall rate.

    The weekday rate is a mean of at most a couple of dozen days and at worst
    three, so untouched it says 0.0 and 1.0 and means "three days agreed". Same
    argument as `baseline._fit_shrink`, and it reuses `baseline.shrink` for the
    clip -- but not that function itself, which is coupled to the wide table's
    `y_{h}h` and `state_now` columns and cannot be called from here.
    """
    base = float(train["left_today"].mean())
    raw = train["wday_rate"].to_numpy(dtype=float)
    truth = train["left_today"].to_numpy(dtype=float)
    keep = ~np.isnan(raw)
    if not keep.any():
        return 1.0, base
    losses = [float(np.mean((baseline.shrink(raw[keep], w, base) - truth[keep]) ** 2))
              for w in baseline.SHRINK_GRID]
    return baseline.SHRINK_GRID[int(np.argmin(losses))], base


def _baseline_hour(row: pd.Series, fallback: float) -> float:
    """The causal weekday median, then lately, then that person's own median.

    A ladder rather than one number, because the top rung is NaN until three
    same-weekdays exist and a model has to be compared against something on
    those days too.
    """
    for column in ("wday_hour", "recent_hour", "all_hour"):
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return float(value)
    return fallback


@dataclass
class DepartureMetrics:
    """What each half scored, and what serving falls back to if it did not ship."""

    subject: str
    n_days: int
    n_candidates: int
    n_departures: int
    n_scored: int
    n_buckets: int
    # A -- will they leave today
    base_rate: float
    brier: float
    baseline_brier: float
    skill_a_pct: float
    buckets_beating_a: int
    sign_test_p_a: float
    ships_will_leave: bool
    # B -- how late, given they leave
    mae_h: float
    median_ae_h: float
    within_1h_pct: float
    over_3h_pct: float
    median_signed_error_h: float
    baseline_mae_h: float
    skill_b_pct: float
    buckets_beating_b: int
    sign_test_p_b: float
    ships_departure_hour: bool
    # Served fallbacks, carried so what is served is exactly what was scored.
    shrink_weight: float = 1.0
    shrink_base: float = 0.5
    fallback_hour: float = float("nan")
    weekday_spread_h: dict = field(default_factory=dict)


def prequential(days: pd.DataFrame) -> pd.DataFrame:
    """Predict each day from a model that has only seen earlier ones.

    One fit per day rather than per fold, and that is the point. A day-level
    label gives ~150 rows per person, so `evaluate.calendar_folds` returns
    NOTHING -- its `MIN_TEST_ROWS` floor of 200 is the wall `eta.py` hit and
    answered with looser windows. Looser windows do not rescue this one: at 14
    days a bucket holds at most 14 rows. Predicting one day at a time gives
    ~90-110 honestly out-of-sample predictions instead of a handful, at
    milliseconds a fit on a table this size.

    The embargo is `label_embargo()` -- a day's outcome is not settled until the
    22:00 cap, so a fit for day `d` may not see `d - 1` either.

    Early days are scored by a model with 60 days behind it and late ones by a
    model with 150. That is not a flaw, it is what a live install experiences.
    """
    embargo = label_embargo()
    out = []
    for subject, part in days.groupby("subject", sort=False):
        part = part[part["candidate"]].sort_values("date").reset_index(drop=True)
        columns = [c for c in feature_columns() if c in part.columns]
        for i, row in enumerate(part.itertuples()):
            train = part[part["date"] < row.date - embargo]
            if len(train) < MIN_TRAIN_DAYS:
                continue
            weight, base = _fit_rate_shrink(train)
            record = part.iloc[i]

            a = _estimator_a().fit(train[columns], train["left_today"].astype(int))
            p_leave = float(a.predict_proba(part.iloc[[i]][columns])[0, 1])

            fallback = float(train.loc[train["left_today"], "departure_hour"].median())

            # **The model is a CORRECTION to the lookup, not a rival to it.**
            #
            # Fitted on `departure_hour - baseline_hour`, so a model that learns
            # nothing predicts 0 and reproduces the lookup exactly -- it cannot
            # lose to it by much -- while anything it does learn is a genuine
            # correction. The first attempt fitted the hour directly with the
            # lookup among its inputs, which asked it to BEAT a number it had
            # been handed; on ~100 events its only move was to copy that number
            # and add variance, and it lost by 10-15%.
            #
            # Same argument `train.py` records for predicting the change from
            # `state_now` rather than the level, and it was measured there too.
            left = train[train["left_today"]]
            anchor_train = left["baseline_hour"].fillna(fallback)
            hour = np.nan
            if len(left) >= MIN_TRAIN_DAYS // 4:
                b = _estimator_b().fit(left[columns],
                                       left["departure_hour"] - anchor_train)
                anchor = record["baseline_hour"]
                anchor = fallback if pd.isna(anchor) else float(anchor)
                hour = anchor + float(b.predict(part.iloc[[i]][columns])[0])
            out.append({
                "subject": subject, "date": record["date"], "dow": record["dow"],
                "left_today": bool(record["left_today"]),
                "departure_hour": record["departure_hour"],
                "p_leave": p_leave,
                "p_leave_baseline": float(baseline.shrink(
                    np.array([record["wday_rate"]], dtype=float), weight, base)[0]),
                "hour": hour,
                "hour_baseline": _baseline_hour(record, fallback),
                "shrink_weight": weight, "shrink_base": base,
                "fallback_hour": fallback,
            })
    return pd.DataFrame(out)


def _buckets(scored: pd.DataFrame) -> pd.Series:
    """Calendar buckets over the prequential predictions.

    Grouped AFTER the predictions are made, so the pooled effect size and the
    per-bucket record come from one set of numbers. There is no second cut of
    the data and no bucket width that could be chosen once the answer is known.
    """
    first = scored["date"].min()
    return ((scored["date"] - first).dt.days // BUCKET_DAYS).rename("bucket")


def score_subject(subject: str, days: pd.DataFrame,
                  scored: pd.DataFrame) -> DepartureMetrics | None:
    """Both halves, gated independently."""
    mine = scored[scored["subject"] == subject]
    if mine.empty:
        return None
    mine = mine.assign(bucket=_buckets(mine))

    truth = mine["left_today"].astype(float).to_numpy()
    brier = float(np.mean((mine["p_leave"] - truth) ** 2))
    base_brier = float(np.mean((mine["p_leave_baseline"] - truth) ** 2))

    left = mine[mine["left_today"] & mine["hour"].notna()]
    err = (left["hour"] - left["departure_hour"]).to_numpy()
    base_err = (left["hour_baseline"] - left["departure_hour"]).to_numpy()
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
    base_mae = float(np.mean(np.abs(base_err))) if len(base_err) else float("nan")

    beat_a = beat_b = 0
    n_buckets_a = n_buckets_b = 0
    for _key, part in mine.groupby("bucket"):
        n_buckets_a += 1
        y = part["left_today"].astype(float).to_numpy()
        beat_a += np.mean((part["p_leave"] - y) ** 2) < np.mean(
            (part["p_leave_baseline"] - y) ** 2)
        hit = part[part["left_today"] & part["hour"].notna()]
        if hit.empty:
            continue
        n_buckets_b += 1
        beat_b += (np.mean(np.abs(hit["hour"] - hit["departure_hour"]))
                   < np.mean(np.abs(hit["hour_baseline"] - hit["departure_hour"])))

    skill_a = 100.0 * (1.0 - brier / base_brier) if base_brier else float("nan")
    skill_b = 100.0 * (1.0 - mae / base_mae) if base_mae else float("nan")
    # `fold_record_allows` rather than a strict majority, and for the reason it
    # records: refuse a record PROVEN worse than a coin flip, and let the effect
    # size decide the rest. A refused half serves its calibrated baseline, so
    # shipping too readily costs nothing worse than that baseline.
    from . import train as train_mod

    all_days = days[days["subject"] == subject]
    candidates = all_days[all_days["candidate"]]
    spread = (left.assign(ae=np.abs(err)).groupby("dow")["ae"].median().to_dict()
              if len(left) else {})

    return DepartureMetrics(
        subject=subject,
        n_days=len(all_days), n_candidates=len(candidates),
        n_departures=int(candidates["left_today"].sum()),
        n_scored=len(mine), n_buckets=n_buckets_a,
        base_rate=float(mine["left_today"].mean()),
        brier=brier, baseline_brier=base_brier, skill_a_pct=skill_a,
        buckets_beating_a=int(beat_a),
        sign_test_p_a=evaluate.sign_test(int(beat_a), n_buckets_a),
        ships_will_leave=bool(
            brier < base_brier
            and train_mod.fold_record_allows(int(beat_a), n_buckets_a)
            and skill_a >= MIN_SKILL_PCT_A),
        mae_h=mae,
        median_ae_h=float(np.median(np.abs(err))) if len(err) else float("nan"),
        within_1h_pct=float(np.mean(np.abs(err) <= 1) * 100) if len(err) else float("nan"),
        over_3h_pct=float(np.mean(np.abs(err) >= 3) * 100) if len(err) else float("nan"),
        median_signed_error_h=float(np.median(err)) if len(err) else float("nan"),
        baseline_mae_h=base_mae, skill_b_pct=skill_b,
        buckets_beating_b=int(beat_b),
        sign_test_p_b=evaluate.sign_test(int(beat_b), n_buckets_b),
        ships_departure_hour=bool(
            mae < base_mae
            and train_mod.fold_record_allows(int(beat_b), n_buckets_b)
            and skill_b >= MIN_SKILL_PCT_B),
        shrink_weight=float(mine["shrink_weight"].iloc[-1]),
        shrink_base=float(mine["shrink_base"].iloc[-1]),
        fallback_hour=float(mine["fallback_hour"].iloc[-1]),
        weekday_spread_h={int(k): float(v) for k, v in spread.items()},
    )
