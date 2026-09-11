"""When does this person leave the house, and will they leave at all today?

Splits what the occupancy curve smears: A, will they leave today; B, the hour,
given they leave. Labels come from the feature parquet, which has blanked any
recorder outage, so such a day is dropped, never labelled "did not leave".
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import baseline, config, evaluate, features, log

_log = log.get(__name__)

# 04:00 local, the same instant for training and serving: the label window lies
# wholly after it, and asleep at home the origin state cannot leak the answer.
# The cost is an estimate that is static across the day.
ORIGIN_HOUR = 4

# Bounded like `eta.MAX_LEAD_MIN`: unbounded, "first sustained absence" becomes
# a fact about shop hours, so `left_today` means LEFT BEFORE 22:00 local.
LAST_DEPARTURE_HOUR = 22

# The line `config.CROSSING_MIN_HOURS` draws: an absence under about an hour has
# no representation in a 30-minute-slot target.
MIN_AWAY_SLOTS = 2

# A mostly-unobserved day scored as "did not leave" teaches the opposite of the
# truth, so such a day is dropped, never labelled False.
MIN_OBSERVED_SLOTS = 36

# And the observed slots have to cover the part of the day the label lives in.
# A three-hour hole at 07:00 is exactly where a departure would be.
MAX_MORNING_GAP_SLOTS = 2


def origin_slot() -> int:
    return ORIGIN_HOUR * features.slots_per_hour()


def last_departure_slot() -> int:
    return LAST_DEPARTURE_HOUR * features.slots_per_hour()


def label_embargo() -> pd.Timedelta:
    """How far a fold stands back from the day it scores: a day's outcome is not
    settled until 18 h after its own origin. Derived from the two constants, so
    moving either moves this.
    """
    return pd.Timedelta(hours=LAST_DEPARTURE_HOUR - ORIGIN_HOUR) + pd.Timedelta(days=1)


def day_grid(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, local date), one column per slot of that day.

    `home_frac`, not its alias `state_now`: what happened, not what was fed.
    """
    local = table["time"].dt.tz_convert(config.tzinfo())
    keyed = table[["subject", "home_frac"]].assign(
        _date=local.dt.date,
        _slot=features.slot_of_day(local),
    )
    # `mean`, not `first`: on the autumn transition day two UTC slots land on
    # one local slot, and `first` threw the second away.
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
    """`(subject, date) -> candidate, left_today, departure_hour`. A day is a
    CANDIDATE only where the question has an answer; one that fails any test is
    dropped, never recorded as "did not leave".
    """
    origin, cap = origin_slot(), last_departure_slot()
    threshold = evaluate.HOME_THRESHOLD
    rows = []
    for (subject, date), day in day_grid(table).iterrows():
        values = day.to_numpy(dtype=float)
        observed = int(np.count_nonzero(~np.isnan(values)))
        at_origin = values[origin]

        # Home when asked: someone already away has answered the question.
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


# Earlier same-weekdays before that weekday's history counts; below it the
# lookup is NaN, which HistGradientBoosting reads natively.
MIN_WEEKDAY_SAMPLES = 3

# The trailing window for "lately": four of each weekday, and quicker than the
# expanding view to follow a change of routine.
RECENT_DAYS = 28


def _causal(days: pd.DataFrame) -> pd.DataFrame:
    """Each person's own history, EXPANDING and SHIFTED to that morning: a
    whole-frame `groupby(dow).median()` has read the day it predicts, and leaks
    twice, because that median is both a feature and the baseline B must beat.
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

    # Also over a trailing window, on a TIME index: days are dropped when nobody
    # watched them, so a fixed row count would reach further back when gappy.
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
    """The OTHER people's weekday hours as of the same morning, from their own
    causal columns so it inherits their causality; their TODAY is never read.
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
    """The hour the model is a CORRECTION to: the causal weekday median, then
    the trailing window, then the person's own median. A ladder, because the top
    rung is NaN until a weekday has been seen three times.
    """
    return (days["wday_hour"]
            .fillna(days["recent_hour"])
            .fillna(days["all_hour"]))


def feature_columns() -> list[str]:
    """What both halves read. Derived, never spelled at a use site."""
    return [
        # A PLAIN INTEGER: `dow == 3` is one split, where isolating Thursday
        # from the sin/cos circle is a conjunction of four.
        "dow", "is_weekend", "is_holiday",
        "wday_rate", "wday_hour", "wday_n",
        "recent_rate", "recent_hour",
        *(f"other_{slug}_wday_hour" for slug in config.all_slugs()),
    ]


def feature_frame(days: pd.DataFrame) -> pd.DataFrame:
    """Labelled days plus features, shared by training and serving, or the model
    is served a number it was never fitted on. `state_now` at the origin is not
    a feature: candidacy makes it constant by construction.
    """
    days = _partner(_causal(days))
    days["baseline_hour"] = anchor_hour(days)
    days["is_weekend"] = features.is_weekend(days["dow"])
    days["is_holiday"] = features.holiday_flags(pd.DatetimeIndex(days["date"]))
    return days


# ---------------------------------------------------------------------------
# Everything below is the model half, deliberately NOT wired in: it did not
# clear the ship bar, and the day-level gate fires on random labels at these
# sample sizes. `outing.py`'s plain arithmetic ships instead.

# Below this a fit has fewer same-weekdays than `MIN_WEEKDAY_SAMPLES` needs.
MIN_TRAIN_DAYS = 60

# 15 rather than the occupancy family's 5: the geometry here is looser, so a
# majority of buckets is weaker evidence.
MIN_SKILL_PCT_A = 15.0
MIN_SKILL_PCT_B = 15.0

# Not a second cut of the data: the predictions are made once and then grouped.
BUCKET_DAYS = 14


def _estimator_a():
    """Will they leave today.

    Deliberately tiny: the independent unit is ~150 person-days.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=15, l2_regularization=1.0, random_state=0)


def _estimator_b():
    """How late, given they leave. `absolute_error` fits the conditional MEDIAN:
    per weekday the target is bimodal, and squared error would fit a mean at an
    hour nobody leaves.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=150, learning_rate=0.05,
        max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1.0,
        random_state=0)


def _fit_rate_shrink(train: pd.DataFrame) -> tuple[float, float]:
    """How far to pull the weekday rate toward that person's overall rate. Same
    argument as `baseline._fit_shrink`, which is coupled to the wide table's
    columns, so only `baseline.shrink` is reused.
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
    """The causal weekday median, then lately, then that person's own median: a
    ladder, because a model has to be compared against something on days the
    top rung is NaN.
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
    """Predict each day from a model that has seen only earlier ones: one fit
    per DAY, as ~150 rows per person leave `evaluate.calendar_folds` no folds.
    Early days get 60 days of history and late ones 150, as on a live install.
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

            # A CORRECTION to the lookup, not a rival: fitted on the residual
            # from `baseline_hour`, so one that learns nothing reproduces it.
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
    """Calendar buckets over the prequential predictions, grouped AFTER they are
    made, so no bucket width could be chosen once the answer is known.
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
    # `fold_record_allows` rather than a strict majority: refuse a record PROVEN
    # worse than a coin flip, and let the effect size decide the rest.
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
