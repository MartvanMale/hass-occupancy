"""Build the modelling table: one row per (subject, 30-minute slot).

The target is `home_frac`, the time-weighted fraction of the slot spent at home.
Last-observation-carried-forward turns a ninety-second GPS blip at a zone edge
into a spurious empty-house slot; time-weighting debounces by construction and
needs no dwell threshold. Slots under `config.MIN_SLOT_COVERAGE` are NaN, never
a guess.
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, log

_log = log.get(__name__)

# ---------------------------------------------------------------------------
# Feature groups. Named so a drop/only probe can price them as a unit.
# ---------------------------------------------------------------------------

# Daily lag offsets, in days, of the TARGET slot. 3 and 21 exist for the long
# horizons: past +25 h `safe_daily_lags` drops the 1-day lag and these keep five
# legal anchors. Both fit inside the 45-day warm-up, so they cost no folds.
DAILY_LAGS = (1, 2, 3, 7, 14, 21)

# Four trailing same-weekdays -- thin, but `evaluate.MIN_TRAIN_DAYS` is 45
# BECAUSE this is the longest in-table feature, so widening it buys noise with folds.
CLIMATOLOGY_WEEKS = 4

# A wider candidate, not a default: it costs no folds but doubles
# `predict.LOOKBACK_DAYS`, and every serving cycle is a full lookback rebuild.
WIDE_CLIMATOLOGY_WEEKS = 8

# Every width the table carries. `climatology_column(h)` still means the served
# one; the rest are candidates.
CLIMATOLOGY_WIDTHS = (CLIMATOLOGY_WEEKS, WIDE_CLIMATOLOGY_WEEKS)

# Which width the transition slopes are differenced from. The narrow one
# deliberately, so the slope arm is one change from the control rather than two.
TRANSITION_SOURCE_WEEKS = CLIMATOLOGY_WEEKS

# The climatology pooled over ALL weekdays: more bias, far less variance, no
# warm-up cost. The tree gets both widths and blends them.
SLOT_CLIMATOLOGY_DAYS = 14

CALENDAR_COLUMNS = (
    "slot_sin", "slot_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday",
)

# The same clock as plain integers. Sin/cos is right about the wrap and wrong
# about the edge: isolating one slot on one weekday costs five conjoined splits
# on the circle and one on an integer. Both are built; the tree picks.
INTEGER_CALENDAR_COLUMNS = ("slot", "dow")

# What the table CARRIES. `CALENDAR_COLUMNS` is what has always been served;
# which of the extras are served is `SHIPPED_EXTRAS` below.
ALL_CALENDAR_COLUMNS = CALENDAR_COLUMNS + INTEGER_CALENDAR_COLUMNS

STATE_COLUMNS = ("state_now", "minutes_in_state", "coverage")

# Where they are and which way they are going. `distance_delta_*` are explicit
# because a tree cannot subtract two columns, and 8 km after 30 km means
# something different from 8 km all afternoon.
PROXIMITY_COLUMNS = (
    "distance_km", "distance_delta_30m", "distance_delta_60m",
    "dir_towards", "dir_away",
)

# Built into the parquet but not served: a column that is NaN for all but the
# last few days trains as "unknown" and is worse than absent. Shipping one is a
# tuple edit plus a retrain.
BUILT_NOT_SHIPPED: tuple[str, ...] = (
    "next_alarm_h",
    "is_charging",
    "detected_activity_still",
)

# Candidates: always built, served only when named here, so a probe can measure
# one against a control on one parquet and one set of fold windows. Every name
# below has been measured against a control and not shipped; re-run
# `python -m occupancy_forecast.probe` when the archive is longer. Shipping a
# winner is a tuple edit plus a MODEL_VERSION bump plus a retrain.
#
#   "int_calendar"  INTEGER_CALENDAR_COLUMNS, origin and target
#   "wclim_wide"    the same-weekday climatology over WIDE_CLIMATOLOGY_WEEKS
#   "wclim_slope"   the climatology's slope either side of the target slot
SHIPPED_EXTRAS: tuple[str, ...] = ()


def extra_origin_columns() -> tuple[str, ...]:
    """Candidate ORIGIN columns this build serves. See SHIPPED_EXTRAS."""
    out: list[str] = []
    if "int_calendar" in SHIPPED_EXTRAS:
        out.extend(INTEGER_CALENDAR_COLUMNS)
    return tuple(out)


def extra_target_columns(horizon: int) -> tuple[str, ...]:
    """Candidate TARGET columns for one horizon, wide names."""
    out = [f"tgt{horizon}h_{name}" for name in extra_origin_columns()]
    if "wclim_wide" in SHIPPED_EXTRAS:
        out.append(wide_climatology_column(horizon))
    if "wclim_slope" in SHIPPED_EXTRAS:
        out.extend(climatology_slope_columns(horizon))
    return tuple(out)


def extra_long_columns() -> tuple[str, ...]:
    """The same, as the melt names them."""
    out = [f"tgt_{name}" for name in extra_origin_columns()]
    if "wclim_wide" in SHIPPED_EXTRAS:
        out.append(f"wclim{WIDE_CLIMATOLOGY_WEEKS}")
    if "wclim_slope" in SHIPPED_EXTRAS:
        out.extend(f"wclim{TRANSITION_SOURCE_WEEKS}_{side}"
                   for side in ("slope_back", "slope_fwd"))
    return tuple(out)


def target_calendar_columns(horizon: int) -> tuple[str, ...]:
    """The SERVED target calendar for one horizon. Extras are separate, so that
    a candidate can be built without being fed to the model."""
    return tuple(f"tgt{horizon}h_{name}" for name in CALENDAR_COLUMNS)


def safe_daily_lags(horizon: int) -> tuple[int, ...]:
    """Daily lags of the target slot that do not reach past the origin.

    `tgt{h}h_lag{k}d` is only observable when `24k >= h`. This is the easiest
    way to leak in this problem, so the gate lives in one function and
    `test_features.py` asserts it.
    """
    return tuple(k for k in DAILY_LAGS if 24 * k >= horizon)


def wide_climatology_column(horizon: int) -> str:
    return f"tgt{horizon}h_wclim{WIDE_CLIMATOLOGY_WEEKS}"


def climatology_slope_columns(horizon: int) -> tuple[str, str]:
    """How far the same-weekday climatology moves either side of the target.

    Explicit because the two climatologies being differenced live on different
    rows of the melted frame, so no model can form the difference itself.
    """
    stem = f"tgt{horizon}h_wclim{TRANSITION_SOURCE_WEEKS}"
    return f"{stem}_slope_back", f"{stem}_slope_fwd"


def climatology_column(horizon: int) -> str:
    return f"tgt{horizon}h_wclim{CLIMATOLOGY_WEEKS}"


def slot_climatology_days(horizon: int) -> tuple[int, ...]:
    """Whole-day offsets the slot climatology may average over, gated exactly
    like `safe_daily_lags`: a day that has not happened cannot be averaged into
    anything."""
    return tuple(k for k in range(1, SLOT_CLIMATOLOGY_DAYS + 1) if 24 * k >= horizon)


def slot_climatology_column(horizon: int) -> str:
    return f"tgt{horizon}h_sclim{SLOT_CLIMATOLOGY_DAYS}"


def zone_columns() -> tuple[str, ...]:
    """Every zone column in build order, `zone_other` last. Derived from
    config, never a literal -- see `train.may_be_nan`."""
    return (*(f"zone_{slug}" for slug in config.zone_slugs()), "zone_other")


def cross_subject_lag_column(horizon: int, slug: str, days: int) -> str:
    return f"tgt{horizon}h_other_{slug}_lag{days}d"


def cross_subject_lag_columns(horizon: int) -> tuple[str, ...]:
    """The other subjects' state at the target slot, on the nearest legal day
    only: the full cross product says the same thing five times with increasing
    staleness."""
    lags = safe_daily_lags(horizon)
    if not lags:
        return ()
    return tuple(cross_subject_lag_column(horizon, slug, min(lags))
                 for slug in config.all_slugs())


# ---------------------------------------------------------------------------
# The long form
#
# The table is wide on disk (footer stats, and the leakage gate stays a
# selection over column names) and melted for the model, one row per (subject,
# slot, horizon). The melt is where the lag gate is applied, by omission.
# ---------------------------------------------------------------------------

HORIZON_COLUMN = "horizon_h"

# The lag whose offset varies with the horizon, carried explicitly rather than
# inferred from `horizon_h`.
OTHER_LAG_DAYS_COLUMN = "other_lag_days"

TARGET_COLUMN = "y"


def long_target_calendar_columns() -> tuple[str, ...]:
    return tuple(f"tgt_{name}" for name in CALENDAR_COLUMNS)


def long_shipped_columns() -> tuple[str, ...]:
    """What the pooled model is FED, against `long_columns`'s what the table
    HAS. They differ only by `SHIPPED_EXTRAS`."""
    return (*long_columns(), *extra_long_columns())


def long_daily_lag_columns() -> tuple[str, ...]:
    return tuple(f"lag{days}d" for days in DAILY_LAGS)


def long_cross_subject_lag_columns() -> tuple[str, ...]:
    return tuple(f"other_{slug}_lag" for slug in config.all_slugs())


def long_columns() -> tuple[str, ...]:
    """Everything `tgt{h}h_*` collapses to, in melt order. Two names shift
    meaning at the h=24/25 boundary, which is why `OTHER_LAG_DAYS_COLUMN`
    exists."""
    return (
        *long_target_calendar_columns(),
        *long_daily_lag_columns(),
        *long_cross_subject_lag_columns(),
        f"wclim{CLIMATOLOGY_WEEKS}",
        f"sclim{SLOT_CLIMATOLOGY_DAYS}",
        OTHER_LAG_DAYS_COLUMN,
    )


def _long_renames(horizon: int) -> dict[str, str]:
    """Wide column -> long column, for one horizon. A POSITIVE selection: a lag
    that reaches past the origin is simply absent, so it lands as NaN with
    nothing to remember to mask."""
    out = {f"y_{horizon}h": TARGET_COLUMN}
    for name in ALL_CALENDAR_COLUMNS:
        out[f"tgt{horizon}h_{name}"] = f"tgt_{name}"
    for days in safe_daily_lags(horizon):
        out[f"tgt{horizon}h_lag{days}d"] = f"lag{days}d"
    out[climatology_column(horizon)] = f"wclim{CLIMATOLOGY_WEEKS}"
    out[wide_climatology_column(horizon)] = f"wclim{WIDE_CLIMATOLOGY_WEEKS}"
    back, forward = climatology_slope_columns(horizon)
    out[back] = f"wclim{TRANSITION_SOURCE_WEEKS}_slope_back"
    out[forward] = f"wclim{TRANSITION_SOURCE_WEEKS}_slope_fwd"
    out[slot_climatology_column(horizon)] = f"sclim{SLOT_CLIMATOLOGY_DAYS}"
    lags = safe_daily_lags(horizon)
    if lags:
        for slug in config.all_slugs():
            out[cross_subject_lag_column(horizon, slug, min(lags))] = f"other_{slug}_lag"
    return out


def origin_columns(table: pd.DataFrame) -> list[str]:
    """The columns that do not vary with the horizon, so are copied per chunk."""
    return [c for c in table.columns
            if not c.startswith("tgt") and not c.startswith("y_")]


def long_frame(table: pd.DataFrame, horizons=None,
               subjects: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Melt the wide table to one row per (subject, slot, horizon). `subjects`
    filters before melting; the add-on passes nothing, it is for the explorer
    and the tests."""
    horizons = config.HORIZONS_H if horizons is None else horizons
    if subjects is not None:
        table = table[table["subject"].isin(subjects)]
    keep = origin_columns(table)
    chunks = []
    for horizon in horizons:
        renames = _long_renames(horizon)
        present = {old: new for old, new in renames.items() if old in table.columns}
        chunk = table[keep + list(present)].rename(columns=present)
        # Absent because this horizon may not see them -- see `_long_renames`.
        for column in (TARGET_COLUMN, *long_columns()):
            if column not in chunk.columns:
                chunk[column] = np.nan
        chunk[HORIZON_COLUMN] = float(horizon)
        lags = safe_daily_lags(horizon)
        chunk[OTHER_LAG_DAYS_COLUMN] = float(min(lags)) if lags else np.nan
        chunks.append(chunk)
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(["subject", "time", HORIZON_COLUMN]).reset_index(drop=True)


# Every family a built column can belong to, in the order the panel lists them.
FAMILIES: tuple[str, ...] = (
    "key", "state", "target", "proximity", "calendar", "zone",
    "cross_subject", "horizon_target", "target_calendar", "daily_lag",
    "cross_subject_lag", "climatology", "climatology_slope", "slot_climatology",
    "not_shipped",
)


def column_family(name: str) -> str:
    """Which family a feature column belongs to.

    Here because this module mints the names; a classifier anywhere else goes
    stale silently, and `test_features` asserts every built column lands
    somewhere.
    """
    if name in ("time", "subject"):
        return "key"
    if name == "home_frac":
        return "target"
    # Separate from `home_frac`: 48 answer columns rather than one input.
    if name.startswith("y_"):
        return "horizon_target"
    if name in STATE_COLUMNS:
        return "state"
    if name in PROXIMITY_COLUMNS:
        return "proximity"
    if name in ALL_CALENDAR_COLUMNS:
        return "calendar"
    if name in BUILT_NOT_SHIPPED:
        return "not_shipped"
    if name.startswith("zone_"):
        return "zone"
    if name.startswith("other_"):
        return "cross_subject"
    if name.startswith("tgt"):
        # `other_` before `lag`: a cross-subject lag ends in one.
        rest = name.split("_", 1)[1] if "_" in name else ""
        if rest.startswith("other_"):
            return "cross_subject_lag"
        if rest.startswith("lag"):
            return "daily_lag"
        # Slope before level, or `wclim4_slope_back` is swallowed by the level
        # branch and the two can never be priced apart.
        if rest.startswith("wclim") and "_slope_" in rest:
            return "climatology_slope"
        if rest.startswith("wclim"):
            return "climatology"
        if rest.startswith("sclim"):
            return "slot_climatology"
        if rest in ALL_CALENDAR_COLUMNS:
            return "target_calendar"
    return "unknown"


# What each family is, in words, for the panel. Kept beside the classifier so a
# new family cannot be added without a sentence explaining it.
FAMILY_WORDS: dict[str, str] = {
    "key": "the row's identity — which subject, and which slot",
    "state": "where they are right now, and how well that is known",
    "target": "home_frac — the fraction of the slot spent home, which is what all of this is about",
    "horizon_target": "the same thing at each of the 48 horizons: one answer column per model",
    "proximity": "how far from home and which way they are moving",
    "calendar": "the time of the slot being read from",
    "zone": "which of the zones you enabled they were in — the model works out "
            "what each one means for each person, so none of them has a role",
    "cross_subject": "where everyone else is, at the origin",
    "target_calendar": "the time of the slot being predicted — known in advance",
    "daily_lag": "the same slot on earlier days, gated so none reaches past the origin",
    "cross_subject_lag": "where everyone else was in the target slot, on the "
                         "nearest earlier day the forecast is allowed to see",
    "climatology": "the same weekday and slot, averaged over the trailing weeks",
    "climatology_slope": "how far that weekday average moves across the hour "
                         "either side of the slot — a tree cannot subtract two "
                         "columns, and in the melted table the two being "
                         "differenced are on different rows",
    "slot_climatology": "the same slot on every recent day, weekdays pooled — "
                        "blunter than the weekday average and far less noisy",
    "not_shipped": "built but deliberately not served yet — too little history to train on",
    "unknown": "not recognised — this is a bug in column_family",
}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def grid(start: pd.Timestamp, stop: pd.Timestamp) -> pd.DatetimeIndex:
    """Left edges of every whole slot in [start, stop]: slot `t` is
    `[t, t+30min)`. The newest row is therefore the IN-PROGRESS slot -- it reads
    as stale and is not, and phase-shifting the grid only makes it less
    responsive. See `nowcast.py`."""
    freq = f"{config.GRID_MINUTES}min"
    return pd.date_range(start.ceil(freq), stop.floor(freq), freq=freq,
                         tz="UTC", name="time")


def slot_fraction(events: list[tuple[str, str]], slots: pd.DatetimeIndex,
                  match: str, minutes: int | None = None) -> pd.DataFrame:
    """Time-weighted fraction of each slot whose state equals `match`, with
    `coverage`. An unobserved slot gets coverage 0 and frac NaN -- it is not an
    empty house. `minutes` is a parameter only so `nowcast` can reuse this
    integration."""
    minutes = config.GRID_MINUTES if minutes is None else minutes
    slot_seconds = minutes * 60
    n = len(slots)
    empty = pd.DataFrame({"frac": np.full(n, np.nan), "coverage": np.zeros(n)},
                         index=slots)
    if n == 0 or not events:
        return empty

    times = pd.to_datetime([t for t, _ in events], utc=True, format="ISO8601")
    # NaN, not 0.0, for "no reading": the segment leaves both numerator and
    # denominator, so a silent tracker costs coverage instead of reading as away.
    values = np.array([np.nan if config.is_empty(v)
                       else 1.0 if str(v).strip() == match else 0.0
                       for _, v in events])
    order = np.argsort(times.asi8, kind="stable")
    times, values = times[order], values[order]

    boundaries = slots.append(
        pd.DatetimeIndex([slots[-1] + pd.Timedelta(minutes=minutes)], tz="UTC"))

    # Every boundary is in the timeline, so each segment lies inside one slot.
    marks = times.union(boundaries)
    idx = np.searchsorted(times.asi8, marks.asi8, side="right") - 1
    held = np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)

    seconds = np.diff(marks.asi8) / 1e9
    seg_value = held[:-1]
    seg_slot = np.searchsorted(boundaries.asi8, marks.asi8[:-1], side="right") - 1

    keep = (seg_slot >= 0) & (seg_slot < n) & ~np.isnan(seg_value)
    if not keep.any():
        return empty

    numerator = np.bincount(seg_slot[keep], weights=seg_value[keep] * seconds[keep],
                            minlength=n)
    denominator = np.bincount(seg_slot[keep], weights=seconds[keep], minlength=n)

    coverage = denominator / slot_seconds
    frac = np.divide(numerator, denominator,
                     out=np.full(n, np.nan), where=denominator > 0)
    frac[coverage < config.MIN_SLOT_COVERAGE] = np.nan
    return pd.DataFrame({"frac": frac, "coverage": coverage}, index=slots)


def observability(event_times: pd.DatetimeIndex, slots: pd.DatetimeIndex,
                  max_silence_h: float = None) -> np.ndarray:
    """Which slots were observed rather than carried forward: a slot inside a
    silence longer than `max_silence_h` is unobservable.

    A recorder outage once produced three weeks of `home_frac == 1.00`. The
    mask comes from the union of the person trackers and is applied to every
    subject including the house, whose group is event-driven and would blank
    itself.
    """
    if max_silence_h is None:
        max_silence_h = config.MAX_SILENCE_H
    observable = np.zeros(len(slots), dtype=bool)
    if len(event_times) == 0 or len(slots) == 0:
        return observable

    times = event_times.sort_values()
    limit = np.int64(max_silence_h * 3600 * 1e9)

    # A slot is observable when its surrounding pair of observations is no
    # further apart than the limit, and it lies inside the observed span at all.
    after = np.searchsorted(times.asi8, slots.asi8, side="right")
    inside = (after > 0) & (after < len(times))
    idx = np.clip(after, 1, len(times) - 1)
    span = times.asi8[idx] - times.asi8[idx - 1]
    return inside & (span <= limit)


def numeric_on_grid(pairs: list[tuple[str, float]], slots: pd.DatetimeIndex,
                    stale_after_min: float | None) -> np.ndarray:
    """Last numeric reading at or before each slot. `stale_after_min=None`
    carries forward indefinitely -- right where silence is meaningful (see
    `config.DISTANCE_STALE_MIN`), wrong for anything that drifts."""
    out = np.full(len(slots), np.nan)
    if not pairs or len(slots) == 0:
        return out

    times = pd.to_datetime([t for t, _ in pairs], utc=True, format="ISO8601")
    values = np.asarray([v for _, v in pairs], dtype=float)
    order = np.argsort(times.asi8, kind="stable")
    times, values = times[order], values[order]

    idx = np.searchsorted(times.asi8, slots.asi8, side="right") - 1
    seen = idx >= 0
    safe = np.clip(idx, 0, None)
    fresh = seen
    if stale_after_min is not None:
        age_min = (slots.asi8 - times.asi8[safe]) / 1e9 / 60
        fresh = seen & (age_min <= stale_after_min)
    out[fresh] = values[safe][fresh]
    return out


def minutes_in_state(events: list[tuple[str, str]], slots: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since the last state *change*, at each slot's left edge.
    Occupancy is duration-dependent and a tree cannot derive this from the
    state alone."""
    n = len(slots)
    if n == 0 or not events:
        return np.full(n, np.nan)

    times = pd.to_datetime([t for t, _ in events], utc=True, format="ISO8601")
    values = [str(v).strip() for _, v in events]
    order = np.argsort(times.asi8, kind="stable")
    times = times[order]
    values = [values[i] for i in order]

    changed_at = np.empty(len(values), dtype="int64")
    last = times.asi8[0]
    previous = None
    for i, value in enumerate(values):
        if value != previous:
            last = times.asi8[i]
            previous = value
        changed_at[i] = last

    idx = np.searchsorted(times.asi8, slots.asi8, side="right") - 1
    out = np.where(idx >= 0,
                   (slots.asi8 - changed_at[np.clip(idx, 0, None)]) / 1e9 / 60,
                   np.nan)
    return out


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _subject_frame(source, subject: config.Subject, start: str, stop: str | None,
                   slots: pd.DatetimeIndex,
                   observable: np.ndarray | None = None) -> pd.DataFrame:
    """One subject's presence, on the grid."""
    events = presence_events(source, subject, start, stop)
    occupancy = slot_fraction(events, slots, config.HOME_STATE)

    frame = pd.DataFrame(index=slots)
    frame["subject"] = subject.slug
    frame["home_frac"] = occupancy["frac"].to_numpy()
    frame["coverage"] = occupancy["coverage"].to_numpy()
    frame["minutes_in_state"] = minutes_in_state(events, slots)

    # People only: the house is a group with no zone of its own; `build` fills
    # its columns from the people.
    if subject.is_person:
        _add_zones(frame, events, slots)

    _add_proximity(frame, source, subject, start, stop, slots)
    # Built, not served: it stays out of `train.base_features()` until it has
    # enough history to be priced. See BUILT_NOT_SHIPPED.
    _add_next_alarm(frame, source, subject, start, stop, slots)

    if observable is not None:
        # Unobserved is neither away nor home. The house's zone columns do not
        # exist yet -- `build` fills them from the people, already blanked here.
        blank = [c for c in ("home_frac", "minutes_in_state",
                             *zone_columns(), *PROXIMITY_COLUMNS)
                 if c in frame.columns]
        frame.loc[~observable, blank] = np.nan
        frame.loc[~observable, "coverage"] = 0.0
    return frame


def presence_events(source, subject: config.Subject, start: str,
                    stop: str | None) -> list[tuple[str, str]]:
    """`home`/away transitions for one subject.

    With no group configured the house is the OR over the people; a person
    enters that merge at their first observation, so somebody added today does
    not make the house unknown retroactively.
    """
    if subject.is_person or subject.entity_id:
        return source.seeded_states(subject.entity_id, start, stop, seed_days=14)

    per_person = [source.seeded_states(p.entity_id, start, stop, seed_days=14)
                  for p in config.PEOPLE]
    merged: list[tuple[str, str]] = []
    latest: dict[int, str] = {}
    stamped = sorted(
        ((when, i, value) for i, rows in enumerate(per_person) for when, value in rows),
        key=lambda item: item[0])
    for when, index, value in stamped:
        latest[index] = value
        # Asymmetric on purpose: one person known home makes the house home
        # despite unknowns; nobody known home plus an unknown is unknown, not empty.
        anyone = any(v == config.HOME_STATE for v in latest.values())
        all_known = all(not config.is_empty(v) for v in latest.values())
        state = config.HOME_STATE if anyone else (
            "not_home" if all_known else "unknown")
        # Several people can write at one instant; only the last state held for
        # any time.
        if merged and merged[-1][0] == when:
            merged.pop()
        if not merged or merged[-1][1] != state:
            merged.append((when, state))
    return merged


def _add_next_alarm(frame: pd.DataFrame, source, subject: config.Subject,
                    start: str, stop: str | None,
                    slots: pd.DatetimeIndex) -> None:
    """Hours from each slot to that person's next phone alarm, or NaN -- the
    only signal here that knows about tomorrow.

    The app writes the literal `absent` when none is set, so this parses
    `seeded_states` itself: `source.numeric` drops that row and would carry a
    cancelled alarm forward forever.
    """
    if not subject.next_alarm_entity:
        frame["next_alarm_h"] = np.nan
        return

    alarms: list[pd.Timestamp] = []
    stamps: list[pd.Timestamp] = []
    for when, value in source.seeded_states(subject.next_alarm_entity, start, stop):
        try:
            alarm = pd.Timestamp(value)
        except (ValueError, TypeError):
            alarm = pd.NaT                      # absent, unavailable, unknown
        if alarm is not pd.NaT and not pd.isna(alarm):
            alarm = (alarm.tz_localize("UTC") if alarm.tzinfo is None
                     else alarm.tz_convert("UTC"))
        stamps.append(pd.Timestamp(when))
        alarms.append(alarm)

    if not stamps:
        frame["next_alarm_h"] = np.nan
        return

    # The state in force at each slot's left edge.
    index = np.searchsorted(pd.DatetimeIndex(stamps), slots, side="right") - 1
    hours = np.full(len(slots), np.nan)
    for i, at in enumerate(index):
        if at < 0:
            continue
        alarm = alarms[at]
        if alarm is pd.NaT or pd.isna(alarm):
            continue
        ahead = (alarm - slots[i]).total_seconds() / 3600.0
        # A fired alarm the phone has not cleared yet is not a forecast.
        if ahead > 0:
            hours[i] = ahead
    frame["next_alarm_h"] = hours


def _add_proximity(frame: pd.DataFrame, source, subject: config.Subject,
                   start: str, stop: str | None, slots: pd.DatetimeIndex) -> None:
    """Distance to home and direction of travel, on the grid. Three cases: a
    real Proximity sensor, a synthesised distance, or nothing at all -- the
    third is not an error, the columns go NaN."""
    from .discover import synthetic_distance_entity

    distance_entity = subject.distance_entity
    if distance_entity is None and subject.is_person:
        distance_entity = synthetic_distance_entity(subject.slug)

    if distance_entity is None:
        for column in PROXIMITY_COLUMNS:
            frame[column] = np.nan
        return

    metres = numeric_on_grid(
        source.numeric(distance_entity, start, stop),
        slots, config.DISTANCE_STALE_MIN)
    frame["distance_km"] = metres / 1000.0

    # The grid is regular, so a positional shift IS a fixed time offset here --
    # unlike the target join, which crosses gaps. One slot is GRID_MINUTES.
    frame["distance_delta_30m"] = frame["distance_km"].diff(1)
    frame["distance_delta_60m"] = frame["distance_km"].diff(slots_per_hour())

    if subject.direction_entity:
        events = source.seeded_states(subject.direction_entity, start, stop)
        frame["dir_towards"] = slot_fraction(events, slots, "towards")["frac"].to_numpy()
        frame["dir_away"] = slot_fraction(events, slots, "away_from")["frac"].to_numpy()
    else:
        # No Proximity integration: derive direction from the sign of the
        # distance delta, which is all `direction_of_travel` is anyway.
        delta = frame["distance_km"].diff()
        frame["dir_towards"] = (delta < -0.05).astype(float).where(delta.notna())
        frame["dir_away"] = (delta > 0.05).astype(float).where(delta.notna())


# Away but in no enabled zone. A real string rather than NaN, so
# `slot_fraction` integrates it like any state and the columns sum to one.
ZONE_OTHER = "__other__"


def _resolve_zone_events(events: list[tuple[str, str]],
                         name_map: dict[str, str]) -> list[tuple[str, str]]:
    """Person state strings -> zone entity ids.

    The one place that reads a zone's friendly name, because a name is the only
    per-person zone signal the archive contains -- entity-id-keyed attributes
    exist only in the present. `name_map` comes from the live zones, never a
    literal, and anything unresolved is counted by `unmatched_away_states`
    rather than folded into "away".
    """
    resolved: list[tuple[str, str]] = []
    for when, value in events:
        text = str(value).strip()
        if text == config.HOME_STATE:
            resolved.append((when, config.HOME_STATE))
        else:
            resolved.append((when, name_map.get(text.lower(), ZONE_OTHER)))
    return resolved


def _add_zones(frame: pd.DataFrame, events: list[tuple[str, str]],
               slots: pd.DatetimeIndex) -> None:
    """One column per enabled zone plus `zone_other`, over the same integration
    as `home_frac`, so a covered slot sums to one. `test_features` pins that."""
    resolved = _resolve_zone_events(events, config.zone_name_map())
    for zone in config.ZONES:
        frame[f"zone_{zone.slug}"] = slot_fraction(
            resolved, slots, zone.entity_id)["frac"].to_numpy()
    frame["zone_other"] = slot_fraction(
        resolved, slots, ZONE_OTHER)["frac"].to_numpy()


def unmatched_away_states(source, start: str, stop: str | None) -> dict[str, int]:
    """Away-states matching no enabled zone, with counts -- a renamed zone shows
    up here. `not_home` is excluded: it is HA's own word for "in no zone"."""
    known = set(config.zone_name_map())
    counts: dict[str, int] = {}
    for person in config.PEOPLE:
        for _, value in source.states(person.entity_id, start, stop):
            text = str(value).strip()
            if text in (config.HOME_STATE, "not_home") or text.lower() in known:
                continue
            counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The calendar, in one place: everything that labels a day or a slot reads
# these, so a change to the grid or the calendar is one edit.
# ---------------------------------------------------------------------------

def slots_per_hour() -> int:
    return 60 // config.GRID_MINUTES


def slot_of_day(local: pd.Series) -> pd.Series:
    """0 .. SLOTS_PER_DAY-1 for a tz-aware LOCAL datetime series: a slot is a
    wall-clock position, and on the autumn transition two UTC slots map to one
    local slot."""
    return local.dt.hour * slots_per_hour() + local.dt.minute // config.GRID_MINUTES


def is_weekend(dow) -> np.ndarray:
    """1.0 on Saturday and Sunday, from pandas' Monday=0 weekday."""
    return (np.asarray(dow) >= 5).astype(float)


def holiday_flags(dates) -> np.ndarray:
    """Public holidays as 1.0/0.0, or zeros when the calendar is unknown. An
    unsupported or unset country must not abort a six-month build, so every
    failure degrades to a flat column with one debug line."""
    index = pd.DatetimeIndex(dates)
    zeros = np.zeros(len(index))
    if not config.HOLIDAY_COUNTRY or len(index) == 0:
        return zeros
    try:
        import holidays
        years = sorted({int(y) for y in index.year if not pd.isna(y)})
        calendar = holidays.country_holidays(config.HOLIDAY_COUNTRY, years=years)
    except Exception as err:  # noqa: BLE001
        _log.debug("no holiday calendar for %r: %s", config.HOLIDAY_COUNTRY, err)
        return zeros
    return np.array([1.0 if d in calendar else 0.0 for d in index.date])


def _cyclical(local: pd.Series, prefix: str = "") -> pd.DataFrame:
    """Sine/cosine encodings of time-of-day, weekday and month: trees can split
    a raw hour but cannot see that 23:30 and 00:00 are adjacent."""
    slot = slot_of_day(local)
    dow = local.dt.dayofweek
    month = local.dt.month

    out = pd.DataFrame(index=local.index)
    out[f"{prefix}slot_sin"] = np.sin(2 * np.pi * slot / config.SLOTS_PER_DAY)
    out[f"{prefix}slot_cos"] = np.cos(2 * np.pi * slot / config.SLOTS_PER_DAY)
    out[f"{prefix}dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out[f"{prefix}dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out[f"{prefix}month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out[f"{prefix}month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out[f"{prefix}is_weekend"] = is_weekend(dow)
    out[f"{prefix}is_holiday"] = holiday_flags(local)
    # The same clock as an integer, always built and served only via
    # SHIPPED_EXTRAS.
    out[f"{prefix}slot"] = slot.astype(float)
    out[f"{prefix}dow"] = dow.astype(float)
    return out


def _liveness(source, start: str, stop: str | None) -> pd.DatetimeIndex:
    """Every moment history can be shown to have been recorded: the collector's
    heartbeats plus every tracked entity's state changes.

    All tracked entities, not just the people -- on a change-only source the
    people alone go quiet for hours and a quiet night reads as an outage.
    """
    times = pd.DatetimeIndex([], tz="UTC")

    heartbeats = getattr(source, "liveness_times", None)
    if heartbeats is not None:
        times = times.union(pd.to_datetime(heartbeats(start, stop), utc=True,
                                           format="ISO8601"))

    tracked = [s.entity_id for s in config.SUBJECTS]
    tracked += [s.distance_entity for s in config.SUBJECTS]
    tracked += [s.direction_entity for s in config.SUBJECTS]
    # Still evidence the recorder was alive, though no feature reads them.
    tracked += [z.entity_id for z in config.ZONES]
    for entity in tracked:
        if not entity:
            continue
        events = source.states(entity, start, stop)
        if events:
            times = times.union(pd.to_datetime([t for t, _ in events], utc=True,
                                               format="ISO8601"))
    return times


def _localise(times: pd.Series) -> pd.Series:
    """Local time, with a readable error: `tz_convert` on an unknown zone
    raises deep inside pandas."""
    try:
        return times.dt.tz_convert(config.TIMEZONE)
    except Exception as err:
        raise ValueError(
            f"cannot use timezone {config.TIMEZONE!r} (from Home Assistant's "
            f"/api/config): {err}") from err


def _at_offset(table: pd.DataFrame, keyed: pd.Series, delta: pd.Timedelta) -> np.ndarray:
    """`home_frac` for the same subject `delta` away. An explicit join on a
    shifted timestamp, never `.shift()`: the grid has holes and a positional
    shift pairs rows hours apart."""
    index = pd.MultiIndex.from_arrays([table["subject"], table["time"] + delta])
    return keyed.reindex(index).to_numpy()


def _at_time_offset(table: pd.DataFrame, keyed: pd.Series,
                    delta: pd.Timedelta) -> np.ndarray:
    """The subject-blind twin of `_at_offset`, for household-wide series keyed
    on time alone. Same explicit join, same reason."""
    return keyed.reindex(table["time"] + delta).to_numpy()


def deepest_lookback_days() -> int:
    """How far back a build must reach before every feature is real.
    `predict.LOOKBACK_DAYS` derives from this so widening a window cannot
    silently start serving NaN."""
    # Tracks what is SERVED, not what is built: an unserved candidate has no
    # NaN hazard, and charging every five-minute rebuild for its lookback is a
    # real cost. The training build always spans the whole archive.
    weeks = (WIDE_CLIMATOLOGY_WEEKS if "wclim_wide" in SHIPPED_EXTRAS
             else CLIMATOLOGY_WEEKS)
    return max(7 * weeks, SLOT_CLIMATOLOGY_DAYS, max(DAILY_LAGS))


def _add_horizon_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Targets, target-time calendar, and target-relative lags, per horizon."""
    keyed = table.set_index(["subject", "time"])["home_frac"]

    wide = table.pivot_table(index="time", columns="subject", values="home_frac",
                             aggfunc="first")

    new: dict[str, np.ndarray] = {}

    for horizon in config.HORIZONS_H:
        ahead = pd.Timedelta(hours=horizon)

        # The target itself.
        new[f"y_{horizon}h"] = _at_offset(table, keyed, ahead)

        # Calendar of the slot being predicted. Deterministic and known at
        # prediction time, so this is information, not leakage.
        target_local = _localise(table["time"] + ahead)
        for name, values in _cyclical(target_local, prefix=f"tgt{horizon}h_").items():
            new[name] = values.to_numpy()

        # Daily lags OF THE TARGET SLOT. Gated -- see safe_daily_lags.
        for days in DAILY_LAGS:
            column = f"tgt{horizon}h_lag{days}d"
            new[column] = _at_offset(table, keyed, ahead - pd.Timedelta(days=days))

        # Same-weekday climatology of the target slot. Widths are prefixes of
        # one offset list, so `wclim4` is unchanged. No horizon gate is needed:
        # the nearest weekly input is at least 120 h before the origin.
        weeks = []
        for week in range(1, max(CLIMATOLOGY_WIDTHS) + 1):
            weeks.append(_at_offset(table, keyed, ahead - pd.Timedelta(days=7 * week)))
        with warnings.catch_warnings():
            # An all-NaN row is the honest answer for the first weeks; the
            # RuntimeWarning is noise.
            warnings.simplefilter("ignore", RuntimeWarning)
            for width in CLIMATOLOGY_WIDTHS:
                column = (climatology_column(horizon) if width == CLIMATOLOGY_WEEKS
                          else wide_climatology_column(horizon))
                new[column] = np.nanmean(np.vstack(weeks[:width]), axis=0)

            # The slopes either side, computed from their own offsets, not by
            # shifting to `t + 1 h`: that version is legal in training and
            # always NaN in production, since serving uses the newest row.
            width = TRANSITION_SOURCE_WEEKS
            neighbours = {}
            for step in (-1, 1):
                shifted = ahead + pd.Timedelta(hours=step)
                stack = [_at_offset(table, keyed, shifted - pd.Timedelta(days=7 * w))
                         for w in range(1, width + 1)]
                neighbours[step] = np.nanmean(np.vstack(stack), axis=0)
            here = np.nanmean(np.vstack(weeks[:width]), axis=0)
            back, forward = climatology_slope_columns(horizon)
            new[back] = here - neighbours[-1]
            new[forward] = neighbours[1] - here

        # The same slot on every recent day, weekdays POOLED -- blunter and
        # much quieter than four weekday samples; the tree gets both.
        daily = [_at_offset(table, keyed, ahead - pd.Timedelta(days=k))
                 for k in slot_climatology_days(horizon)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            new[slot_climatology_column(horizon)] = (
                np.nanmean(np.vstack(daily), axis=0) if daily
                else np.full(len(table), np.nan))

        # Everyone else's state IN THE TARGET SLOT, `k` days ago: not the
        # answer, the same construction as the row's own `lag{k}d`, gated the
        # same way.
        lags = safe_daily_lags(horizon)
        if lags:
            back = ahead - pd.Timedelta(days=min(lags))
            is_self = table["subject"].to_numpy()
            for slug in config.all_slugs():
                values = (_at_time_offset(table, wide[slug], back)
                          if slug in wide.columns else np.full(len(table), np.nan))
                # A subject never mirrors itself: that column is already
                # tgt{h}h_lag{k}d.
                new[cross_subject_lag_column(horizon, slug, min(lags))] = np.where(
                    is_self == slug, np.nan, values)

    return pd.concat([table, pd.DataFrame(new, index=table.index)], axis=1)


def _add_cross_subject(table: pd.DataFrame) -> pd.DataFrame:
    """The other people's state at the ORIGIN, never the target: reading the
    partner at the target slot would be reading the answer."""
    wide = table.pivot_table(index="time", columns="subject", values="home_frac",
                             aggfunc="first")
    for slug in config.all_slugs():
        if slug not in wide.columns:
            wide[slug] = np.nan

    merged = table.merge(wide.add_prefix("other_"), left_on="time",
                         right_index=True, how="left")
    # Blank out a subject's own column so `other_*` never mirrors `state_now`.
    for slug in config.all_slugs():
        column = f"other_{slug}"
        merged.loc[merged["subject"] == slug, column] = np.nan
    return merged


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def history_start(source) -> str:
    """The earliest moment worth asking for. A store knows its own span; an
    Influx does not, so it gets a floor and returns what it has."""
    span = getattr(getattr(source, "store", None), "span", None)
    if span:
        first = span().get("first")
        if first:
            return first

    # An Influx carries no span, so ask for the earliest point rather than guess.
    first_seen = getattr(source, "first_seen", None)
    if first_seen:
        earliest = first_seen([s.entity_id for s in config.SUBJECTS if s.entity_id])
        if earliest:
            return earliest

    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=400)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def usable_history_days(source, stop: str | None = None) -> float:
    """Days of history a model could actually be fitted on -- counted the way
    training counts it, so missing data cannot make the add-on look ready
    early. EXPENSIVE: call it from the worker, never a request handler."""
    if not config.PEOPLE:
        return 0.0
    start = history_start(source)
    end = (pd.Timestamp(stop) if stop else pd.Timestamp.now(tz="UTC")).floor(
        f"{config.GRID_MINUTES}min")
    slots = grid(pd.Timestamp(start), end)
    if not len(slots):
        return 0.0

    valid = observability(_liveness(source, start, end.isoformat()), slots)
    for person in config.PEOPLE:
        events = presence_events(source, person, start, end.isoformat())
        valid &= slot_fraction(events, slots, config.HOME_STATE)["frac"].notna().to_numpy()
    return float(np.count_nonzero(valid)) / config.SLOTS_PER_DAY


def build(source, start: str | None = None, stop: str | None = None) -> pd.DataFrame:
    """Build the modelling table. Full rebuild every time: an incremental
    version would be a correctness risk for no measurable saving."""
    config.require()
    start = start or history_start(source)
    start_ts = pd.Timestamp(start)
    stop_ts = pd.Timestamp(stop) if stop else pd.Timestamp.now(tz="UTC")
    slots = grid(start_ts, stop_ts)
    if len(slots) == 0:
        raise ValueError(f"empty grid for {start}..{stop_ts.isoformat()}")

    # One observability mask for everyone -- see `observability`.
    observable = observability(_liveness(source, start, stop), slots)

    frames = {
        subject.slug: _subject_frame(source, subject, start, stop, slots, observable)
        for subject in config.SUBJECTS
    }

    # The house gets the union over the people -- "is anyone in this zone".
    # `fmax` so one unobserved person does not blank the household.
    people = [frames[s.slug] for s in config.PEOPLE]
    house = frames[config.HOUSE_SLUG]
    for column in zone_columns():
        if people:
            house[column] = np.fmax.reduce([f[column].to_numpy() for f in people])
        else:
            house[column] = np.nan

    table = pd.concat(frames.values()).reset_index()
    table = table.rename(columns={"index": "time"})

    local = _localise(table["time"])
    table = pd.concat([table, _cyclical(local)], axis=1)

    table["state_now"] = table["home_frac"]

    table = _add_cross_subject(table)
    table = _add_horizon_columns(table)

    # Placeholders for the companion-app sensors nothing computes yet;
    # `next_alarm_h` only lands here for a subject with no alarm entity.
    for column in BUILT_NOT_SHIPPED:
        if column not in table.columns:
            table[column] = np.nan

    return table.sort_values(["subject", "time"]).reset_index(drop=True)


def write(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    from . import runtime

    parser = argparse.ArgumentParser(description="Build the occupancy feature table")
    parser.add_argument("--start", default=None)
    parser.add_argument("--stop", default=None)
    parser.add_argument("--out", type=Path, default=config.FEATURES_PATH)
    args = parser.parse_args(argv)

    _, _, source, _ = runtime.bootstrap()

    began = dt.datetime.now()
    table = build(source, args.start, args.stop)
    write(table, args.out)

    labelled = table["home_frac"].notna().sum()
    print(f"{len(table)} rows, {labelled} with a label "
          f"({100 * labelled / max(len(table), 1):.1f}% coverage), "
          f"{len(table.columns)} columns -> {args.out} "
          f"in {(dt.datetime.now() - began).total_seconds():.1f}s")
    for subject, part in table.groupby("subject"):
        print(f"  {subject:<10} mean home_frac {part['home_frac'].mean():.3f}  "
              f"P(home>=0.5) {(part['home_frac'] >= 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
