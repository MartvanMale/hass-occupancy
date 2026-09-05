"""Build the modelling table: one row per (subject, 30-minute slot).

The target is `home_frac` -- the **time-weighted fraction of the slot spent at
home** -- not a last-observation-carried-forward binary.

That choice is the whole reason this module does not simply reuse an ordinary
last-value resample. Measured on a real installation, 19% of one person's presence
episodes are shorter than five minutes, and 11 of their 28 workplace-zone
episodes are under *two* minutes -- GPS jitter at a zone edge, e.g.

    14:22:11 zone.office -> 14:22:49 not_home -> 14:26:16 zone.office

Taking the last observation at or before each grid point is right for a
thermostat and wrong here: a ninety-second blip that happens to land on a grid
point becomes a spurious empty-house slot. Time-weighting debounces by
construction, needs no dwell threshold to tune, and gives a more informative
target into the bargain.

Slots with less than `config.MIN_SLOT_COVERAGE` of their duration observed are
NaN rather than a guess. On the current history that costs about 1% of slots.
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Feature groups. Named so a drop/only probe can price them as a unit.
# ---------------------------------------------------------------------------

# Daily lag offsets, in days, of the TARGET slot.
#
# 3 and 21 are here for the LONG horizons specifically. `safe_daily_lags` throws
# away every lag `k` with `24k < horizon`, so past +25 h the 1-day lag is gone
# and the band that was left with three legal anchors now has five. Both sit
# inside the 45-day warm-up `evaluate.MIN_TRAIN_DAYS` already pays for, so they
# cost no folds. 28 would not -- see CLIMATOLOGY_WEEKS.
DAILY_LAGS = (1, 2, 3, 7, 14, 21)

# How many trailing same-weekdays go into the in-table climatology feature.
#
# Four is thin -- roughly four samples per (weekday, slot) cell, and baseline.py
# says outright that cell means at that sample size are mostly noise. Widening
# it is not free: `evaluate.MIN_TRAIN_DAYS` is 45 BECAUSE the longest in-table
# feature is a four-week climatology, so eight weeks pushes the warm-up to ~65
# days and folds are exactly what the ship gate spends. Probe it before moving
# it; SLOT_CLIMATOLOGY below is the cheaper half of the same idea.
CLIMATOLOGY_WEEKS = 4

# The same climatology over twice as many weekdays. A candidate, not a default.
#
# The per-(subject, weekday, slot) cell has a MEASURED median of 21 observations
# available in this household's history; `wclim4` uses four of them, so its cell
# means take values on a five-point grid and the tree has good reason to
# discount them. Eight is the cheapest step that halves that.
#
# Costs no folds: `evaluate.MIN_TRAIN_DAYS` is a hand-set 45 and `nanmean`
# returns as soon as ONE week exists, so the first fold does not move. What it
# does move is `deepest_lookback_days` and therefore `predict.LOOKBACK_DAYS`,
# 32 -> 60 days, and every serving cycle is a full lookback rebuild -- that is
# the price to weigh, and it is per cycle rather than per week.
WIDE_CLIMATOLOGY_WEEKS = 8
# MEASURED 2026-09-02 and NOT SHIPPED. Four arms over one parquet and one set of
# 16 fold windows (`occupancy_forecast.probe`), against the control:
#
#     arm            transition    flat   | transition folds   flat folds
#     control            0.1966  0.1359   |
#     wclim_slope        0.1897  0.1322   |  10/16 p=0.454   12/16 p=0.077
#     int_calendar       0.1917  0.1358   |   9/16 p=0.804    9/16 p=0.804
#     wclim_wide         0.1853  0.1315   |   9/16 p=0.804   10/16 p=0.454
#
# On the pre-registered primary metric -- departure-hour error through
# `predict._crossing` -- every arm was neutral or worse than the control, and
# none won a fold majority. That metric has n=72 person-days over 13 folds and
# is too weak to decide anything; the fold records above are the usable
# evidence, and the largest pooled effect has the weakest one. Sixteen cells
# were inspected, so the best-looking is selected-for.
#
# Nothing here is disproven, and one thing is handicapped: on 173 days an
# 8-week climatology is only fully populated for the later folds, so
# `wclim_wide` is measured on an archive too short to show it at its best. Worth
# re-running when the history is longer -- `python -m occupancy_forecast.probe
# --features <copy> --arms control,wclim_wide`.

# Every width the table carries. `climatology_column(h)` still means the served
# one; the rest are candidates.
CLIMATOLOGY_WIDTHS = (CLIMATOLOGY_WEEKS, WIDE_CLIMATOLOGY_WEEKS)

# Which width the transition slopes are differenced from.
#
# The narrow one deliberately, so that the slope arm is ONE change from the
# control rather than two: a slope built on a width that is itself a candidate
# would conflate "the slope helps" with "the wider window helps". If the wide
# width earns its place, rebuild the slopes on it and measure that separately --
# the difference of two 4-sample means is the noisier quantity and it may be
# what limits the slope.
TRANSITION_SOURCE_WEEKS = CLIMATOLOGY_WEEKS

# The other half of the climatology, pooled over ALL weekdays rather than split
# by them. Fourteen samples per (slot) cell against the weekday version's four:
# more bias, much less variance, and no warm-up cost at all because it reuses
# days the table already has. The tree gets both and can blend them, which is
# the honest way to answer "is four samples enough" -- let it decide per split
# rather than picking one width for every horizon.
SLOT_CLIMATOLOGY_DAYS = 14

CALENDAR_COLUMNS = (
    "slot_sin", "slot_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday",
)

# The same clock, as plain integers, beside the trig pair.
#
# The sine/cosine encoding is right about the WRAP -- 23:30 and 00:00 are
# adjacent and a raw hour cannot say so -- and wrong about the EDGE. Isolating
# one slot on one weekday for one person out of a circle costs a conjunction of
# splits on `slot_sin` AND `slot_cos` AND `dow_sin` AND `dow_cos` AND the
# subject one-hot, five deep, under `min_samples_leaf=100` on ~500 independent
# person-days. With an integer the same cut is `slot in [14, 16)`: one split.
#
# MEASURED symptom that prompted this: a person who leaves for work between
# 07:00 and 08:00 every Thursday was forecast at 0.93 at 07:14 and did not
# reach her true level until ~11:00 -- the right answer, three to four hours
# late. Her own weekday climatology for that slot said 0.354 at the time.
#
# Both encodings are served. The tree picks per split, which is the same
# argument SLOT_CLIMATOLOGY_DAYS makes about widths.
INTEGER_CALENDAR_COLUMNS = ("slot", "dow")
# MEASURED 2026-09-02 and NOT SHIPPED. Four arms over one parquet and one set of
# 16 fold windows (`occupancy_forecast.probe`), against the control:
#
#     arm            transition    flat   | transition folds   flat folds
#     control            0.1966  0.1359   |
#     wclim_slope        0.1897  0.1322   |  10/16 p=0.454   12/16 p=0.077
#     int_calendar       0.1917  0.1358   |   9/16 p=0.804    9/16 p=0.804
#     wclim_wide         0.1853  0.1315   |   9/16 p=0.804   10/16 p=0.454
#
# On the pre-registered primary metric -- departure-hour error through
# `predict._crossing` -- every arm was neutral or worse than the control, and
# none won a fold majority. That metric has n=72 person-days over 13 folds and
# is too weak to decide anything; the fold records above are the usable
# evidence, and the largest pooled effect has the weakest one. Sixteen cells
# were inspected, so the best-looking is selected-for.
#
# Nothing here is disproven, and one thing is handicapped: on 173 days an
# 8-week climatology is only fully populated for the later folds, so
# `wclim_wide` is measured on an archive too short to show it at its best. Worth
# re-running when the history is longer -- `python -m occupancy_forecast.probe
# --features <copy> --arms control,wclim_wide`.

# What the table CARRIES. `CALENDAR_COLUMNS` is what has always been served;
# which of the extras are served is `SHIPPED_EXTRAS` below.
ALL_CALENDAR_COLUMNS = CALENDAR_COLUMNS + INTEGER_CALENDAR_COLUMNS

STATE_COLUMNS = ("state_now", "minutes_in_state", "coverage")

# Where they are and which way they are going. See config.PROXIMITY.
#
# `distance_delta_*` are explicit differences because a tree cannot subtract two
# columns, so the difference has to be handed to it as its own feature.
# A distance of 8 km means something completely different when it was 30 km half
# an hour ago than when it was 8 km all afternoon, and only the delta says which.
PROXIMITY_COLUMNS = (
    "distance_km", "distance_delta_30m", "distance_delta_60m",
    "dir_towards", "dir_away",
)

# Columns that are computed into the parquet but deliberately NOT served yet.
# Companion-app sensors land here: they are usually enabled long after the
# recorder started, and a feature that is NaN for all but the last few days of
# the history trains as "unknown" and is worse than not having the feature at
# all. Shipping one is a single tuple edit plus a retrain, once it has history.
BUILT_NOT_SHIPPED: tuple[str, ...] = (
    "next_alarm_h",
    "is_charging",
    "detected_activity_still",
)

# Candidate features: always BUILT into the parquet, served only when named.
#
# The same split as BUILT_NOT_SHIPPED above and for a different reason. That one
# withholds a column until it has history; this one exists so a candidate can be
# MEASURED against a control on ONE parquet, one set of fold windows and
# identical rows. Rebuilding the table between arms would change the newest row
# and can move a fold edge, and `train.shared_windows` exists precisely because
# a comparison cut two ways is not a comparison.
#
# `occupancy_forecast/probe.py` flips this in-process, one arm at a time. Shipping a
# winner is one tuple edit plus a MODEL_VERSION bump plus a retrain.
#
#   "int_calendar"  INTEGER_CALENDAR_COLUMNS, origin and target
#   "wclim_wide"    the same-weekday climatology over WIDE_CLIMATOLOGY_WEEKS
#   "wclim_slope"   the climatology's slope either side of the target slot
#
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

    `tgt{h}h_lag{k}d` is `home_frac` at `t + h - 24k`. That is only observable
    at prediction time when `24k >= h`. At a 36 h horizon the target's
    "yesterday" is twelve hours into the *future* -- including it would train a
    model that cannot be served, and would score beautifully doing it.

    This is the single easiest way to leak in this problem, so the gate lives in
    one function and `test_features.py` asserts it.
    """
    return tuple(k for k in DAILY_LAGS if 24 * k >= horizon)


def wide_climatology_column(horizon: int) -> str:
    return f"tgt{horizon}h_wclim{WIDE_CLIMATOLOGY_WEEKS}"


def climatology_slope_columns(horizon: int) -> tuple[str, str]:
    """How far the same-weekday climatology moves either side of the target.

    `slope_back` is what has already happened across the hour ending at the
    target slot; `slope_fwd` is what usually happens across the hour after it.
    A tree cannot subtract two columns -- the same argument
    `PROXIMITY_COLUMNS`'s `distance_delta_*` makes -- and in the melted frame
    the two climatologies being differenced live on DIFFERENT ROWS, one per
    horizon, so no amount of capacity would let it form the difference itself.
    """
    stem = f"tgt{horizon}h_wclim{TRANSITION_SOURCE_WEEKS}"
    return f"{stem}_slope_back", f"{stem}_slope_fwd"


def climatology_column(horizon: int) -> str:
    return f"tgt{horizon}h_wclim{CLIMATOLOGY_WEEKS}"


def slot_climatology_days(horizon: int) -> tuple[int, ...]:
    """Whole-day offsets the slot climatology may average over.

    Gated exactly like `safe_daily_lags` and for exactly the same reason -- a
    day that has not happened yet cannot be averaged into anything. The window
    therefore narrows with the horizon rather than shifting: 14 days of samples
    below +25 h, 13 above it. That is still three times what the same-weekday
    climatology gets.
    """
    return tuple(k for k in range(1, SLOT_CLIMATOLOGY_DAYS + 1) if 24 * k >= horizon)


def slot_climatology_column(horizon: int) -> str:
    return f"tgt{horizon}h_sclim{SLOT_CLIMATOLOGY_DAYS}"


def zone_columns() -> tuple[str, ...]:
    """Every zone column, in build order. `zone_other` is always the last.

    Derived from config, never a literal -- the same rule `train.may_be_nan`
    learned the hard way. An install with no zones ticked gets just
    `zone_other`, which is then simply "away", and nothing downstream cares.
    """
    return (*(f"zone_{slug}" for slug in config.zone_slugs()), "zone_other")


def cross_subject_lag_column(horizon: int, slug: str, days: int) -> str:
    return f"tgt{horizon}h_other_{slug}_lag{days}d"


def cross_subject_lag_columns(horizon: int) -> tuple[str, ...]:
    """The other subjects' state at the target slot, on the nearest legal day.

    Nearest only, not every lag. The full cross product is 48 horizons x every
    subject x every safe lag, which would more than double a table that is
    already too wide to browse, to say the same thing five times with
    increasing staleness.
    """
    lags = safe_daily_lags(horizon)
    if not lags:
        return ()
    return tuple(cross_subject_lag_column(horizon, slug, min(lags))
                 for slug in config.all_slugs())


# ---------------------------------------------------------------------------
# The long form
#
# The built table is WIDE: one row per (subject, slot), with a `tgt{h}h_*` block
# per horizon. That is the right shape on disk -- the Data tab reads per-column
# statistics straight out of the parquet footer, and `safe_daily_lags` stays a
# positive selection over column NAMES, which is what keeps the leakage gate
# structural.
#
# It is the wrong shape for the model. One fit per horizon cannot share what
# h=25 and h=26 obviously have in common, and MEASURED on 173 days that costs
# real skill: the per-horizon models run from +17.8% at h=1 to **-20.2% at
# h=48**, worse than the trivial baseline they are scored against, because each
# of the 48 is overfitting the same ~500 person-days independently.
#
# So the table is melted to one row per (subject, slot, horizon) with
# horizon-relative names, and `horizon_h` becomes a feature the model can split
# on. The melt is where the gate is applied: a lag this horizon may not see is
# never copied across, so it arrives as NaN by omission rather than by a mask
# somebody has to remember to write.
# ---------------------------------------------------------------------------

HORIZON_COLUMN = "horizon_h"

# The lag whose offset varies with the horizon (1 d below +25 h, 2 d above it),
# carried explicitly so the model is told which it is holding rather than having
# to infer it from `horizon_h`.
OTHER_LAG_DAYS_COLUMN = "other_lag_days"

TARGET_COLUMN = "y"


def long_target_calendar_columns() -> tuple[str, ...]:
    return tuple(f"tgt_{name}" for name in CALENDAR_COLUMNS)


def long_shipped_columns() -> tuple[str, ...]:
    """What the pooled model is FED, against `long_columns`'s what the table HAS.

    The two differ only by `SHIPPED_EXTRAS`. Keeping them apart is what lets
    `long_frame` mint a candidate column -- so it exists to be measured, and so
    `nan_allowed` still tolerates it -- while `train.base_features` leaves it
    out until it has earned a place.
    """
    return (*long_columns(), *extra_long_columns())


def long_daily_lag_columns() -> tuple[str, ...]:
    return tuple(f"lag{days}d" for days in DAILY_LAGS)


def long_cross_subject_lag_columns() -> tuple[str, ...]:
    return tuple(f"other_{slug}_lag" for slug in config.all_slugs())


def long_columns() -> tuple[str, ...]:
    """Everything `tgt{h}h_*` collapses to, in melt order.

    Seventeen-plus-n_subjects columns per horizon become this many columns
    total. The two whose *definition* shifts at the h=24/25 boundary --
    `sclim{N}` averages 14 days below it and 13 above, `other_{slug}_lag` steps
    from a 1 d to a 2 d offset -- keep one name each, which is why
    `OTHER_LAG_DAYS_COLUMN` exists.
    """
    return (
        *long_target_calendar_columns(),
        *long_daily_lag_columns(),
        *long_cross_subject_lag_columns(),
        f"wclim{CLIMATOLOGY_WEEKS}",
        f"sclim{SLOT_CLIMATOLOGY_DAYS}",
        OTHER_LAG_DAYS_COLUMN,
    )


def _long_renames(horizon: int) -> dict[str, str]:
    """Wide column -> long column, for one horizon.

    A POSITIVE selection, exactly as `train.features_for` has always been: a
    daily lag that reaches past the origin is simply absent from this mapping,
    so it lands in the melted frame as NaN. Nothing has to remember to mask it.
    `test_features` asserts the resulting NaNs directly, because a test written
    against the feature list would pass vacuously on a long table.
    """
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
    """Melt the wide table to one row per (subject, slot, horizon).

    Chunked per horizon and concatenated once, so peak memory is the result
    rather than the result plus a 48-way intermediate.

    `subjects` filters before melting rather than after, so a caller that wants
    one subject does not pay to melt the other two. Both families train over
    every subject, the house included, so the add-on itself passes nothing --
    the parameter is for the explorer and the tests.
    """
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


# Every family a built column can belong to, in the order the panel lists them:
# what the model is aiming at, then what it knows now, then what it knows about
# the slot it is aiming at.
FAMILIES: tuple[str, ...] = (
    "key", "state", "target", "proximity", "calendar", "zone",
    "cross_subject", "horizon_target", "target_calendar", "daily_lag",
    "cross_subject_lag", "climatology", "climatology_slope", "slot_climatology",
    "not_shipped",
)


def column_family(name: str) -> str:
    """Which family a feature column belongs to.

    Here rather than in the panel's server because THIS module mints the names:
    `_add_horizon_columns` builds `tgt{h}h_lag{k}d`, `_add_cross_subject` builds
    `other_{slug}`, `_add_zones` builds `zone_{slug}`. A classifier living
    anywhere else goes stale the moment a family is added, and goes stale
    silently -- a thousand columns is far past the point where anyone would notice a
    handful quietly reclassified as "unknown". `test_features` asserts that
    every column of a real `build()` lands somewhere, which is the guard.

    A thousand columns is also why this exists at all: the table cannot be browsed as a
    table, so the panel summarises it by family instead.
    """
    if name in ("time", "subject"):
        return "key"
    if name == "home_frac":
        return "target"
    # Separate from `home_frac`, because they are 48 columns rather than one and
    # they are answers rather than inputs -- the panel offers the origin target
    # as something to chart and summarises these by count.
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
        # tgt{h}h_<rest>. The suffix decides, and the shapes are disjoint.
        # `other_` is tested before `lag`, because a cross-subject lag ends in
        # one: tgt48h_other_alice_lag2d would otherwise never be reached.
        rest = name.split("_", 1)[1] if "_" in name else ""
        if rest.startswith("other_"):
            return "cross_subject_lag"
        if rest.startswith("lag"):
            return "daily_lag"
        # BEFORE the plain `wclim` test, and for the same reason `other_` is
        # tested before `lag`: tgt36h_wclim4_slope_back starts with `wclim`, so
        # the level branch would swallow it and the two could never be priced
        # apart -- which is the whole reason families exist.
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
    """Left edges of every whole slot in [start, stop].

    A slot's window runs FORWARD from its left edge: slot `t` is `[t, t+30min)`,
    which `slot_fraction` and `test_half_a_slot_away_is_a_half` both depend on.
    So the newest row here is the IN-PROGRESS slot, labelled by an `observed_at`
    up to 30 minutes old but carrying the present state carried forward through
    the rest of its window. It reads as stale and is not; do not "fix" it by
    phase-shifting the grid, which only makes the newest row a longer trailing
    average and strictly less responsive. See `nowcast.py`.
    """
    freq = f"{config.GRID_MINUTES}min"
    return pd.date_range(start.ceil(freq), stop.floor(freq), freq=freq,
                         tz="UTC", name="time")


def slot_fraction(events: list[tuple[str, str]], slots: pd.DatetimeIndex,
                  match: str, minutes: int | None = None) -> pd.DataFrame:
    """Time-weighted fraction of each slot whose state equals `match`.

    A slot runs FORWARD from its left edge: slot `t` covers `[t, t+minutes)`.
    `test_half_a_slot_away_is_a_half` pins that down, and it is the reason the
    newest row of a `build` is the in-progress slot rather than a finished one.

    Returns a frame indexed by `slots` with `frac` and `coverage`, where
    coverage is the fraction of the slot actually spanned by observations.
    A slot with no observation at all gets coverage 0 and frac NaN -- an
    unobserved slot is not an empty house.

    The step function is integrated exactly: slot boundaries are inserted into
    the event timeline before integrating, so no segment straddles two slots and
    a state change mid-slot is split between them in proportion to its duration.

    `minutes` defaults to the modelling grid. It is a parameter only so that
    `nowcast` can reuse this exact integration over a shorter window -- the
    arithmetic here is subtle enough that a second copy of it would be a
    liability, and two definitions of "fraction of a window spent at home"
    would be worse still.
    """
    minutes = config.GRID_MINUTES if minutes is None else minutes
    slot_seconds = minutes * 60
    n = len(slots)
    empty = pd.DataFrame({"frac": np.full(n, np.nan), "coverage": np.zeros(n)},
                         index=slots)
    if n == 0 or not events:
        return empty

    times = pd.to_datetime([t for t, _ in events], utc=True, format="ISO8601")
    values = np.array([1.0 if str(v).strip() == match else 0.0 for _, v in events])
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
    """Which slots were actually observed, as opposed to merely carried forward.

    A slot is unobservable if it sits inside a silence longer than
    `max_silence_h`. Returns a boolean array aligned to `slots`.

    This exists because the step function in `slot_fraction` holds the last
    known state forward for as long as it has to, and "as long as it has to"
    turned out to be 653 hours: HA recorded nothing between 2026-06-26 and
    2026-07-23, and the first build of this table duly reported three straight
    weeks of `home_frac == 1.00` for all three subjects. See config.MAX_SILENCE_H
    for why 12 hours separates that from the nightly phone doze.

    The mask is computed once from the union of the *person* trackers and
    applied to every subject, including the house. group.home_people is
    event-driven -- it only writes on a change, so its own median gap is 67
    minutes and its p95 is 23 hours -- so judging it by its own silence would
    blank almost all of it. When the trackers were down, nobody's occupancy is
    known, and that includes the house's.
    """
    if max_silence_h is None:
        max_silence_h = config.MAX_SILENCE_H
    observable = np.zeros(len(slots), dtype=bool)
    if len(event_times) == 0 or len(slots) == 0:
        return observable

    times = event_times.sort_values()
    limit = np.int64(max_silence_h * 3600 * 1e9)

    # For each slot, the surrounding pair of observations. A slot is observable
    # when that pair is no further apart than the limit, and when the slot is
    # inside the observed span at all.
    after = np.searchsorted(times.asi8, slots.asi8, side="right")
    inside = (after > 0) & (after < len(times))
    idx = np.clip(after, 1, len(times) - 1)
    span = times.asi8[idx] - times.asi8[idx - 1]
    return inside & (span <= limit)


def numeric_on_grid(pairs: list[tuple[str, float]], slots: pd.DatetimeIndex,
                    stale_after_min: float | None) -> np.ndarray:
    """Last numeric reading at or before each slot.

    `stale_after_min=None` carries the last value forward indefinitely, which is
    right for a sensor whose silence is meaningful -- see config.DISTANCE_STALE_MIN
    for the measurements behind that. Pass a number for anything that drifts
    while you are not looking.
    """
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

    Occupancy is strongly duration-dependent -- "home for eight hours" and "home
    for four minutes" predict very different next hours -- and a boosted tree
    cannot derive this from the state alone.
    """
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

    # People only. The house is a group whose state collapses to home/not_home,
    # so it has no zone of its own; `build` fills its columns with the union
    # over the people instead.
    if subject.is_person:
        _add_zones(frame, events, slots)

    _add_proximity(frame, source, subject, start, stop, slots)
    # Built, not served: it stays out of `train.base_features()` until it has
    # enough history to be priced. See BUILT_NOT_SHIPPED.
    _add_next_alarm(frame, source, subject, start, stop, slots)

    if observable is not None:
        # Unobserved is not "away", and it is not "home" either.
        # The house's zone columns do not exist yet -- `build` fills them from
        # the people, who have already been blanked here.
        blank = [c for c in ("home_frac", "minutes_in_state",
                             *zone_columns(), *PROXIMITY_COLUMNS)
                 if c in frame.columns]
        frame.loc[~observable, blank] = np.nan
        frame.loc[~observable, "coverage"] = 0.0
    return frame


def presence_events(source, subject: config.Subject, start: str,
                    stop: str | None) -> list[tuple[str, str]]:
    """`home`/away transitions for one subject.

    The house is normally a person group, but a group is not required: with no
    group configured its presence is the OR over the people, merged from their
    individual traces. That keeps `group` from being a hard dependency for
    something Home Assistant can already tell us.
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
        anyone = any(v == config.HOME_STATE for v in latest.values())
        state = config.HOME_STATE if anyone else "not_home"
        if not merged or merged[-1][1] != state:
            merged.append((when, state))
    return merged


def _add_next_alarm(frame: pd.DataFrame, source, subject: config.Subject,
                    start: str, stop: str | None,
                    slots: pd.DatetimeIndex) -> None:
    """Hours from each slot to that person's next phone alarm, or NaN.

    **The only signal here that knows about TOMORROW.** Everything else is a
    lagging measurement of what this household usually does -- a weekday median,
    a trailing rate, a partner's habits. An alarm is a statement of intent made
    the evening before, and it moves on exactly the nights the routine moves.

    The companion app writes an ISO timestamp while an alarm is set and the
    literal `absent` when none is, so `absent` must CLEAR the value rather than
    be skipped. That is why this reads `seeded_states` and parses each event
    itself instead of going through `source.numeric`, which drops every
    non-float row and would carry a cancelled alarm forward forever.

    Self-limiting against a phone that stops reporting: the carried value is an
    absolute timestamp, so once it is in the past it stops producing a positive
    number and the column goes NaN on its own.
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

    # The state in force at each slot's left edge: the last event at or before
    # it. `searchsorted` rather than a merge, because the slots are already
    # sorted and this is one pass.
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
    """Distance to home and direction of travel, on the grid.

    Three cases, all of which have to work: a real Proximity sensor, a distance
    synthesised by the collector from GPS, and nothing at all. The third is not
    an error -- the columns go NaN, HistGradientBoosting ignores them, and the
    ship gate prices what is left.
    """
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
    per_hour = 60 // config.GRID_MINUTES
    frame["distance_delta_30m"] = frame["distance_km"].diff(1)
    frame["distance_delta_60m"] = frame["distance_km"].diff(per_hour)

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


# The value a resolved event takes when the person is away but in none of the
# enabled zones. A real string rather than NaN so `slot_fraction` can integrate
# it exactly like any other state, which is what makes the columns sum to one.
ZONE_OTHER = "__other__"


def _resolve_zone_events(events: list[tuple[str, str]],
                         name_map: dict[str, str]) -> list[tuple[str, str]]:
    """Person state strings -> zone entity ids.

    **The one place in this package that reads a zone's friendly name**, and it
    is worth saying why it reads one at all. Home Assistant writes the zone's
    NAME into the person's state, and the store keeps `(entity_id, ts, value)`
    with no attributes -- so `zone.*`'s `persons` list and `person.*`'s
    `in_zones` list, both of which are entity-id keyed and rename-proof, exist
    only in the present and cannot be reconstructed over history. A name is the
    only per-person zone signal history actually contains.

    What broke before was not the join, it was a hardcoded `OFFICE_STATES =
    {"work"}` that matched nothing and said nothing about it for five months.
    So: `name_map` is built from the LIVE zone entities (config.zone_name_map,
    refreshed every boot), never from a literal, and everything it fails to
    translate is counted by `unmatched_away_states` rather than being folded
    silently into "away". A rename now shows up as a number on the status page.

    A zone the user has NOT enabled resolves to ZONE_OTHER, which is the whole
    point of the enable list: an untick means "lump this in with everywhere
    else", not "pretend they were home".
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
    """One column per enabled zone, plus `zone_other`, for one person.

    Reuses the events `_subject_frame` already fetched and the same
    `slot_fraction` integration as `home_frac`, so within a fully covered slot
    `home_frac + sum(zone_*) + zone_other == 1` by construction. test_features
    pins that.
    """
    resolved = _resolve_zone_events(events, config.zone_name_map())
    for zone in config.ZONES:
        frame[f"zone_{zone.slug}"] = slot_fraction(
            resolved, slots, zone.entity_id)["frac"].to_numpy()
    frame["zone_other"] = slot_fraction(
        resolved, slots, ZONE_OTHER)["frac"].to_numpy()


def unmatched_away_states(source, start: str, stop: str | None) -> dict[str, int]:
    """Away-states that match no enabled zone, and how many times each occurs.

    The guard the old friendly-name match never had. A renamed zone turns its
    whole history into rows this returns, and `server._feature_groups` puts the
    count where the user will see it. `not_home` is excluded: it is Home
    Assistant's own word for "away, in no zone at all", not a name that failed
    to resolve.
    """
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

def _cyclical(local: pd.Series, prefix: str = "") -> pd.DataFrame:
    """Sine/cosine encodings of time-of-day, weekday and month.

    Trees can split a raw hour just fine, but they cannot see that 23:30 and
    00:00 are adjacent, so every model has to relearn the wrap from data it does
    not have much of. The abandoned first attempt at this problem used the same
    encoding; it was the one good idea in it.
    """
    slot = local.dt.hour * (60 // config.GRID_MINUTES) + local.dt.minute // config.GRID_MINUTES
    dow = local.dt.dayofweek
    month = local.dt.month

    out = pd.DataFrame(index=local.index)
    out[f"{prefix}slot_sin"] = np.sin(2 * np.pi * slot / config.SLOTS_PER_DAY)
    out[f"{prefix}slot_cos"] = np.cos(2 * np.pi * slot / config.SLOTS_PER_DAY)
    out[f"{prefix}dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out[f"{prefix}dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out[f"{prefix}month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out[f"{prefix}month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out[f"{prefix}is_weekend"] = (dow >= 5).astype(float)
    out[f"{prefix}is_holiday"] = _holiday_flags(local)
    # The same clock as an integer, for the edge the circle cannot cut cheaply.
    # See INTEGER_CALENDAR_COLUMNS. Always built; served only when named in
    # SHIPPED_EXTRAS.
    out[f"{prefix}slot"] = slot.astype(float)
    out[f"{prefix}dow"] = dow.astype(float)
    return out


def _holiday_flags(local: pd.Series) -> np.ndarray:
    """Public holidays, or all-zeros when we cannot know.

    `country_holidays` raises `NotImplementedError` for a country the library
    does not cover, and Home Assistant's `country` can also be unset entirely.
    Neither is a reason to abort a six-month feature build -- a flat column is
    simply a feature that carries nothing, which the model already handles.
    """
    zeros = np.zeros(len(local))
    if not config.HOLIDAY_COUNTRY:
        return zeros
    try:
        import holidays
        years = sorted(set(local.dt.year.dropna().astype(int)))
        calendar = holidays.country_holidays(config.HOLIDAY_COUNTRY, years=years)
    except (NotImplementedError, KeyError, AttributeError):
        return zeros
    return np.array([1.0 if d in calendar else 0.0 for d in local.dt.date])


def _liveness(source, start: str, stop: str | None) -> pd.DatetimeIndex:
    """Every moment we can show that history was being recorded.

    The union of the collector's heartbeats (exact, but only since the add-on
    was installed) and every tracked entity's state changes (a proxy, and the
    only thing available for the backfilled prefix).

    Using ALL tracked entities rather than just the people matters on a
    change-only source: MEASURED, the two person entities alone have a p95 gap
    of 9.2 h, while the union of the nine tracked entities has a p95 of 0.1 h.
    A mask built on the former calls a quiet night an outage.
    """
    times = pd.DatetimeIndex([], tz="UTC")

    heartbeats = getattr(source, "liveness_times", None)
    if heartbeats is not None:
        times = times.union(pd.to_datetime(heartbeats(start, stop), utc=True,
                                           format="ISO8601"))

    tracked = [s.entity_id for s in config.SUBJECTS]
    tracked += [s.distance_entity for s in config.SUBJECTS]
    tracked += [s.direction_entity for s in config.SUBJECTS]
    # Still collected, and still counted as evidence the recorder was alive,
    # even though no feature reads a zone's own count any more.
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
    """Local time, with a message worth reading when the zone is wrong.

    `tz_convert` on an unknown zone raises deep inside pandas, which is a poor
    way to learn that Home Assistant reported something unexpected.
    """
    try:
        return times.dt.tz_convert(config.TIMEZONE)
    except Exception as err:
        raise ValueError(
            f"cannot use timezone {config.TIMEZONE!r} (from Home Assistant's "
            f"/api/config): {err}") from err


def _at_offset(table: pd.DataFrame, keyed: pd.Series, delta: pd.Timedelta) -> np.ndarray:
    """`home_frac` for the same subject, `delta` away from each row's time.

    An explicit join on a shifted timestamp, never `.shift()`. The grid has
    holes wherever a slot fell below the coverage threshold, and a positional
    shift would then quietly pair rows that are hours apart while labelling them
    as one slot.
    """
    index = pd.MultiIndex.from_arrays([table["subject"], table["time"] + delta])
    return keyed.reindex(index).to_numpy()


def _at_time_offset(table: pd.DataFrame, keyed: pd.Series,
                    delta: pd.Timedelta) -> np.ndarray:
    """The subject-blind twin of `_at_offset`, for household-wide series.

    The work zones and the other subjects' traces describe somebody other than
    the row being built, so they are keyed on time alone. Same explicit join for
    the same reason -- the grid has holes.
    """
    return keyed.reindex(table["time"] + delta).to_numpy()


def deepest_lookback_days() -> int:
    """How far back a build has to reach before every feature is real.

    `predict.LOOKBACK_DAYS` is derived from this rather than hand-set, so that
    widening any window here cannot silently start serving a NaN column. Every
    term is a whole-day reach measured backwards from the TARGET slot; the
    horizon itself is added on top by the caller.
    """
    # Tracks what is SERVED, not what is built.
    #
    # Both halves matter. A served candidate whose build did not reach far
    # enough would average fewer weeks in production than in training --
    # silently, since a short reach fills it with NaN rather than failing --
    # which is the hazard `predict.LOOKBACK_DAYS` derives from this function to
    # avoid. But an UNSERVED candidate has no such hazard, because nothing reads
    # it, and charging the serving path for it is a real cost: every cycle is a
    # full lookback rebuild, every five minutes. Building `wclim8` unconditionally
    # while nothing served it took LOOKBACK_DAYS from 32 to 60 days for nothing.
    #
    # The training build is unaffected either way: `server.do_train` calls
    # `features.build(source)` with no start, so the parquet always spans the
    # whole archive and a candidate stays measurable while it is unserved.
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
        weekly = []
        for days in DAILY_LAGS:
            column = f"tgt{horizon}h_lag{days}d"
            new[column] = _at_offset(table, keyed, ahead - pd.Timedelta(days=days))

        # Same-weekday climatology of the target slot, from the trailing weeks.
        #
        # Every width is a PREFIX of one offset list, so `wclim4` is bit for bit
        # what it was before the wide one existed and the extra cost is the
        # extra joins alone.
        #
        # No horizon gate is needed here and it is worth saying why, since the
        # daily lags next door need one: the nearest weekly input is
        # `t + h - 168 h`, which for h <= 48 is at least 120 h before the
        # origin. A weekly offset cannot reach the future.
        weeks = []
        for week in range(1, max(CLIMATOLOGY_WIDTHS) + 1):
            weeks.append(_at_offset(table, keyed, ahead - pd.Timedelta(days=7 * week)))
        with warnings.catch_warnings():
            # An all-NaN row is the honest answer for the first weeks of
            # history -- there is no same-weekday climatology yet. NaN is what
            # the model should see; the warning is noise.
            warnings.simplefilter("ignore", RuntimeWarning)
            for width in CLIMATOLOGY_WIDTHS:
                column = (climatology_column(horizon) if width == CLIMATOLOGY_WEEKS
                          else wide_climatology_column(horizon))
                new[column] = np.nanmean(np.vstack(weeks[:width]), axis=0)

            # The slopes either side, from the SAME climatology an hour earlier
            # and an hour later.
            #
            # Computed from their own offsets rather than by shifting to the
            # row at `t + 1 h`. That version is legal in training and always
            # NaN in production -- `predict.current_rows` serves the newest row
            # per subject, so no later row exists -- which is the quietest kind
            # of train/serve skew there is: populated in every training row and
            # absent in every served one.
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

        # The same slot on every recent day, weekdays POOLED. Fourteen samples
        # against the weekday climatology's four: blunter, and much quieter.
        # baseline.py measured that ~24 samples per (weekday, slot) cell is not
        # enough to estimate a probability, and the weekday version has far
        # fewer -- so give the tree both widths and let it choose per split
        # rather than picking one for all 48 horizons.
        daily = [_at_offset(table, keyed, ahead - pd.Timedelta(days=k))
                 for k in slot_climatology_days(horizon)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            new[slot_climatology_column(horizon)] = (
                np.nanmean(np.vstack(daily), axis=0) if daily
                else np.full(len(table), np.nan))

        # Everyone else's state IN THE TARGET SLOT, on the nearest legal day.
        #
        # `_add_cross_subject` is right that reading the partner at the target
        # slot would be reading the answer -- but the partner at the target slot
        # `k` days ago is not the answer, it is the same construction
        # `tgt{h}h_lag{k}d` already uses for the row's own subject, gated the
        # same way. In a household this is exactly the information the long
        # horizons are missing.
        lags = safe_daily_lags(horizon)
        if lags:
            back = ahead - pd.Timedelta(days=min(lags))
            is_self = table["subject"].to_numpy()
            for slug in config.all_slugs():
                values = (_at_time_offset(table, wide[slug], back)
                          if slug in wide.columns else np.full(len(table), np.nan))
                # A subject never mirrors itself: that column is already
                # tgt{h}h_lag{k}d, and two names for one number is how a tree
                # gets talked into splitting on it twice.
                new[cross_subject_lag_column(horizon, slug, min(lags))] = np.where(
                    is_self == slug, np.nan, values)

    return pd.concat([table, pd.DataFrame(new, index=table.index)], axis=1)


def _add_cross_subject(table: pd.DataFrame) -> pd.DataFrame:
    """The other people's state at the ORIGIN time.

    Origin, never target: at prediction time we know where everyone is now, and
    nothing about where they will be. Reading the partner's state at the target
    slot would be reading the answer.
    """
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
    """The earliest moment worth asking for.

    There is no fixed date any more -- the original hardcoded the day one
    particular InfluxDB got its first write. A store knows its own span; an
    Influx does not, so it gets a generous floor and returns what it has.
    """
    span = getattr(getattr(source, "store", None), "span", None)
    if span:
        first = span().get("first")
        if first:
            return first

    # An Influx does not carry its own span, so ask it for the earliest point
    # rather than guessing. See InfluxSource.first_seen for why guessing high is
    # expensive rather than merely untidy.
    first_seen = getattr(source, "first_seen", None)
    if first_seen:
        earliest = first_seen([s.entity_id for s in config.SUBJECTS if s.entity_id])
        if earliest:
            return earliest

    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=400)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def build(source, start: str | None = None, stop: str | None = None) -> pd.DataFrame:
    """Build the modelling table.

    Full rebuild every time: the whole history is a handful of range reads and
    an incremental version would be a correctness risk for no measurable saving.
    """
    config.require()
    start = start or history_start(source)
    start_ts = pd.Timestamp(start)
    stop_ts = pd.Timestamp(stop) if stop else pd.Timestamp.now(tz="UTC")
    slots = grid(start_ts, stop_ts)
    if len(slots) == 0:
        raise ValueError(f"empty grid for {start}..{stop_ts.isoformat()}")

    # One observability mask for everyone, from the trackers that report
    # continuously. See features.observability.
    observable = observability(_liveness(source, start, stop), slots)

    frames = {
        subject.slug: _subject_frame(source, subject, start, stop, slots, observable)
        for subject in config.SUBJECTS
    }

    # The house has no zone of its own -- `presence_events` collapses a group to
    # home/not_home -- so it gets the union over the people: "is ANYONE in this
    # zone", which is exactly the household reading the old per-person
    # `office_{slug}` columns carried. fmax rather than max so a person whose
    # slot is unobserved does not blank the whole household.
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

    # Placeholders for the companion-app sensors that nothing computes yet.
    # `next_alarm_h` is now filled in `_add_next_alarm` and only lands here on
    # a subject that has no alarm entity; the rest are still declared-and-empty.
    # See BUILT_NOT_SHIPPED.
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

    _, _, source = runtime.bootstrap()

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
