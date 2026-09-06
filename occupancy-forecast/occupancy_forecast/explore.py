"""Reading the add-on's own data back out, for the panel's Data tab.

Everything the model eats is on disk already -- the raw archive in
`/data/history.db`, the feature table in `/data/features.parquet`, the scores in
`/data/models/metrics.json` -- and until this module there was no way to look at
any of it without shelling into the container. The endpoints in `server.py` are
thin wrappers over the functions here, which is what makes them testable without
an HTTP client (the shipped image carries no httpx, and `test_server.py` says so).

Two rules hold everything here together.

**Nothing in this module computes a feature.** It calls `features.slot_fraction`,
`features.numeric_on_grid` and `train.features_for` and reports what they say.
The failure mode being avoided is specific and quiet: an explorer that derives
`home_frac` its own way shows a number the model never saw, and the page is then
worse than no page, because it is confidently wrong. `test_explore.py` asserts
the agreement rather than trusting this docstring.

**Nothing here reads a whole parquet.** `feature_inventory` reads the footer,
where pyarrow has already written per-column null counts and min/max; the series
endpoint reads three columns. The table is well over a thousand columns wide, so an
unqualified `pd.read_parquet` would turn a panel tab into a several-second stall
and a spike in RSS on a machine that is also running Home Assistant.

Missing data is an answer, not an error. Every function returns
`{"available": False, "reason": <sentence>}` rather than raising or 404ing: a
fresh install genuinely has no models and no feature table, that is the normal
state for the first ten days, and a 404 in the browser console reads as a bug in
the add-on rather than as the truthful "not yet".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, discover, evaluate, features, runtime
from .sources.ha import HEARTBEAT_ENTITY

# How far back an entity view reaches by default, and the most it will reach.
# The cap is not about the database -- it is about the response: 90 days of
# 30-minute slots is 4,320 points, which draws and transfers comfortably.
DEFAULT_DAYS = 7
MAX_DAYS = 90

# Raw transitions returned with an entity, at most. A person entity is a few
# hundred a week and a synthesised distance sensor is one per collection pass,
# so this only bites on the latter -- and there the TAIL is what anyone
# inspecting wants, so that is what gets kept.
MAX_EVENTS = 2000


def unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _clamp_days(days: int | None) -> int:
    if not days:
        return DEFAULT_DAYS
    return max(1, min(MAX_DAYS, int(days)))


# --- the raw archive ------------------------------------------------------

def _classify(entity_id: str, values: list[tuple[str, int]]) -> str:
    """What kind of series this is, from a peek at its commonest values.

    By shape rather than by name: matching on `sensor.*_distance` would be
    guessing at the user's naming, and a person who calls their proximity sensor
    something else would get the wrong chart. Every value parsing as a float is
    a fact about the data.

    The heartbeat is the one exception, and it is not a guess: this package
    writes that row itself, so it owns the name. It is every-value-"ok" and
    would otherwise be classified as presence, which would offer to chart the
    fraction of each slot the collector spent at home.
    """
    if entity_id == HEARTBEAT_ENTITY:
        return "heartbeat"
    if not values:
        return "other"
    try:
        for value, _ in values:
            float(value)
    except (TypeError, ValueError):
        return "presence"
    return "numeric"


def _role(entity_id: str, settings) -> str:
    """What the add-on uses this entity FOR.

    Derived from the settings and from `discover.synthetic_distance_entity`,
    never from matching on the entity id's text -- the same reason `_classify`
    looks at values. The one exception is the collector's own heartbeat, which
    this package writes itself and therefore does own the name of.
    """
    if entity_id == HEARTBEAT_ENTITY:
        return "heartbeat"
    if entity_id in settings.people:
        return "person"
    if entity_id == settings.house_entity:
        return "house"
    if entity_id in settings.zones:
        return "zone"
    for pair in settings.proximity.values():
        if not pair:
            continue
        if entity_id == (pair[0] if len(pair) > 0 else None):
            return "proximity"
        if entity_id == (pair[1] if len(pair) > 1 else None):
            return "direction"
    if entity_id in {discover.synthetic_distance_entity(s.slug) for s in config.PEOPLE}:
        return "synthetic-distance"
    return "untracked"


def archive_inventory(source, settings) -> dict:
    """What is in `/data/history.db`, entity by entity.

    `tracked` is the field this card exists for. An entity in the archive that
    nothing reads is invisible otherwise, and so is the opposite and worse case:
    a person configured on the Setup tab whose entity has never produced a row,
    where the model is quietly training on a household with a member missing.

    The collector's own heartbeat is left out. It is neither of those things --
    it is this package's bookkeeping, not a signal anybody configured -- and
    listing it did active harm: `tracked` is false for it, so it was drawn with
    an "unused" chip and it was the entity the "not read" count was counting,
    under a sentence explaining that such entities are left over from an earlier
    configuration. It is not left over and there is nothing to act on. The row
    is still in the archive and `entity_series` still serves it; `_classify` and
    `_role` still name it, so a direct call gets a truthful answer.
    """
    store = getattr(source, "store", None)
    if store is None:
        return unavailable("this installation reads its history from InfluxDB, "
                           "so there is no local archive to inspect")
    if settings is None:
        return unavailable("nothing is configured yet — pick at least one person "
                           "on the Setup tab")

    tracked = set(runtime.tracked_entities(settings))
    rows = store.inventory()
    entities = []
    for row in rows:
        entity_id = row["entity_id"]
        if entity_id == HEARTBEAT_ENTITY:
            continue
        entities.append({**row,
                         "kind": _classify(entity_id, store.value_counts(entity_id, limit=4)),
                         "role": _role(entity_id, settings),
                         "tracked": entity_id in tracked})

    # A configured entity with no rows at all cannot appear in a GROUP BY over a
    # table it is not in, and it is the single most useful thing this can report.
    seen = {row["entity_id"] for row in rows}
    for entity_id in sorted(tracked - seen):
        entities.append({"entity_id": entity_id, "rows": 0,
                         "first": None, "last": None,
                         "kind": "other", "role": _role(entity_id, settings),
                         "tracked": True})

    return {"available": True, "span": store.span(), "entities": entities}


def entity_series(source, settings, entity_id: str, days: int | None = None) -> dict:
    """One entity: what arrived, and what the feature builder makes of it.

    The two halves are the point. The raw transitions are what Home Assistant
    reported; the gridded series is `features.slot_fraction` over exactly those
    events, which is the number that reaches the model. Seeing them together is
    how a gap in the top half is recognisable as the blank slot it causes in the
    bottom one, rather than as a flat line somebody has to take on trust.
    """
    store = getattr(source, "store", None)
    if store is None:
        return unavailable("this installation reads its history from InfluxDB, "
                           "so there is no local archive to inspect")
    if entity_id not in store.entities():
        return unavailable(f"{entity_id} has never produced a row in the archive")

    days = _clamp_days(days)
    stop = pd.Timestamp.now(tz="UTC")
    start = stop - pd.Timedelta(days=days)
    slots = features.grid(start, stop)
    kind = _classify(entity_id, store.value_counts(entity_id, limit=4))

    # Seeded, so a slot at the very start of the window is not blank merely
    # because the last change happened before it. That is the same reason
    # `_subject_frame` seeds its own read.
    events = store.seeded_states(entity_id, start.isoformat(), stop.isoformat())
    raw_rows = len(events)
    truncated = raw_rows > MAX_EVENTS

    if kind == "numeric":
        pairs = [(when, float(value)) for when, value in events
                 if _is_float(value)]
        values = features.numeric_on_grid(pairs, slots, config.DISTANCE_STALE_MIN)
        # Metres to kilometres for a distance, matching `distance_km` -- the
        # column the model actually sees, so the axis agrees with the feature.
        unit = settings.units.get(entity_id) if settings else None
        if _is_distance(entity_id, settings):
            values = values / 1000.0
            unit = "km"
        gridded = [{"t": t.isoformat(), "v": _num(v), "coverage": None}
                   for t, v in zip(slots, values)]
        label = f"last reading carried onto each {config.GRID_MINUTES}-minute slot"
    else:
        frame = features.slot_fraction(events, slots, config.HOME_STATE)
        unit = None
        gridded = [{"t": t.isoformat(), "v": _num(v), "coverage": _num(c)}
                   for t, v, c in zip(slots, frame["frac"].to_numpy(),
                                      frame["coverage"].to_numpy())]
        label = (f"home_frac — the fraction of each {config.GRID_MINUTES}-minute "
                 f"slot spent {config.HOME_STATE}")

    series = np.array([g["v"] for g in gridded], dtype=float)
    return {
        "available": True,
        "entity_id": entity_id,
        "kind": kind,
        "role": _role(entity_id, settings) if settings else "untracked",
        "start": start.isoformat(),
        "stop": stop.isoformat(),
        "unit": unit,
        "raw_rows": raw_rows,
        "truncated": truncated,
        "events": [{"t": t, "v": v} for t, v in events[-MAX_EVENTS:]],
        "grid_minutes": config.GRID_MINUTES,
        "gridded": gridded,
        "gridded_label": label,
        "min_coverage": config.MIN_SLOT_COVERAGE,
        "summary": _summarise(series),
    }


def _is_float(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_distance(entity_id: str, settings) -> bool:
    """Whether this series is a distance in metres, and so should be shown in km.

    Both sources of one are asked by identity: the synthesised entity this
    package names itself, and the first half of a configured proximity pair.
    """
    if entity_id in {discover.synthetic_distance_entity(s.slug) for s in config.PEOPLE}:
        return True
    if settings is None:
        return False
    return any(pair and pair[0] == entity_id for pair in settings.proximity.values())


def _num(value) -> float | None:
    """NaN out to JSON as null. `float('nan')` is not valid JSON and FastAPI
    will happily emit a bare `NaN` that `JSON.parse` then rejects."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, 4)


# --- the feature table ----------------------------------------------------

# Columns worth offering as a chart: the origin block, which is the part a reader
# can name. Everything else is 48 parallel copies of the same target-relative
# ideas, one per horizon; those are summarised by family and never listed one by
# one, because it is 85 KB of JSON nobody reads and a thousand-entry dropdown is
# not a way to find anything. No count is written down here on purpose -- it
# moves whenever a family is added, and the panel derives it from `columns`
# minus `browsable` rather than being told.
BROWSABLE_FAMILIES = ("target", "state", "proximity", "calendar", "zone",
                      "cross_subject", "not_shipped")

# Points in a feature series before it is thinned. 400 days of 30-minute slots
# is 19,200 per subject, which is more resolution than a 1000px chart can draw.
MAX_POINTS = 4000

MAX_SERIES_DAYS = 400


def feature_inventory(path) -> dict:
    """What is in `/data/features.parquet`, by family, WITHOUT reading it.

    Parquet's footer already carries a per-row-group, per-column null count and
    min/max, and `features.write` does nothing to disable them. So the whole of
    this answer -- a thousand columns, their null fractions and their ranges -- comes
    out of the metadata. An unqualified `pd.read_parquet` here would be a
    several-second stall and 134 MiB of RSS on a box that is also running Home
    Assistant, every time somebody opened a tab.

    `test_explore.py` makes `pd.read_parquet` raise and asserts this still
    answers, which is what keeps that true under future edits.
    """
    import pyarrow.parquet as pq

    if not path.exists():
        return unavailable("there is no feature table yet — one is written by the "
                           "first training run")
    try:
        meta = pq.ParquetFile(path).metadata
    except Exception as err:  # noqa: BLE001
        return unavailable(f"the feature table could not be read: {err}")

    names = list(meta.schema.names)
    rows = meta.num_rows
    stats = _footer_stats(meta, names)

    families: dict[str, dict] = {}
    for name in names:
        family = features.column_family(name)
        entry = families.setdefault(family, {"columns": 0, "nulls": 0, "counted": 0})
        entry["columns"] += 1
        null = stats.get(name, {}).get("nulls")
        if null is not None:
            entry["nulls"] += null
            entry["counted"] += 1

    ordered = [f for f in features.FAMILIES if f in families]
    ordered += [f for f in families if f not in features.FAMILIES]

    return {
        "available": True,
        "path": str(path),
        "built_at": _mtime(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": len(names),
        "row_groups": meta.num_row_groups,
        # A row is one subject in one slot, and the slot length is this. The
        # panel used to print its own literal 30 beside the row count, which is
        # the sort of duplication that is right until the day it is not.
        "grid_minutes": config.GRID_MINUTES,
        # False when pyarrow wrote no column statistics. The families below are
        # read off the schema and are still right; only the null fractions and
        # ranges go missing. Degrading here is the point -- the alternative is
        # reading 134 MiB to fill in a percentage.
        "statistics": bool(stats),
        "families": [{
            "family": f,
            "words": features.FAMILY_WORDS.get(f, f),
            "columns": families[f]["columns"],
            "null_frac": (round(families[f]["nulls"] / (families[f]["counted"] * rows), 4)
                          if families[f]["counted"] and rows else None),
        } for f in ordered],
        "browsable": [{
            "name": name,
            "family": features.column_family(name),
            "null_frac": (round(stats[name]["nulls"] / rows, 4)
                          if rows and stats.get(name, {}).get("nulls") is not None
                          else None),
            "min": _num(stats.get(name, {}).get("min")),
            "max": _num(stats.get(name, {}).get("max")),
        } for name in names
            if features.column_family(name) in BROWSABLE_FAMILIES],
    }


def _footer_stats(meta, names: list[str]) -> dict[str, dict]:
    """Null counts and ranges, summed over the row groups. Footer only."""
    out: dict[str, dict] = {}
    for j, name in enumerate(names):
        nulls: int | None = 0
        lo = hi = None
        for g in range(meta.num_row_groups):
            st = meta.row_group(g).column(j).statistics
            if st is None:
                nulls = None
                break
            if st.null_count is not None and nulls is not None:
                nulls += st.null_count
            if st.has_min_max:
                lo = st.min if lo is None else min(lo, st.min)
                hi = st.max if hi is None else max(hi, st.max)
        if nulls is None and lo is None:
            continue
        out[name] = {"nulls": nulls, "min": lo, "max": hi}
    return out


def _mtime(path) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()


def feature_series(path, subject: str, column: str, days: int | None = None) -> dict:
    """One column of the feature table, for one subject, over time.

    Three columns of the parquet, never the table -- the same `columns=` trick
    `train.load` uses, and which the changelog records as three quarters of the
    training time when it was introduced there.
    """
    import pyarrow.parquet as pq

    if not path.exists():
        return unavailable("there is no feature table yet — one is written by the "
                           "first training run")
    if subject not in config.all_slugs():
        return unavailable(f"{subject} is not a subject in this installation")

    # Validated against the schema before it is handed to pyarrow. An
    # unchecked name here is an unvalidated string reaching a file reader, and
    # the error it produces is a stack trace rather than an answer.
    schema = pq.read_schema(path)
    if column not in schema.names:
        return unavailable(f"{column} is not a column in the feature table")

    days = max(1, min(MAX_SERIES_DAYS, int(days or 30)))
    frame = pd.read_parquet(path, columns=["time", "subject", column])
    frame = frame[frame["subject"] == subject]
    if frame.empty:
        return unavailable(f"the feature table holds no rows for {subject}")

    frame = frame.sort_values("time")
    cutoff = frame["time"].max() - pd.Timedelta(days=days)
    frame = frame[frame["time"] >= cutoff]

    # Thin rather than truncate: a window is more useful at lower resolution
    # than a window that silently stops early.
    thinned = False
    if len(frame) > MAX_POINTS:
        frame = frame.iloc[:: (len(frame) // MAX_POINTS) + 1]
        thinned = True

    values = frame[column].to_numpy(dtype=float)
    family = features.column_family(column)
    return {
        "available": True,
        "subject": subject,
        "column": column,
        "family": family,
        "words": features.FAMILY_WORDS.get(family, family),
        "grid_minutes": config.GRID_MINUTES,
        "points": [{"t": t.isoformat(), "v": _num(v)}
                   for t, v in zip(frame["time"], values)],
        "thinned": thinned,
        "safe_for": _lag_safety(column),
        "start": frame["time"].min().isoformat(),
        "stop": frame["time"].max().isoformat(),
        "summary": _summarise(values),
    }


def _lag_safety(column: str) -> dict | None:
    """Whether a daily-lag column may actually be used by its own horizon.

    `tgt{h}h_lag{k}d` is written into the parquet for EVERY horizon and is only
    valid where `24k >= h`. Charting one without saying so would present a
    column the model is forbidden to read as though it were a live feature,
    which is the most misleading thing this page could do.
    """
    if features.column_family(column) != "daily_lag":
        return None
    head, _, tail = column.partition("_")
    horizon = int(head[3:-1])
    days = int(tail[3:-1])
    safe = days in features.safe_daily_lags(horizon)
    return {
        "horizon_h": horizon, "days": days, "safe": safe,
        "why": None if safe else _lag_reason(horizon, days),
    }


def _lag_reason(horizon: int, days: int) -> str:
    return (f"home_frac {horizon - 24 * days} hours after the moment being "
            f"predicted, so it cannot be known at +{horizon} h")


# --- what one horizon uses ------------------------------------------------

def horizon_recipe(horizon: int, models: dict) -> dict:
    """Which columns horizon `h` actually fits on, and which it may not touch.

    Reads nothing from disk. It exists because the leakage gate in
    `features.safe_daily_lags` has always been correct and always been
    invisible: the lag columns sit in the table for every horizon, and only this
    says which of them the model at +36 h is allowed to look at.

    Calls into `train` rather than rebuilding the list, so the page cannot
    describe a recipe that has drifted from the one the pickle carries.

    **Which list depends on which family won this horizon.** A pooled model
    reads one list at every horizon, with `horizon_h` among the columns; a
    dedicated model reads that horizon's own, which is ~44 columns rather than
    ~1000. Reporting the pooled list unconditionally was describing the wrong
    model for every dedicated horizon. Where nothing is served there is no
    winner to describe, and the pooled list is the honest default -- it is the
    one the melt this card explains is built from.
    """
    from . import evaluate, train as train_mod

    if horizon not in config.HORIZONS_H:
        return unavailable(f"+{horizon} h is not one of the horizons this add-on serves")

    metrics = (models.get(horizon) or {}).get("metrics") or {}
    kind = metrics.get("kind")
    columns = (train_mod.features_for(horizon) if kind == "dedicated"
               else train_mod.base_features())
    counts: dict[str, int] = {}
    for name in columns:
        counts[features.column_family(name)] = (
            counts.get(features.column_family(name), 0) + 1)

    safe = features.safe_daily_lags(horizon)
    return {
        "available": True,
        "horizon_h": horizon,
        "target": features.TARGET_COLUMN,
        # Asked per horizon rather than read off the RESIDUAL_BASE constant.
        # It answers `state_now` every time today, and the card would be
        # identical either way -- but whether the anchor should vary with the
        # horizon is a live question (train.py records one measured attempt),
        # and this way the card follows the answer instead of restating a
        # constant.
        "residual_base": train_mod.residual_base(horizon),
        "n_features": len(columns),
        "features": columns,
        "families": [{"family": f, "words": features.FAMILY_WORDS.get(f, f),
                      "columns": counts[f]}
                     for f in features.FAMILIES if f in counts],
        "daily_lags": [{
            "days": k,
            # The long name: what the melt copies `tgt{h}h_lag{k}d` INTO, and
            # only for the lags this horizon is allowed. An unsafe lag has no
            # value on these rows at all rather than a value nobody reads.
            "column": f"lag{k}d",
            "safe": k in safe,
            "why": None if k in safe else _lag_reason(horizon, k),
        } for k in features.DAILY_LAGS],
        "climatology": f"wclim{features.CLIMATOLOGY_WEEKS}",
        # What this horizon's fit reads off the parquet. For a dedicated model
        # that is far fewer columns than it fits on is misleading only if the
        # two are conflated -- `columns_for` includes the keys and the target.
        "columns_read": (len(train_mod.columns_for(horizon))
                         if kind == "dedicated" else len(columns)),
        "embargo_hours": round(
            evaluate.embargo_for(horizon).total_seconds() / 3600, 2),
        # Three values, and the third is not the second: "none" means a model
        # was trained for this horizon and lost, `None` means none was ever
        # trained. Both publish nothing; only one of them has a bake-off to
        # report. `ships` below carries the same distinction as a bool|None.
        "served_by": ("model" if metrics.get("ships")
                      else "none" if metrics else None),
        "ships": bool(metrics.get("ships")) if metrics else None,
        # Which family's recipe the list above actually is.
        "kind": kind,
    }


# --- was it right? --------------------------------------------------------

def verification(source, settings, subject: str, horizon_h: int,
                 days: int | None = None) -> dict:
    """What was forecast for each slot at one horizon, against what happened.

    This is the only number the add-on reports that is about the SERVING path.
    Everything on the Judge step is rolling-origin cross-validation computed at
    training time over `features.parquet`: it answers "how would a model fitted
    on folds [0,k) have scored on fold k". It cannot see the nowcast pin, a
    sensor that went stale at 07:00, or the ship gate deciding this horizon is
    not worth publishing -- and after "model or nothing" that last one is the
    whole story, because an unserved horizon leaves a hole here rather than a
    plausible number.

    So the live Brier and the backtest Brier are different quantities and are
    meant to be read side by side. A large gap between them is a finding about
    the deployment, not a bug in this function.

    Truth comes from `features.presence_events` + `features.slot_fraction` --
    the same two calls `entity_series` makes and the same ones the feature
    builder makes. Deriving it any other way here would show a number the model
    never saw, which this module's docstring forbids for good reason. It also
    means the house is handled correctly: with no group configured its presence
    is the OR over the people, and `presence_events` already knows that.
    """
    if subject not in config.all_slugs():
        return unavailable(f"{subject} is not a subject in this installation")
    if horizon_h not in config.HORIZONS_H:
        return unavailable(f"+{horizon_h} h is not a horizon this add-on forecasts")

    store = getattr(source, "store", None)
    if store is None:
        return unavailable("this installation reads its history from InfluxDB, "
                           "so the add-on keeps no record of what it published")

    days = _clamp_days(days)
    stop = pd.Timestamp.now(tz="UTC")
    start = stop - pd.Timedelta(days=days)
    slots = features.grid(start, stop)
    if len(slots) == 0:
        return unavailable("the window is shorter than one slot")

    forecast_rows = store.forecast_series(subject, horizon_h, start.isoformat(),
                                          stop.isoformat())
    if not forecast_rows:
        return unavailable(
            f"nothing published for +{horizon_h} h has come due yet. The "
            f"add-on records each forecast as it is made, so this fills in "
            f"{horizon_h} h after it first serves this horizon -- and stays "
            f"empty for as long as the horizon goes unserved.")
    by_slot = dict(forecast_rows)

    events = features.presence_events(source, config.subject(subject),
                                      start.isoformat(), stop.isoformat())
    observed = features.slot_fraction(events, slots, config.HOME_STATE)

    actual = observed["frac"].to_numpy()
    # `slots` is tz-aware UTC and the table stores epoch ms, so this is the
    # equality join the write side was designed to make possible.
    forecast = np.array([by_slot.get(int(t.timestamp() * 1000), np.nan)
                         for t in slots], dtype=float)

    points = [{"t": t.isoformat(), "actual": _num(a), "forecast": _num(f)}
              for t, a, f in zip(slots, actual, forecast)]

    scores = evaluate.score(actual, forecast)
    served = int(np.count_nonzero(~np.isnan(forecast)))
    return {
        "available": True,
        "subject": subject,
        "horizon_h": horizon_h,
        "grid_minutes": config.GRID_MINUTES,
        "start": start.isoformat(),
        "stop": stop.isoformat(),
        "points": points,
        "slots": len(slots),
        "served": served,
        "scored": scores.n,
        "brier": _num(scores.brier),
        "mae": _num(scores.mae_frac),
        "retention_days": config.FORECAST_RETENTION_DAYS,
        "summary": _verification_summary(len(slots), served, scores),
    }


def _verification_summary(slots: int, served: int, scores) -> str:
    """One sentence, and it leads with the holes rather than the accuracy.

    A horizon that is only published a third of the time is a more important
    fact about it than its Brier over the third that was, and stating the score
    first invites reading it as the horizon's record when it is the record of
    its good days.
    """
    if served == 0:
        return ("nothing was published at this horizon over the window -- the "
                "model has not earned it, so the add-on said nothing")
    share = f"{100 * served / slots:.0f}%"
    head = (f"published for {served} of {slots} slots ({share})"
            if served < slots else f"published for all {slots} slots")
    if not scores.n:
        return f"{head}, and none of them has an observation to score against yet"
    return (f"{head}; over the {scores.n} with an observation to compare, "
            f"Brier {scores.brier:.3f} and mean error {scores.mae_frac:.3f}")


# --- model quality --------------------------------------------------------

# The scalars that fit in a list. `per_fold`, `reliability`, `baselines` and
# `fallback` are the bulky ones and are served only for the horizon being looked
# at -- all 48 with their curves attached is about 250 KB for a card that shows
# one of them at a time.
SCALARS = ("horizon_h", "brier", "log_loss", "auc", "mae_frac", "base_rate",
           "n_folds", "n_scored", "n_train_final", "best_baseline",
           "best_baseline_brier", "skill_vs_best_baseline_pct",
           "folds_beating_best_baseline", "sign_test_p", "ships",
           "brier_fold_min", "brier_fold_max",
           # Which family won and what the other one scored. Cheap, and without
           # them the two-family split is invisible everywhere except a
           # training log -- the table is the only place the crossover between
           # them can be read off.
           "kind", "rival_brier", "rival_kind")


def _read_summary(models_dir) -> dict | None:
    """`metrics.json`, or nothing.

    Preferred over unpickling the 48 artifacts: one JSON read, no scikit-learn
    import in the request path, and `write_summary` puts the same `asdict` in
    both places. A corrupt file is no answer rather than a reason to fail, which
    is how `train.last_summary` already treats it.
    """
    import json

    from . import train as train_mod

    path = train_mod.summary_path(models_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def metrics_summary(models_dir, models: dict | None = None) -> dict:
    """How every horizon scored, against the baseline it had to beat.

    All of this has been written to disk on every train since the beginning and
    none of it has ever been rendered -- the panel could say a horizon ships,
    but not by how much, nor how much the folds disagreed.
    """
    summary = _read_summary(models_dir)
    if summary is None:
        # The pickles carry their own copy, so a lost or truncated metrics.json
        # is not the end of the answer.
        if models:
            horizons = {str(h): (a.get("metrics") or {}) for h, a in models.items()}
            summary = {"model_version": None, "trained_at": None,
                       "duration_s": None, "evaluation": None,
                       "horizons": horizons, "failed": {}}
        else:
            return unavailable("nothing has been trained yet — no horizon is "
                               "published until it has been")

    rows = []
    for key, metrics in sorted(summary.get("horizons", {}).items(), key=lambda kv: int(kv[0])):
        if not metrics:
            continue
        row = {name: metrics.get(name) for name in SCALARS}
        row["horizon_h"] = int(row["horizon_h"] or key)
        rows.append(row)

    if not rows:
        return unavailable("nothing has been trained yet — no horizon is "
                           "published until it has been")

    return {
        "available": True,
        "trained_at": summary.get("trained_at"),
        "model_version": summary.get("model_version"),
        "evaluation": summary.get("evaluation"),
        "duration_s": summary.get("duration_s"),
        "shipping": sum(1 for r in rows if r["ships"]),
        "horizons": rows,
        "failed": summary.get("failed") or {},
    }


def metrics_detail(models_dir, horizon: int, models: dict | None = None) -> dict:
    """One horizon in full, including the two series nothing has ever drawn.

    `per_fold` is how much the folds disagreed -- a pooled Brier that beats its
    baseline on the strength of one lucky week is a different claim from one
    that beats it in eleven weeks out of fifteen. `reliability` is a ten-bin
    calibration curve, and calibration is the property this add-on actually
    needs: a model can rank perfectly and still say 0.9 when it means 0.6, which
    for "pre-heat if they will be home" is a wasted hour of gas.
    """
    summary = _read_summary(models_dir)
    metrics = None
    if summary:
        metrics = (summary.get("horizons") or {}).get(str(horizon))
    if not metrics and models:
        metrics = (models.get(horizon) or {}).get("metrics")
    if not metrics:
        return unavailable(f"+{horizon} h has not been trained yet")

    return {"available": True, "horizon_h": horizon, **metrics}


def _summarise(values: np.ndarray) -> dict:
    finite = values[~np.isnan(values)] if values.size else values
    return {
        "n": int(values.size),
        "nulls": int(values.size - finite.size),
        "min": _num(finite.min()) if finite.size else None,
        "max": _num(finite.max()) if finite.size else None,
        "mean": _num(finite.mean()) if finite.size else None,
        "last": _num(finite[-1]) if finite.size else None,
    }
