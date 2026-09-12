"""The Data tab's view of the add-on's own data; `server.py` wraps it thinly.

Nothing here computes a feature: it calls `features`/`train` and reports what
they say, since an explorer deriving `home_frac` its own way is worse than no
page. Nothing reads a whole parquet; missing data is `unavailable(reason)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, discover, evaluate, features, log, runtime
from .sources.ha import HEARTBEAT_ENTITY

_log = log.get(__name__)

# The cap is about the response, not the database.
DEFAULT_DAYS = 7
MAX_DAYS = 90

# Raw transitions per entity; the TAIL is what anyone inspecting wants, so
# that is what gets kept.
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

    By shape, not name: matching `sensor.*_distance` would guess at the user's
    naming. The heartbeat is the exception, because this package owns its name.
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
    """What the add-on uses this entity FOR: from settings, never its name."""
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
    """What is in `/data/history.db`, entity by entity, bar our own heartbeat.

    `tracked` is the point: archived but unread, or configured but never seen.
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

    Together, a gap in the raw transitions shows as the blank slot it causes.
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

    # Seeded, so a slot at the start of the window is not blank merely because
    # the last change happened before it.
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

    Asked by identity: the synthesised entity, or a proximity pair's first half.
    """
    if entity_id in {discover.synthetic_distance_entity(s.slug) for s in config.PEOPLE}:
        return True
    if settings is None:
        return False
    return any(pair and pair[0] == entity_id for pair in settings.proximity.values())


def _num(value) -> float | None:
    """NaN out to JSON as null: FastAPI renders with `allow_nan=False`, so a
    NaN reaching a handler's return value is a 500, not a bare `NaN`."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, 4)


def _json_safe(value):
    """`value` with every NaN (and infinity) replaced by None, recursively.

    `metrics.json` is written with `allow_nan=True`, and a NaN back out through
    FastAPI is a 500; cleaned here so an artifact already on disk still serves.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# --- the feature table ----------------------------------------------------

# Chartable: the origin block, the part a reader can name. The rest are 48
# per-horizon copies, summarised by family rather than listed.
BROWSABLE_FAMILIES = ("target", "state", "proximity", "calendar", "zone",
                      "cross_subject", "not_shipped")

# Points in a feature series before it is thinned: more than a chart can draw.
MAX_POINTS = 4000

MAX_SERIES_DAYS = 400


def feature_inventory(path) -> dict:
    """What is in `/data/features.parquet`, by family, from its footer ONLY.

    A `pd.read_parquet` here would stall the box that also runs Home Assistant.
    """
    import pyarrow.parquet as pq

    if not path.exists():
        return unavailable("there is no feature table yet — one is written by the "
                           "first training run")
    try:
        meta = pq.ParquetFile(path).metadata
    except Exception as err:  # noqa: BLE001
        # pyarrow's error names the path, and this endpoint needs no login.
        _log.warning("the feature table could not be read: %s", err, exc_info=True)
        return unavailable("the feature table could not be read — the add-on log "
                           "has the error")

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
        # A row is one subject in one slot, and the slot length is this.
        "grid_minutes": config.GRID_MINUTES,
        # False when pyarrow wrote no column statistics: the families still come
        # off the schema, and only null fractions and ranges go missing.
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

    Three columns of the parquet, never the table.
    """
    import pyarrow.parquet as pq

    if not path.exists():
        return unavailable("there is no feature table yet — one is written by the "
                           "first training run")
    if subject not in config.all_slugs():
        return unavailable(f"{subject} is not a subject in this installation")

    # Checked against the schema first: an unchecked name is an unvalidated
    # string reaching a file reader.
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
    """Whether daily lag `tgt{h}h_lag{k}d` is legal for its horizon: `24k >= h`.

    It exists for EVERY horizon, so unmarked it would pass for a live feature.
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

    Calls `train` and reads no disk, so it cannot drift from the pickle's recipe.
    The list is the winning family's; with nothing served, the pooled one.
    """
    from . import train as train_mod

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
        # Asked per horizon rather than read off RESIDUAL_BASE, so the card
        # follows the answer instead of restating a constant.
        "residual_base": train_mod.residual_base(horizon),
        "n_features": len(columns),
        "features": columns,
        "families": [{"family": f, "words": features.FAMILY_WORDS.get(f, f),
                      "columns": counts[f]}
                     for f in features.FAMILIES if f in counts],
        "daily_lags": [{
            "days": k,
            # The long name: what the melt copies `tgt{h}h_lag{k}d` INTO, and
            # only for the lags this horizon is allowed.
            "column": f"lag{k}d",
            "safe": k in safe,
            "why": None if k in safe else _lag_reason(horizon, k),
        } for k in features.DAILY_LAGS],
        "climatology": f"wclim{features.CLIMATOLOGY_WEEKS}",
        # What this horizon's fit reads off the parquet: for a dedicated model,
        # `columns_for` adds the keys and the target to what it fits on.
        "columns_read": (len(train_mod.columns_for(horizon))
                         if kind == "dedicated" else len(columns)),
        "embargo_hours": round(
            evaluate.embargo_for(horizon).total_seconds() / 3600, 2),
        # Three values: "none" means a model was trained and lost, `None` means
        # none was ever trained; `ships` carries the same as a bool|None.
        "served_by": ("model" if metrics.get("ships")
                      else "none" if metrics else None),
        "ships": bool(metrics.get("ships")) if metrics else None,
        # Which family's recipe the list above actually is.
        "kind": kind,
    }


# --- was it right? --------------------------------------------------------

def verification(source, log, settings, subject: str, horizon_h: int,
                 days: int | None = None) -> dict:
    """What was forecast for each slot at one horizon, against what happened.

    The only SERVING-path score: cross-validation cannot see the nowcast pin, a
    stale sensor or the ship gate, so a gap between the two Briers is a finding.
    Truth is `presence_events` + `slot_fraction`, as the feature builder makes it.
    `log` is its own argument because an Influx install has no `source.store`.
    """
    if subject not in config.all_slugs():
        return unavailable(f"{subject} is not a subject in this installation")
    if horizon_h not in config.HORIZONS_H:
        return unavailable(f"+{horizon_h} h is not a horizon this add-on forecasts")

    if log is None:
        return unavailable("the add-on has not finished starting up, so there "
                           "is no record of what it published yet")

    days = _clamp_days(days)
    stop = pd.Timestamp.now(tz="UTC")
    start = stop - pd.Timedelta(days=days)
    slots = features.grid(start, stop)
    if len(slots) == 0:
        return unavailable("the window is shorter than one slot")

    forecast_rows = log.forecast_series(subject, horizon_h, start.isoformat(),
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
        "retention_days": settings.forecast_retention_days,
        "summary": _verification_summary(len(slots), served, scores),
    }


def _verification_summary(slots: int, served: int, scores) -> str:
    """One sentence, and it leads with the holes rather than the accuracy.

    A horizon published a third of the time matters more than its Brier there.
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

# Scalars only: `per_fold`, `reliability`, `baselines` and `fallback` are
# served just for the horizon being looked at.
SCALARS = ("horizon_h", "brier", "log_loss", "auc", "mae_frac", "base_rate",
           "n_folds", "n_scored", "n_train_final", "best_baseline",
           "best_baseline_brier", "skill_vs_best_baseline_pct",
           "folds_beating_best_baseline", "sign_test_p", "ships",
           "brier_fold_min", "brier_fold_max",
           # Which family won and what the other scored; without them the
           # two-family split is invisible outside a training log.
           "kind", "rival_brier", "rival_kind")


def _read_summary(models_dir) -> dict | None:
    """`metrics.json`, or None if it is missing or corrupt.

    Not the 48 pickles: one JSON read, and no scikit-learn in the request path.
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
    """How every horizon scored, against the baseline it had to beat."""
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
        row = _json_safe({name: metrics.get(name) for name in SCALARS})
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
    """One horizon in full, with its per-fold scores and calibration curve."""
    summary = _read_summary(models_dir)
    metrics = None
    if summary:
        metrics = (summary.get("horizons") or {}).get(str(horizon))
    if not metrics and models:
        metrics = (models.get(horizon) or {}).get("metrics")
    if not metrics:
        return unavailable(f"+{horizon} h has not been trained yet")

    # `per_fold` is the usual carrier: an empty fold is padded with NaN scores
    # by design (see test_an_empty_fold_reaches_the_panel_as_a_null_and_not_a_zero).
    return {"available": True, "horizon_h": horizon, **_json_safe(metrics)}


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
