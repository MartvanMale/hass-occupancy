#!/usr/bin/env python3
"""Build a throwaway `/data` for a household that does not exist, for screenshots.

The panel renders person and zone ids verbatim, so this runs the REAL pipeline
against `tests/synthetic.py` instead -- nothing to redact. Commands are in
docs/demo-instance.md. Run OUTSIDE an add-on, so `config.topic_prefix()` falls
back to the unsuffixed name.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "occupancy-forecast-edge"))

from occupancy_forecast import config, departure, features, train  # noqa: E402
from occupancy_forecast import eta as eta_mod, outing as outing_mod  # noqa: E402
from occupancy_forecast.sources import HistoryStore  # noqa: E402
from occupancy_forecast.tests import synthetic  # noqa: E402

# Two people and two workplaces, so `out_columns()` has more than one zone to
# separate. Names from `conftest.settings()`.
PEOPLE = {"alice": "person.alice", "bob": "person.bob"}
ZONES = {"alice": ("zone.office", "Office"), "bob": ("zone.workshop", "Workshop")}
HOUSE_ENTITY = "group.household"

# Only the shape matters: `eta.py` reads the distance TRACE, so it needs a
# closing speed rather than a step change.
COMMUTE_M = {"alice": 12_400, "bob": 8_100}
# How faithfully the alarm tracks the day it precedes. 0 says nothing the
# weekday median does not; 1 is an oracle.
ALARM_FIDELITY = 0.8

COMMUTE_MIN = 24              # door to door
COMMUTE_STEP_MIN = 4          # points along the ramp: ~31 km/h closing
IDLE_STEP_MIN = 120           # while parked, matching the real trace's long gaps


def settings() -> config.Settings:
    return config.Settings(
        people=list(PEOPLE.values()),
        zones=[z for z, _ in ZONES.values()],
        zone_names={z: n for z, n in ZONES.values()},
        house_entity=HOUSE_ENTITY,
        proximity={
            PEOPLE[who]: [f"sensor.home_{who}_distance",
                          f"sensor.home_{who}_direction_of_travel"]
            for who in PEOPLE},
        units={f"sensor.home_{who}_distance": "m" for who in PEOPLE},
        next_alarm={PEOPLE[who]: f"sensor.phone_{who}_next_alarm"
                    for who in PEOPLE},
        # Decoration only -- no feature, no model, no entity. demo-serve.py must
        # name the same entity.
        day_schedule="schedule.household_day",
        timezone="Europe/Amsterdam",
        country="NL",
        holiday_country="NL",
        home_latitude=52.09, home_longitude=5.12,
    )


def _ms(when: pd.Timestamp) -> int:
    return int(when.value // 1_000_000)


def state_events(frame: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """Per person, the `person.*` state stream the store would have recorded.

    HA writes a person's state as `home`, `not_home` or the zone's FRIENDLY NAME,
    which is what `features._resolve_zone_events` decodes. Emitted on CHANGE
    only, like a real recorder.
    """
    out: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for who, part in frame.groupby("subject", sort=False):
        part = part.sort_values("time")
        zone_name = ZONES[who][1]
        events: list[tuple[pd.Timestamp, str]] = []
        previous = None
        for when, home, work in zip(part["time"], part["home_frac"],
                                    part.get("zone_work", pd.Series(0, index=part.index))):
            if pd.isna(home):
                # A hole in the generated history is a hole here too: emitting
                # nothing is what makes `MAX_SILENCE_H` mean something.
                continue
            if home >= 0.5:
                state = config.HOME_STATE
            elif work and work > 0:
                state = zone_name
            else:
                state = "not_home"
            if state != previous:
                events.append((when, state))
                previous = state
        out[who] = events
    return out


def distance_rows(events: list[tuple[pd.Timestamp, str]],
                  who: str, until: pd.Timestamp) -> list[tuple[str, int, str]]:
    """A metres-from-home trace with commute-shaped ramps.

    `eta.py` refuses below `MIN_CLOSING_KMH`, so a trace that steps from 12 km
    to 0 would train and serve nothing.
    """
    entity = f"sensor.home_{who}_distance"
    far = COMMUTE_M[who]
    rows: list[tuple[str, int, str]] = []

    def at(when: pd.Timestamp, metres: float) -> None:
        rows.append((entity, _ms(when), f"{max(0.0, metres):.0f}"))

    for index, (when, state) in enumerate(events):
        target = 0.0 if state == config.HOME_STATE else (
            far if state == ZONES[who][1] else far * 0.45)
        previous = 0.0 if index == 0 else (
            0.0 if events[index - 1][1] == config.HOME_STATE else
            far if events[index - 1][1] == ZONES[who][1] else far * 0.45)
        # The journey: interpolate over COMMUTE_MIN so a closing speed exists.
        steps = max(1, COMMUTE_MIN // COMMUTE_STEP_MIN)
        for step in range(steps + 1):
            at(when + pd.Timedelta(minutes=step * COMMUTE_STEP_MIN),
               previous + (target - previous) * step / steps)
        # Sparse hold until the next change, the way a parked phone reports. `until`
        # for the LAST event, or the gap exceeds `MAX_SILENCE_H` and those slots
        # never become backtest origins.
        stop = events[index + 1][0] if index + 1 < len(events) else until
        cursor = when + pd.Timedelta(minutes=COMMUTE_MIN + IDLE_STEP_MIN)
        while cursor < stop:
            at(cursor, target)
            cursor += pd.Timedelta(minutes=IDLE_STEP_MIN)
    return rows


def zone_rows(events: dict[str, list[tuple[pd.Timestamp, str]]],
              until: pd.Timestamp) -> list[tuple[str, int, str]]:
    """Each tracked zone's own history, as a COUNT of persons inside.

    Nothing in the model reads it, but `runtime.tracked_entities` includes the
    zones and the Data view lists every tracked entity.
    """
    moments = sorted({when for stream in events.values() for when, _ in stream})
    state: dict[str, str] = {}
    per_person = {who: dict(stream) for who, stream in events.items()}
    rows: list[tuple[str, int, str]] = []
    last: dict[str, str] = {}
    for when in moments:
        for who in per_person:
            if when in per_person[who]:
                state[who] = per_person[who][when]
        for entity, name in ZONES.values():
            count = str(sum(1 for v in state.values() if v == name))
            if last.get(entity) != count:
                rows.append((entity, _ms(when), count))
                last[entity] = count
    return rows


def direction_rows(distances: list[tuple[str, int, str]],
                   who: str) -> list[tuple[str, int, str]]:
    """`towards` / `away_from` / `stationary`, from the sign of the distance trace.

    The demo's settings NAME a direction entity, so `features._add_proximity`
    prefers it and the sign fallback never runs.
    """
    entity = f"sensor.home_{who}_direction_of_travel"
    rows: list[tuple[str, int, str]] = []
    previous: float | None = None
    last = ""
    for _, when, value in distances:
        metres = float(value)
        if previous is not None:
            if metres <= 0.0 < previous:
                state = "arrived"
            elif metres < previous - 50:
                state = "towards"
            elif metres > previous + 50:
                state = "away_from"
            else:
                state = "stationary"
            # Only on change: the real sensor is event-driven, and writing a row
            # per sample would triple the archive for no extra information.
            if state != last:
                rows.append((entity, when, state))
                last = state
        previous = metres
    return rows


def alarm_rows(frame: pd.DataFrame, who: str) -> list[tuple[str, int, str]]:
    """The phone's next-alarm sensor: an ISO timestamp while set, `absent` after.

    The literal `absent` matters -- `_add_next_alarm` carries the last value
    forward, so a cancelled alarm that is merely omitted is carried forever.
    """
    entity = f"sensor.phone_{who}_next_alarm"
    person = frame[frame["subject"] == who].sort_values("time")
    if "next_alarm_h" not in person.columns:
        return []
    rows: list[tuple[str, int, str]] = []
    last = ""
    for when, ahead in zip(person["time"], person["next_alarm_h"]):
        if pd.isna(ahead):
            value = "absent"
        else:
            # Quantised to the minute, the way a phone reports a set alarm.
            value = (when + pd.Timedelta(hours=float(ahead))
                     ).floor("min").tz_convert("UTC").isoformat()
        if value != last:
            rows.append((entity, _ms(when), value))
            last = value
    return rows


def build(out: Path, days: int, seed: int, irregular: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    conf = settings()
    config.configure(conf)
    conf.save(out / "config.json")

    # End the generated history today, so "the last 48 hours" on the panel is the
    # last 48 hours and the verification card has something recent to score.
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    # `irregular`: without it this household is a timetable, the per-weekday
    # lookup is optimal by construction, and the panel's headline number is a
    # picture of the generator.
    frame = synthetic.household(days=days, seed=seed, realistic=True,
                                start=start, irregular=irregular,
                                alarm_fidelity=ALARM_FIDELITY)
    print(f"generated {len(frame)} slot-rows for {frame['subject'].nunique()} people "
          f"from {start} over {days} days"
          f"{'' if irregular else ' (timetable household)'}")

    events = state_events(frame)
    rows: list[tuple[str, int, str]] = []
    for who, stream in events.items():
        entity = PEOPLE[who]
        rows += [(entity, _ms(when), state) for when, state in stream]
        # Right up to the present, so the newest slots are observable and the
        # panel has a live "right now" rather than a three-day-old one.
        distances = distance_rows(stream, who, pd.Timestamp.now(tz="UTC"))
        rows += distances
        rows += direction_rows(distances, who)
        rows += alarm_rows(frame, who)
        print(f"  {who}: {len(stream)} state changes")

    rows += zone_rows(events, pd.Timestamp.now(tz="UTC"))

    # The house group, so `group.household` has a history of its own rather than
    # only being inferred. Anyone home -> home.
    per_person = {who: dict(stream) for who, stream in events.items()}
    moments = sorted({when for stream in events.values() for when, _ in stream})
    latest: dict[str, str] = {}
    previous = None
    for when in moments:
        for who in per_person:
            if when in per_person[who]:
                latest[who] = per_person[who][when]
        state = (config.HOME_STATE
                 if any(v == config.HOME_STATE for v in latest.values()) else "not_home")
        if state != previous:
            rows.append((HOUSE_ENTITY, _ms(when), state))
            previous = state

    store = HistoryStore(out / "history.db")
    written = store.append(rows)
    print(f"wrote {written} state rows to {out / 'history.db'}")
    for item in store.inventory():
        print(f"  {item['entity_id']:38s} {item['rows']:6d}  "
              f"{item['first'][:10]} -> {item['last'][:10]}")
    store.close()


def fit(out: Path, n_jobs: int) -> None:
    """Feature table, both families, the ETA and the out routine.

    The same four steps `server._retrain` runs, minus `runtime.bootstrap`, which
    would write this household's real coordinates into the demo's config.json.
    """
    conf = config.Settings.load(out / "config.json")
    if conf is None:
        raise SystemExit(f"no config.json in {out} -- run `build` first")
    config.configure(conf)
    store = HistoryStore(out / "history.db")

    began = dt.datetime.now()
    # The span explicitly: `features.history_start` looks for `source.store` and
    # this passes the HistoryStore itself, so it falls back to a 400-day floor.
    table = features.build(store, start=store.span()["first"])
    features.write(table, out / "features.parquet")
    labelled = int(table["home_frac"].notna().sum())
    print(f"features: {len(table)} rows, {labelled} labelled "
          f"({100 * labelled / max(len(table), 1):.1f}%), {len(table.columns)} columns "
          f"in {(dt.datetime.now() - began).total_seconds():.0f}s")

    summary = train.train_all(out / "features.parquet", out / "models",
                              n_jobs=n_jobs)
    ships = [h for h in config.HORIZONS_H
             if (summary.get(str(h)) or {}).get("ships")]
    print(f"occupancy: {len(ships)}/{len(config.HORIZONS_H)} horizons ship")

    eta_summary = eta_mod.train_all(store, out / "models")
    print("eta:", {k: (v or {}).get("ships") for k, v in (eta_summary or {}).items()})

    routine = outing_mod.fit_routine(
        outing_mod.label_out_days(table, departure.label_days(table)))
    outing_mod.save_routine(routine, out / "models")
    print(f"out routine: fitted for {len(routine)} person(s)")
    store.close()


def forecasts(out: Path, days: int) -> None:
    """Backtest the trained model and write the record the verification card reads.

    A real backtest, batched per horizon -- a row at a time is ~1400 model calls
    a subject for no different answer.
    """
    conf = config.Settings.load(out / "config.json")
    if conf is None:
        raise SystemExit(f"no config.json in {out} -- run `build` first")
    config.configure(conf)

    models = __import__("occupancy_forecast.predict", fromlist=["x"]).load_models(out / "models")
    if not models:
        raise SystemExit(f"no models in {out / 'models'} -- run the trainer first")

    store = HistoryStore(out / "history.db")
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days + features.deepest_lookback_days() + 3)
    table = features.build(store, start=start.strftime("%Y-%m-%dT%H:%M:%SZ"))
    window = end - pd.Timedelta(days=days)
    table = table[table["time"] >= window].dropna(subset=["state_now"])
    print(f"backtesting {len(table)} origin rows over the last {days} days")

    shipping = [h for h in config.HORIZONS_H
                if (models.get(h) or {}).get("metrics", {}).get("ships")]
    pooled = [h for h in shipping if models[h].get("kind") == "pooled"]
    dedicated = [h for h in shipping if models[h].get("kind") != "pooled"]
    print(f"  {len(shipping)} shipping horizons: {len(dedicated)} dedicated, "
          f"{len(pooled)} pooled")

    rows: list[tuple[str, int, int, float]] = []
    for subject, part in table.groupby("subject", sort=False):
        part = part.sort_values("time")
        if pooled:
            long = features.long_frame(part, horizons=tuple(pooled))
            values = train.predict_pooled(models[pooled[0]]["model"], long)
            for when, horizon, p in zip(long["time"], long[features.HORIZON_COLUMN],
                                        values):
                if not np.isnan(p):
                    rows.append((subject,
                                 _ms(pd.Timestamp(when) + pd.Timedelta(hours=int(horizon))),
                                 int(horizon), float(np.clip(p, 0, 1))))
        for horizon in dedicated:
            values = train.predict_dedicated(models[horizon]["model"], part, horizon)
            for when, p in zip(part["time"], values):
                if not np.isnan(p):
                    rows.append((subject,
                                 _ms(pd.Timestamp(when) + pd.Timedelta(hours=horizon)),
                                 horizon, float(np.clip(p, 0, 1))))
        print(f"  {subject}: {len(part)} origins")

    written = store.append_forecasts(rows)
    print(f"wrote {written} forecast rows ({len(rows)} offered)")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["build", "fit", "forecasts"])
    parser.add_argument("--out", type=Path, required=True)
    # 180, not 400: folds scale with history, and at 51 folds the sign test can
    # refuse a horizon with real skill.
    parser.add_argument("--days", type=int, default=180,
                        help="history to generate (build) / to backtest (forecasts)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=6,
                        help="joblib workers; pair with OMP_NUM_THREADS=1")
    parser.add_argument("--irregular", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="a household rather than a timetable (build only)")
    args = parser.parse_args()
    if args.step == "build":
        build(args.out, args.days, args.seed, args.irregular)
    elif args.step == "fit":
        fit(args.out, args.n_jobs)
    else:
        forecasts(args.out, min(args.days, config.FORECAST_RETENTION_DAYS))


if __name__ == "__main__":
    main()
