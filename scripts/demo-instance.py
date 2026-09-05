#!/usr/bin/env python3
"""Build a throwaway `/data` for a household that does not exist, for screenshots.

Every screenshot of this add-on is a picture of somebody's home: who lives
there, where they work, when the house is empty, and the entity ids naming all
three. The panel's Config view renders `person.*` and `zone.*` ids verbatim and
the Data view lists every archived series, so redacting a capture afterwards
means catching every one of them in an image editor.

This does the opposite: it runs the REAL pipeline against a household that was
generated, so there is nothing to redact. `occupancy_forecast/tests/synthetic.py`
already builds one -- Alice and Bob, with holidays, a mid-history routine change
and partner coupling -- and it exists for exactly this reason: "the whole point
of a synthetic household here is that it belongs to nobody, which is what stops
one particular installation creeping into the package."

**The numbers on screen are not invented.** This writes a state history, then
trains on it and backtests the trained model to fill the forecast record. So the
curves, the ship gate, the fold bars and the verification card all show what the
add-on actually does with that history. Only the household is fictional.

    scripts/demo-instance.py build      --out ~/occupancy-demo/data
    python -m occupancy_forecast.train                     # against the same /data
    scripts/demo-instance.py forecasts  --out ~/occupancy-demo/data

Run OUTSIDE an add-on, where `config.topic_prefix()` falls back to
`DEFAULT_TOPIC_PREFIX` -- so the panel titles itself "Occupancy Forecast" with no
build suffix, which is what a release screenshot wants, and no Supervisor, broker
or Home Assistant is involved at any point.
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

# Two people and two workplaces, so the Config view has a list worth showing and
# `out_columns()` has more than one zone to separate. The names come from
# `conftest.settings()`, which is the package's own synthetic installation.
PEOPLE = {"alice": "person.alice", "bob": "person.bob"}
ZONES = {"alice": ("zone.office", "Office"), "bob": ("zone.workshop", "Workshop")}
HOUSE_ENTITY = "group.household"

# Metres from home to each workplace. Only the shape matters -- `eta.py` reads
# the distance TRACE, not the number, so what it needs is a closing speed that
# looks like a commute rather than a step change.
COMMUTE_M = {"alice": 12_400, "bob": 8_100}
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
        timezone="Europe/Amsterdam",
        country="NL",
        holiday_country="NL",
        home_latitude=52.09, home_longitude=5.12,
    )


def _ms(when: pd.Timestamp) -> int:
    return int(when.value // 1_000_000)


def state_events(frame: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """Per person, the `person.*` state stream the store would have recorded.

    Home Assistant writes a person's state as `home`, `not_home`, or the FRIENDLY
    NAME of the zone they are in -- which is the only per-person zone signal the
    history contains, and what `features._resolve_zone_events` decodes. So the
    generator's `home_frac` and `zone_work` have to come back out in that shape
    rather than as three separate columns.

    Emitted on CHANGE only, like a real recorder. Carry-forward then does the
    rest, and `seeded_states` handles a window that opens mid-episode.
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
                # nothing is what makes `MAX_SILENCE_H` mean something, and the
                # generator puts holes in on purpose.
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
                  who: str) -> list[tuple[str, int, str]]:
    """A metres-from-home trace with commute-shaped ramps between episodes.

    `eta.py` refuses to answer below `MIN_CLOSING_KMH`, so a trace that steps
    from 12 km to 0 in one sample would train and serve nothing. The ramp is what
    makes "minutes until home" a question the model can be asked.
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
        # Then a sparse hold until the next change, the way a parked phone reports.
        stop = events[index + 1][0] if index + 1 < len(events) else when + pd.Timedelta(hours=6)
        cursor = when + pd.Timedelta(minutes=COMMUTE_MIN + IDLE_STEP_MIN)
        while cursor < stop:
            at(cursor, target)
            cursor += pd.Timedelta(minutes=IDLE_STEP_MIN)
    return rows


def build(out: Path, days: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    conf = settings()
    config.configure(conf)
    conf.save(out / "config.json")

    # End the generated history today, so "the last 48 hours" on the panel is the
    # last 48 hours and the verification card has something recent to score.
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    frame = synthetic.household(days=days, seed=seed, realistic=True, start=start)
    print(f"generated {len(frame)} slot-rows for {frame['subject'].nunique()} people "
          f"from {start} over {days} days")

    events = state_events(frame)
    rows: list[tuple[str, int, str]] = []
    for who, stream in events.items():
        entity = PEOPLE[who]
        rows += [(entity, _ms(when), state) for when, state in stream]
        rows += distance_rows(stream, who)
        print(f"  {who}: {len(stream)} state changes")

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
    """Feature table, both model families, the ETA and the out routine.

    The same four steps `server._retrain` runs, in the same order, minus
    `runtime.bootstrap` -- which reaches for Home Assistant to re-read timezone,
    country and the home's COORDINATES. Calling it here would write this
    household's real latitude and longitude into the demo's config.json and put
    them on the Config screenshot, which is the whole thing this script exists to
    avoid.
    """
    conf = config.Settings.load(out / "config.json")
    if conf is None:
        raise SystemExit(f"no config.json in {out} -- run `build` first")
    config.configure(conf)
    store = HistoryStore(out / "history.db")

    began = dt.datetime.now()
    table = features.build(store)
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

    A real backtest rather than plausible noise: the forecast for `t + h` is what
    THIS model says given the features at `t`, so the card scores the add-on and
    not a random number generator. Batched per horizon -- one predict over every
    origin at once -- because doing it a row at a time is ~1400 model calls a
    subject and takes minutes for no different answer.
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
    parser.add_argument("--days", type=int, default=400,
                        help="history to generate (build) / to backtest (forecasts)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=6,
                        help="joblib workers; pair with OMP_NUM_THREADS=1")
    args = parser.parse_args()
    if args.step == "build":
        build(args.out, args.days, args.seed)
    elif args.step == "fit":
        fit(args.out, args.n_jobs)
    else:
        forecasts(args.out, min(args.days, config.FORECAST_RETENTION_DAYS))


if __name__ == "__main__":
    main()
