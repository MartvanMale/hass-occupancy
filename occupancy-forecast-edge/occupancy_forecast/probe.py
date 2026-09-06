"""Price a candidate feature against a control, honestly.

**Why this exists rather than "edit, retrain, compare".** Both `train.read_wide`
and `train.load_for` refuse a parquet that lacks a named feature, so the naive
experiment is edit -> rebuild -> train -> compare, and the two arms then differ
by a rebuild: a different newest row, possibly a different fold edge.
`train.shared_windows` exists because a comparison cut two ways is not a
comparison. So every candidate is BUILT into the parquet always
(`features.SHIPPED_EXTRAS`) and this flips which are SERVED, in-process, over
one table and one set of windows.

**Why the metrics here are not the ship gate's.** The gate scores a pooled Brier
over all 48 horizons, and MEASURED on this household only **1.2% of rows** sit in
an hour where the occupancy climatology moves by 0.30 or more. A change that
repairs those rows perfectly moves the pooled number by a few thousandths --
well under `min_ship_skill_pct`. The gate is the right instrument for "does this
model beat its baseline" and the wrong one for "did the timing get sharper".

Two metrics instead, one pre-registered as primary:

  1. **Departure-hour error**, through the SHIPPED reduction rule. The published
     `sensor.*_hours_until_away` is `predict._crossing` walking the curve, so a
     three-hour smear is a three-hour error in the number a person reads. This
     reuses that function rather than inventing a timing metric.
  2. **Transition-slot Brier**, stratified per fold from that fold's TRAINING
     rows only.

Never writes to `config.FEATURES_PATH` or `config.MODELS_DIR`; it can run beside
a live add-on.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, evaluate, features, log, predict, train

_log = log.get(__name__)

# One origin per person-day, so the units are near-independent -- ~500
# person-days is the real sample size however many rows the melt produces.
# 22:00 local the evening before puts a 07:00 departure at +9 h, inside the mid
# band where the pooled family serves.
ORIGIN_HOUR = 22

# A cell needs enough observations before its slope is allowed to mean anything.
MIN_CELL_OBSERVATIONS = 6

# The transition stratum is the top decile of |slope| -- a quantile rather than
# an absolute cut, so the stratum's size is stable across folds and subjects and
# there is no tuned constant to argue about.
TRANSITION_QUANTILE = 0.90


def _local(times: pd.Series) -> pd.Series:
    return times.dt.tz_convert(config.tzinfo())


def _cells(frame: pd.DataFrame) -> pd.DataFrame:
    """`(subject, dow, slot) -> mean binarised occupancy`, from whatever is given.

    Callers pass a fold's TRAINING rows only. Computing this over the whole
    history would mean the CHOICE of which rows to score had seen the test
    period's answers -- it would not favour either arm, since both are scored on
    identical rows, but the number would not reproduce on live data.
    """
    local = _local(frame["time"])
    out = pd.DataFrame({
        "subject": frame["subject"].to_numpy(),
        "dow": local.dt.dayofweek.to_numpy(),
        "slot": (local.dt.hour * (60 // config.GRID_MINUTES)
                 + local.dt.minute // config.GRID_MINUTES).to_numpy(),
        "home": (frame[train.RESIDUAL_BASE] >= evaluate.HOME_THRESHOLD).astype(float),
    })
    table = out.groupby(["subject", "dow", "slot"])["home"].agg(["mean", "count"])
    return table.rename(columns={"mean": "clim", "count": "n"}).reset_index()


def _slopes(cells: pd.DataFrame) -> pd.DataFrame:
    """How far the climatology moves across one hour, per cell."""
    step = 60 // config.GRID_MINUTES          # slots in an hour
    cells = cells.sort_values(["subject", "dow", "slot"]).copy()
    cells["slope"] = (cells.groupby(["subject", "dow"])["clim"]
                           .transform(lambda s: s - s.shift(step)))
    eligible = (cells["n"] >= MIN_CELL_OBSERVATIONS) & cells["slope"].notna()
    cells.loc[~eligible, "slope"] = np.nan
    return cells[["subject", "dow", "slot", "slope"]]


def _target_cell(scored: pd.DataFrame) -> pd.DataFrame:
    """The (dow, slot) each scored row is ABOUT, not the one it was made from."""
    target = scored["time"] + pd.to_timedelta(scored[features.HORIZON_COLUMN], unit="h")
    local = _local(target)
    return pd.DataFrame({
        "dow": local.dt.dayofweek.to_numpy(),
        "slot": (local.dt.hour * (60 // config.GRID_MINUTES)
                 + local.dt.minute // config.GRID_MINUTES).to_numpy(),
    }, index=scored.index)


# Only origins whose next departure is inside this many hours count.
#
# Added after a first run, and the reason is worth recording rather than
# quietly fixing: unconditioned, the metric's observed IQR was 10-34 h and its
# MAE ~9.7 h, because for a household that is mostly home the "first sustained
# crossing in 48 h" is often a day and a half out. That is a real question and
# it is not THIS question, which is about a 07:00 departure tomorrow morning.
#
# Conditioning on the OBSERVED horizon is safe here and would not be safe in
# general: `observed_h` is computed from the outcome alone, so the same origins
# are selected for every arm, and no arm can influence its own inclusion.
MAX_OBSERVED_H = 24

# What counts as "the curve is late" rather than "the curve is a bit off".
LATE_HOURS = 3


def departure_errors(scored: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Predicted minus observed departure hour, one row per (subject, origin).

    Both curves go through `predict._crossing` with the configured thresholds,
    so the metric is the sensor rather than a proxy for it. Origins where either
    curve never crosses are dropped -- an unproven crossing is the same "unknown"
    the sensor publishes, and scoring it as a number would invent one.
    """
    rows = []
    keyed = anchors.set_index(["subject", "time"])[train.RESIDUAL_BASE]
    for (subject, origin, fold), part in scored.groupby(
            ["subject", "time", "fold"], sort=False):
        try:
            state_now = float(keyed.loc[(subject, origin)])
        except KeyError:
            continue
        row = pd.Series({train.RESIDUAL_BASE: state_now})
        horizons = part[features.HORIZON_COLUMN].astype(int).to_numpy()
        predicted = dict(zip(horizons, part["p"].to_numpy()))
        observed = dict(zip(horizons, (part[features.TARGET_COLUMN].to_numpy()
                                       >= evaluate.HOME_THRESHOLD).astype(float)))
        args = (config.DEPARTURE_THRESHOLD, int(config.CROSSING_MIN_HOURS))
        got = predict._crossing(row, predicted, False, *args)
        truth = predict._crossing(row, observed, False, *args)
        if got is None or truth is None:
            continue
        if truth > MAX_OBSERVED_H:
            continue
        rows.append({"subject": subject, "time": origin, "fold": fold,
                     "predicted_h": got, "observed_h": truth,
                     "error_h": got - truth})
    return pd.DataFrame(rows)


def stratified_brier(scored: pd.DataFrame, wide: pd.DataFrame,
                     windows: list) -> dict[str, dict]:
    """Brier inside and outside the transition stratum, defined per fold.

    The stratum comes from each fold's TRAINING rows, so its membership shifts
    slightly between folds. That is why the per-fold sign test is the inference
    and the pooled number is only the effect size -- the same asymmetry
    `train.fold_record_allows` already encodes.
    """
    cell = _target_cell(scored)
    scored = scored.assign(dow=cell["dow"], slot=cell["slot"])
    out = {"transition": [], "flat": []}
    cuts = []
    for index, (start, _stop) in enumerate(windows):
        rows = scored[scored["fold"] == index]
        if rows.empty:
            continue
        past = wide[wide["time"] < start]
        if past.empty:
            continue
        slopes = _slopes(_cells(past))
        usable = slopes["slope"].abs().dropna()
        if usable.empty:
            continue
        cut = float(usable.quantile(TRANSITION_QUANTILE))
        cuts.append(cut)
        merged = rows.merge(slopes, on=["subject", "dow", "slot"], how="left")
        steep = merged["slope"].abs() >= cut
        for name, mask in (("transition", steep), ("flat", ~steep)):
            part = merged[mask.fillna(False) if name == "transition"
                          else ~steep.fillna(False)]
            if part.empty:
                continue
            out[name].append(evaluate.score(part[features.TARGET_COLUMN].to_numpy(),
                                            part["p"].to_numpy()))
    report = {"per_fold": {k: [s.brier for s in v] for k, v in out.items()}}
    for name, scores in out.items():
        if not scores:
            continue
        summary = evaluate.summarize(scores)
        rate = summary["base_rate"]
        report[name] = {
            "brier": summary["brier"], "n": summary["n"], "base_rate": rate,
            # The Brier a constant base-rate forecast would score on exactly
            # these rows. Without it a stratified number means nothing: cells in
            # the transition stratum sit near p=0.5 and flat cells near 0 or 1,
            # so p(1-p) differs threefold before any model is involved.
            "spread": rate * (1.0 - rate),
        }
    report["cut"] = {"median": float(np.median(cuts)) if cuts else float("nan"),
                     "folds": len(cuts)}
    return report


def run_arm(path: Path, windows: list, extras: tuple[str, ...],
            horizons: tuple[int, ...], n_jobs: int | None) -> pd.DataFrame:
    """Train the pooled family with `extras` served, return its out-of-fold runs.

    The estimator is thrown away. What is being measured is the feature list, not
    a model anybody is going to serve, and keeping it would only invite somebody
    to serve it.
    """
    before = features.SHIPPED_EXTRAS
    features.SHIPPED_EXTRAS = extras
    try:
        _log.info("arm %s: %d features",
                  ",".join(extras) or "control", len(train.base_features()))
        _estimator, scored, _rows = train.train_pooled(
            path, windows, horizons=horizons, n_jobs=n_jobs)
        return scored
    finally:
        features.SHIPPED_EXTRAS = before


def compare_strata(strata: dict[str, dict], control: str) -> dict[str, dict]:
    """Per-fold sign test on each stratum's Brier, arm against control.

    The pooled Brier is the effect size and this is the inference -- the same
    asymmetry `train.fold_record_allows` encodes, and the reason it matters here
    is that the stratum's membership shifts slightly between folds because it is
    defined from each fold's own training rows.
    """
    out = {}
    for arm, report in strata.items():
        if arm == control:
            continue
        for stratum in ("transition", "flat"):
            mine = report.get("per_fold", {}).get(stratum, [])
            theirs = strata[control].get("per_fold", {}).get(stratum, [])
            pairs = [(a, b) for a, b in zip(mine, theirs)
                     if not (np.isnan(a) or np.isnan(b))]
            if not pairs:
                continue
            wins = sum(1 for a, b in pairs if a < b)
            out[f"{arm}/{stratum}"] = {
                "folds_better": wins, "folds": len(pairs),
                "sign_p": evaluate.sign_test(wins, len(pairs))}
    return out


def compare(errors: dict[str, pd.DataFrame], control: str) -> dict[str, dict]:
    """Per-fold sign test of each arm's median departure error against control.

    Signed error, not absolute: the symptom is a curve that is LATE, and an arm
    that halved the lateness while doubling the scatter would look identical on
    an absolute metric.
    """
    base = errors[control]
    out = {}
    for name, frame in errors.items():
        if name == control or frame.empty or base.empty:
            continue
        wins = played = 0
        for fold, part in frame.groupby("fold"):
            theirs = base[base["fold"] == fold]
            if part.empty or theirs.empty:
                continue
            played += 1
            wins += abs(part["error_h"].median()) < abs(theirs["error_h"].median())
        out[name] = {"folds_better": wins, "folds": played,
                     "sign_p": evaluate.sign_test(wins, played)}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Price a candidate feature")
    # No default from `config`: a probe that can silently read the add-on's own
    # table is a probe somebody will eventually point at production.
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--arms", default="control,int_calendar",
                        help="comma-separated; 'control' serves nothing extra")
    parser.add_argument("--horizons", type=int, nargs="*",
                        default=list(config.HORIZONS_H))
    parser.add_argument("--origin-hour", type=int, default=ORIGIN_HOUR)
    parser.add_argument("--n-jobs", type=int, default=None)
    # Configured from the saved settings, NOT `runtime.bootstrap`. A probe runs
    # offline against a copy of the archive; requiring a live Home Assistant to
    # measure a feature would mean the only place it could run is the box it
    # must not touch.
    parser.add_argument("--config", type=Path, default=config.CONFIG_PATH)
    args = parser.parse_args(argv)

    log.configure("info")
    config.configure(config.Settings.from_json(args.config.read_text()))

    horizons = tuple(args.horizons)
    windows, _geometry = train.shared_windows(args.features)
    wide = train.read_wide(args.features)
    _log.info("%d folds, %d horizons, %d wide rows",
              len(windows), len(horizons), len(wide))

    # One origin per person-day, at a fixed local hour. Declared before running,
    # not chosen after seeing which hour flatters an arm.
    local = _local(wide["time"])
    anchors = wide.loc[local.dt.hour == args.origin_hour,
                       ["subject", "time", train.RESIDUAL_BASE]]
    anchors = anchors.assign(_d=_local(anchors["time"]).dt.date) \
                     .drop_duplicates(["subject", "_d"]).drop(columns="_d")
    _log.info("%d anchor origins at %02d:00 local", len(anchors), args.origin_hour)

    errors, strata = {}, {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        extras = () if arm == "control" else tuple(arm.split("+"))
        started = dt.datetime.now()
        scored = run_arm(args.features, windows, extras, horizons, args.n_jobs)
        errors[arm] = departure_errors(scored, anchors)
        strata[arm] = stratified_brier(scored, wide, windows)
        _log.info("arm %s scored in %s", arm, dt.datetime.now() - started)

    print(f"\n{'arm':<16} {'n':>5} {'median err':>11} {'MAE':>7} "
          f"{f'|err|>={LATE_HOURS}h':>10} {'observed IQR':>13}   departure hour")
    for arm, frame in errors.items():
        if frame.empty:
            print(f"{arm:<16}     -  no origin had a crossing in both curves")
            continue
        q1, q3 = frame["observed_h"].quantile([0.25, 0.75])
        late = (frame["error_h"].abs() >= LATE_HOURS).mean()
        print(f"{arm:<16} {len(frame):>5} {frame['error_h'].median():>+11.2f} "
              f"{frame['error_h'].abs().mean():>7.2f} {late*100:>9.0f}% "
              f"{f'{q1:.1f}-{q3:.1f} h':>13}")

    print(f"\n{'arm':<16} {'stratum':<11} {'brier':>7} {'spread':>7} {'n':>7}"
          f"   (compare an arm to an arm, NEVER a stratum to a stratum)")
    for arm, report in strata.items():
        for name in ("transition", "flat"):
            cell = report.get(name)
            if cell:
                print(f"{arm:<16} {name:<11} {cell['brier']:>7.4f} "
                      f"{cell['spread']:>7.4f} {cell['n']:>7}")
    any_cut = next(iter(strata.values()), {}).get("cut", {})
    print(f"  stratum cut: median |slope| >= {any_cut.get('median', float('nan')):.3f} "
          f"over {any_cut.get('folds', 0)} folds")

    strata_verdict = compare_strata(strata, "control")
    if strata_verdict:
        print(f"\n{'arm / stratum':<28} {'folds better':>13} {'sign p':>8}"
              f"   vs control, per-fold Brier")
        for key, cell in strata_verdict.items():
            print(f"{key:<28} {cell['folds_better']:>6}/{cell['folds']:<6} "
                  f"{cell['sign_p']:>8.3f}")

    verdict = compare(errors, "control")
    if verdict:
        print(f"\n{'arm':<16} {'folds better':>13} {'sign p':>8}   vs control, "
              f"per-fold median signed error")
        for arm, cell in verdict.items():
            print(f"{arm:<16} {cell['folds_better']:>6}/{cell['folds']:<6} "
                  f"{cell['sign_p']:>8.3f}")


# --- the departure-timing comparison ---------------------------------------
#
# The two numbers this settles were never comparable. "2.5 h" came from a
# weekday lookup scored on days a departure was OBSERVED; "7.4 h" came from the
# occupancy curve scored on origins where `_crossing` happened to cross. They
# differ in origin hour, in what counts as a departure, in units, and -- worst --
# in SELECTION: dropping origins whose predicted curve never crosses lets an arm
# choose the events it is judged on.
#
# So: one event set, taken from the LABEL. Candidate days with an observed
# departure, defined from the outcome alone, so no arm can influence its own
# inclusion. Every arm reduces to the same quantity, the local hour of the first
# departure.

DEPARTURE_ORIGIN_HOUR = 4       # matches `departure.ORIGIN_HOUR`


def _curve_hour(row: pd.Series, curve: dict[int, float]) -> float | None:
    """The hour a curve says they leave, through the SHIPPED reduction rule.

    `predict._crossing` is what produces `sensor.*_hours_until_away`, so using it
    here measures the sensor rather than a proxy for it.
    """
    crossing = predict._crossing(row, curve, False, config.DEPARTURE_THRESHOLD,
                                 int(config.CROSSING_MIN_HOURS))
    return None if crossing is None else DEPARTURE_ORIGIN_HOUR + crossing


def _summarise(name: str, errors: np.ndarray, misses: int) -> dict:
    scored = len(errors)
    return {
        "arm": name, "n": scored, "misses": misses,
        "mae": float(np.mean(np.abs(errors))) if scored else float("nan"),
        "median_ae": float(np.median(np.abs(errors))) if scored else float("nan"),
        "within_1h": float(np.mean(np.abs(errors) <= 1) * 100) if scored else float("nan"),
        "over_3h": float(np.mean(np.abs(errors) >= 3) * 100) if scored else float("nan"),
        # Signed, because the symptom is a curve that is LATE and an arm that
        # halved the lateness while doubling the scatter would look identical on
        # an absolute metric.
        "median_signed": float(np.median(errors)) if scored else float("nan"),
    }


def compare_departure_timing(path: Path, horizons: tuple[int, ...] = tuple(range(1, 21)),
                             n_jobs: int | None = None) -> pd.DataFrame:
    """Score the occupancy curve and the weekday lookup on the same departures.

    The production arm is the DEDICATED family, because from an 04:00 origin a
    07:30 departure is +3.5 h and that band is the dedicated family's -- the
    pooled one serves from +19 h. This is therefore a different part of the stack
    from the one the feature probes scored.

    A day whose predicted curve never crosses is a MISS, not a dropped row.
    Both readings are reported: excluding them is "on the events it was willing
    to call", and imputing that person's own median hour is what a consumer
    experiences when the sensor says unknown. The gap between the two is the
    number this comparison has been missing.
    """
    from . import departure

    wide = train.read_wide(path)
    windows, _geometry = train.shared_windows(path)
    days = departure.feature_frame(departure.label_days(
        wide[["subject", "time", "home_frac"]]))
    events = days[days["candidate"] & days["left_today"]
                  & (days["subject"] != config.HOUSE_SLUG)]
    _log.info("%d departure events, %d fold windows", len(events), len(windows))

    # Out-of-fold predictions for the horizons an 04:00 origin needs.
    per_horizon = {}
    for horizon in horizons:
        try:
            _estimator, scored, _rows = train.train_dedicated(path, horizon, windows)
        except Exception as err:                                  # noqa: BLE001
            _log.warning("+%sh: skipped -- %s", horizon, err)
            continue
        per_horizon[horizon] = scored.set_index(["subject", "time"])["p"]
    _log.info("scored %d horizons out of fold", len(per_horizon))

    anchor = _local(wide["time"])
    origins = wide.loc[anchor.dt.hour == DEPARTURE_ORIGIN_HOUR,
                       ["subject", "time", train.RESIDUAL_BASE]]
    origins = origins.assign(_d=_local(origins["time"]).dt.date)

    rows = []
    for event in events.itertuples():
        at = origins[(origins["subject"] == event.subject)
                     & (origins["_d"] == event.date.date())]
        if at.empty:
            continue
        key = (event.subject, at["time"].iloc[0])
        curve = {h: float(s.loc[key]) for h, s in per_horizon.items()
                 if key in s.index}
        if not curve:
            continue
        row = pd.Series({train.RESIDUAL_BASE: float(at[train.RESIDUAL_BASE].iloc[0])})
        rows.append({
            "subject": event.subject, "date": event.date,
            "truth": event.departure_hour,
            "production": _curve_hour(row, curve),
            # The lookup, and the flat median it has to justify itself against:
            # without the second arm a reader cannot tell how much of the
            # lookup's skill is WEEKDAY and how much is merely "mornings".
            "weekday_median": event.wday_hour,
            "flat_median": event.all_hour,
        })
    scored = pd.DataFrame(rows)

    out = []
    for name in ("production", "weekday_median", "flat_median"):
        got = scored[name]
        have = got.notna()
        errors = (got[have] - scored.loc[have, "truth"]).to_numpy()
        out.append(_summarise(name, errors, int((~have).sum())))
        if name == "production" and (~have).any():
            # The same arm with its misses imputed at that person's own median,
            # which is what a consumer sees when the sensor reads unknown.
            filled = got.fillna(scored["flat_median"])
            ok = filled.notna()
            out.append(_summarise("production (misses imputed)",
                                  (filled[ok] - scored.loc[ok, "truth"]).to_numpy(), 0))
    return pd.DataFrame(out)


# Last in the file, not in the middle of it. The guard used to sit above the
# departure-timing block, so running this as a script called `main()` before
# those definitions existed and the whole comparison was reachable only from a
# REPL that imported the module.
if __name__ == "__main__":
    main()
