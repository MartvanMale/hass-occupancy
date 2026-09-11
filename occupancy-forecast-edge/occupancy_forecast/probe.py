"""Price a candidate feature against a control, honestly.

Every candidate is BUILT into the parquet (`features.SHIPPED_EXTRAS`) and this
flips which are SERVED over one table and one set of folds, scoring timing that
the ship gate's pooled Brier cannot see. Never writes to the add-on's own paths.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, evaluate, features, log, predict, train

_log = log.get(__name__)

# One origin per person-day, so the units are near-independent; 22:00 puts a
# 07:00 departure at +9 h, inside the band the pooled family serves.
ORIGIN_HOUR = 22

# A cell needs enough observations before its slope is allowed to mean anything.
MIN_CELL_OBSERVATIONS = 6

# The top decile of |slope|: a quantile, so the stratum's size is stable across
# folds and there is no tuned constant to argue about.
TRANSITION_QUANTILE = 0.90


def _local(times: pd.Series) -> pd.Series:
    return times.dt.tz_convert(config.tzinfo())


def _cells(frame: pd.DataFrame) -> pd.DataFrame:
    """`(subject, dow, slot) -> mean binarised occupancy`, from whatever is given.
    Callers pass a fold's TRAINING rows only: over the whole history the CHOICE
    of which rows to score would depend on the test period's answers.
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


# Unconditioned, "first sustained crossing in 48 h" is often a day and a half
# out, a different question. Conditioning on the OBSERVED horizon is safe: it
# comes from the outcome alone, so no arm can influence its own inclusion.
MAX_OBSERVED_H = 24

# What counts as "the curve is late" rather than "the curve is a bit off".
LATE_HOURS = 3


def departure_errors(scored: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Predicted minus observed departure hour, one row per (subject, origin).
    Both curves go through `predict._crossing`, so the metric is the sensor
    rather than a proxy; an origin where either never crosses is dropped.
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
    """Brier inside and outside the transition stratum, defined per fold from
    TRAINING rows, so membership shifts between folds: the per-fold sign test is
    the inference and the pooled number only the effect size.
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
        # `>=` on a float column is already plain bool (NaN compares False),
        # so a NaN slope lands in "flat" without any fillna.
        steep = merged["slope"].abs() >= cut
        for name, mask in (("transition", steep), ("flat", ~steep)):
            part = merged[mask]
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
            # Without the base-rate spread a stratified Brier means nothing:
            # transition cells sit near p=0.5 and flat cells near 0 or 1.
            "spread": rate * (1.0 - rate),
        }
    report["cut"] = {"median": float(np.median(cuts)) if cuts else float("nan"),
                     "folds": len(cuts)}
    return report


def run_arm(path: Path, windows: list, extras: tuple[str, ...],
            horizons: tuple[int, ...], n_jobs: int | None) -> pd.DataFrame:
    """Train the pooled family with `extras` served, return its out-of-fold runs.
    The estimator is thrown away: what is measured is the feature list, and
    keeping it would invite somebody to serve it.
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
    """Per-fold sign test on each stratum's Brier, arm against control."""
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
    Signed, not absolute: the symptom is a LATE curve, and halving the lateness
    while doubling the scatter would look identical on an absolute metric.
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
    # From the saved settings, not `runtime.bootstrap`: requiring a live Home
    # Assistant would mean it could only run on the box it must not touch.
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
# One event set, from the LABEL: candidate days with an observed departure, so
# no arm can influence its own inclusion.

DEPARTURE_ORIGIN_HOUR = 4       # matches `departure.ORIGIN_HOUR`


def _curve_hour(row: pd.Series, curve: dict[int, float]) -> float | None:
    """The hour a curve says they leave, through `predict._crossing`, which is
    what produces the published sensor.
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
        # Signed, for the reason `compare` gives.
        "median_signed": float(np.median(errors)) if scored else float("nan"),
    }


def compare_departure_timing(path: Path, horizons: tuple[int, ...] = tuple(range(1, 21)),
                             n_jobs: int | None = None) -> pd.DataFrame:
    """Score the DEDICATED family's curve (from 04:00 a 07:30 departure is
    +3.5 h) and the weekday lookup on the same departures. A curve that never
    crosses is a MISS, not a dropped row; both readings are reported.
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
            # Without the flat median nobody can tell how much of the lookup's
            # skill is WEEKDAY and how much is merely "mornings".
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


# Last in the file, so a script run sees every definition above it.
if __name__ == "__main__":
    main()
