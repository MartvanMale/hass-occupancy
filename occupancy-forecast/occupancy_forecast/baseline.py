"""The ladder every model has to climb.

Six rungs, on the same folds and rows as the model. Persistence is close to
unbeatable below about four hours, same-slot-yesterday is the long-horizon bar,
and both are PROBABILITIES: scored as a hard 0/1 they lose to the base rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config, evaluate, features


# Candidate shrink weights for the row-local rungs. Fitted on TRAINING rows
# only, per fold.
SHRINK_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


def _fit_shrink(train: pd.DataFrame, raw: np.ndarray, horizon: int) -> tuple[float, float]:
    """Pull a row-local rung toward the base rate: `state_now` and daily lags
    are single observations saying 0.0 or 1.0; shrinking makes the baselines
    STRONGER. Fitted on the fold's training rows, so calibration, not a peek.
    """
    outcome = (train[f"y_{horizon}h"] >= evaluate.HOME_THRESHOLD).astype(float)
    base = float(outcome.mean())
    keep = ~np.isnan(raw)
    if not keep.any():
        return 1.0, base
    truth, values = outcome.to_numpy()[keep], raw[keep]
    losses = [float(np.mean((w * values + (1 - w) * base - truth) ** 2))
              for w in SHRINK_GRID]
    return SHRINK_GRID[int(np.argmin(losses))], base


def shrink(raw: np.ndarray, weight: float, base: float) -> np.ndarray:
    return np.clip(weight * raw + (1 - weight) * base, 0.0, 1.0)


def _climatology(train: pd.DataFrame, test: pd.DataFrame, keys: list[str],
                 horizon: int) -> np.ndarray:
    """P(home) in the target slot, from the training rows sharing `keys`,
    grouped on the TARGET slot's calendar so it is fair against a model that
    also sees it.
    """
    outcome = (train[f"y_{horizon}h"] >= evaluate.HOME_THRESHOLD).astype(float)
    frame = train[keys].copy()
    frame["_y"] = outcome
    frame = frame.dropna(subset=["_y"])

    table = frame.groupby(keys, dropna=False)["_y"].mean()
    fallback = float(outcome.mean())

    joined = test[keys].merge(table.rename("_p"), left_on=keys, right_index=True,
                              how="left")
    return joined["_p"].fillna(fallback).to_numpy()


def _target_calendar(table: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The target slot's weekday and slot index, rebuilt from `time` because
    the parquet's sin/cos calendar is useless as a groupby key.
    """
    local = (table["time"] + pd.Timedelta(hours=horizon)).dt.tz_convert(config.TIMEZONE)
    return pd.DataFrame({
        "_dow": local.dt.dayofweek,
        "_slot": features.slot_of_day(local),
        "subject": table["subject"].to_numpy(),
    }, index=table.index)


def predictors(horizon: int, calendar: pd.DataFrame | None = None) -> dict:
    """The ladder, in increasing order of what it is allowed to know. `calendar`
    is `_target_calendar` over the whole frame, computed once by the caller:
    row-wise, so a fold's slice of it is what rebuilding would give.
    """

    def keys_for(train, test):
        if calendar is None:
            keys = _target_calendar(pd.concat([train, test]), horizon)
            return keys.iloc[:len(train)], keys.iloc[len(train):]
        return calendar.loc[train.index], calendar.loc[test.index]

    def base_rate(train, test):
        outcome = (train[f"y_{horizon}h"] >= evaluate.HOME_THRESHOLD).astype(float)
        return np.full(len(test), float(outcome.mean()))

    def persistence(train, test):
        # State now, carried forward -- shrunk toward the base rate, because one
        # observation is not a probability. See _fit_shrink.
        weight, base = _fit_shrink(train, train["state_now"].to_numpy(), horizon)
        return shrink(test["state_now"].to_numpy(), weight, base)

    def same_slot_yesterday(train, test):
        # The nearest safe daily lag of the target slot; at 36-48 h "yesterday"
        # is in the future, so this falls back to two days.
        lags = features.safe_daily_lags(horizon)
        if not lags:
            return np.full(len(test), np.nan)
        column = f"tgt{horizon}h_lag{min(lags)}d"
        weight, base = _fit_shrink(train, train[column].to_numpy(), horizon)
        return shrink(test[column].to_numpy(), weight, base)

    def slot_climatology(train, test):
        tr, te = keys_for(train, test)
        return _climatology(train.assign(_slot=tr["_slot"].to_numpy()),
                            test.assign(_slot=te["_slot"].to_numpy()),
                            ["_slot"], horizon)

    def weekday_slot_climatology(train, test):
        tr, te = keys_for(train, test)
        return _climatology(
            train.assign(_slot=tr["_slot"].to_numpy(), _dow=tr["_dow"].to_numpy()),
            test.assign(_slot=te["_slot"].to_numpy(), _dow=te["_dow"].to_numpy()),
            ["_dow", "_slot"], horizon)

    def weekday_slot_subject_climatology(train, test):
        tr, te = keys_for(train, test)
        return _climatology(
            train.assign(_slot=tr["_slot"].to_numpy(), _dow=tr["_dow"].to_numpy()),
            test.assign(_slot=te["_slot"].to_numpy(), _dow=te["_dow"].to_numpy()),
            ["subject", "_dow", "_slot"], horizon)

    return {
        "base_rate": base_rate,
        "persistence": persistence,
        "same_slot_yesterday": same_slot_yesterday,
        "slot_climatology": slot_climatology,
        "weekday_slot_climatology": weekday_slot_climatology,
        "weekday_slot_subject_climatology": weekday_slot_subject_climatology,
    }


def columns_for(horizon: int) -> list[str]:
    """Every column `predictors(horizon)` reads, and nothing else. Kept beside
    the rungs, so a new rung reaching for another column cannot silently outgrow
    a slice made where it is invisible.
    """
    columns = ["time", "subject", f"y_{horizon}h", "state_now"]
    lags = features.safe_daily_lags(horizon)
    if lags:
        columns.append(f"tgt{horizon}h_lag{min(lags)}d")
    return columns


def run(table: pd.DataFrame, horizon: int, geometry: dict | None = None,
        windows: list | None = None,
        required: Iterable[str] = ()) -> dict[str, dict]:
    """Score every rung on the model's folds and rows: `geometry`, `windows` and
    `required` come from the caller because the ship gate walks `per_fold`
    POSITIONALLY against the model's and compares Briers over the same rows.
    """
    required = [c for c in required if c in table.columns]
    frame = (table.dropna(subset=[f"y_{horizon}h", *required])
             .sort_values("time").reset_index(drop=True))
    if geometry is None:
        geometry = evaluate.fold_geometry(frame["time"])

    if windows is None:
        folds = evaluate.calendar_folds(
            frame["time"], embargo=evaluate.embargo_for(horizon), **geometry)
        cuts = [(f.train_idx, f.test_idx) for f in folds]
    else:
        embargo = evaluate.embargo_for(horizon)
        times = frame["time"]
        cuts = [(np.flatnonzero(((times + embargo) < start).to_numpy()),
                 np.flatnonzero(((times >= start) & (times < stop)).to_numpy()))
                for start, stop in windows]
    if not cuts:
        return {}

    # Once, for the whole frame and all six rungs.
    rungs = predictors(horizon, _target_calendar(frame, horizon))
    per_rung: dict[str, list] = {name: [] for name in rungs}
    for train_idx, test_idx in cuts:
        if len(train_idx) == 0 or len(test_idx) == 0:
            # Still scored, as an empty fold, so the positional alignment with
            # the model's fold list survives a window neither of them can use.
            for name in rungs:
                per_rung[name].append(evaluate.score(np.array([]), np.array([])))
            continue
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        truth = test[f"y_{horizon}h"].to_numpy()
        for name, fn in rungs.items():
            per_rung[name].append(evaluate.score(truth, fn(train, test)))

    return {name: evaluate.summarize(scores) for name, scores in per_rung.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the baseline ladder")
    parser.add_argument("--features", type=Path, default=config.FEATURES_PATH)
    parser.add_argument("--horizons", type=int, nargs="*", default=list(config.HORIZONS_H))
    parser.add_argument("--subject", default=None,
                        help="restrict to one subject (a person's slug, or house)")
    args = parser.parse_args(argv)

    from . import runtime
    runtime.bootstrap()

    table = pd.read_parquet(args.features)
    if args.subject:
        table = table[table["subject"] == args.subject]

    names = list(predictors(args.horizons[0]))
    print(f"{'horizon':>8} " + " ".join(f"{n[:13]:>14}" for n in names))
    for horizon in args.horizons:
        result = run(table, horizon)
        if not result:
            print(f"{horizon:>7}h  (no folds)")
            continue
        cells = " ".join(f"{result[n]['brier']:>14.3f}" for n in names)
        print(f"{horizon:>7}h {cells}")
    print("\nBrier, lower is better. Folds:",
          result.get("base_rate", {}).get("n_folds", "?"))


if __name__ == "__main__":
    main()
