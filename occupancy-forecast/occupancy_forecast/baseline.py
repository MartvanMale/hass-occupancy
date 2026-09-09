"""The ladder every model has to climb, and the numbers it should reproduce.

Measured over about six months of one two-person household, 9 rolling folds,
Brier on the binarised outcome (lower is better):

    subject   base    persist @1h   persist @24h   same-slot-yday   wday x slot
    alice     0.236      0.056          0.184           0.184           0.226
    bob       0.226      0.037          0.168           0.168           0.222
    house     0.212      0.035          0.103           0.103           0.214

Three facts that should shape how results here are read:

  * **Below about four hours, persistence is close to unbeatable.** "They are
    home now" scores 0.056 against a 0.236 base at 1 h for alice. A 1-3 h
    forecast that does not clearly beat it is `state.get()` with extra steps.

  * **Same-slot-yesterday, not the calendar, is the long-horizon bar.** It is
    the best baseline at every horizon past 6 h -- 0.184 for alice against 0.226
    for weekday x slot climatology, so ~22% Brier skill against the calendar's
    ~4%. It has to be used as a *probability* (the fractional `home_frac`), not
    as a hard 0/1 call: scoring the same information as a confident binary made
    it look *worse* than the base rate.

  * **Calendar climatology barely works, and for the house it is worse than
    useless** -- 0.214 against a 0.212 base rate. Roughly 24 samples per
    (weekday, slot) cell is not enough to estimate a probability, so the cell
    means are mostly noise. Note the contrast with the daily office flag, where
    weekday is *strong*: the weekday says a lot about whether somebody goes to
    the office and very little about whether they are in the house at 15:00.

So the model has to beat ~0.184 at 24 h, not the ~0.226 the calendar suggests.
That is a harder bar than the calendar-shaped intuition implies, and it is the
right one.
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
    """Choose how far to pull a row-local prediction toward the base rate.

    `state_now` and the daily lags are single observations, so as probabilities
    they are wildly overconfident: they say 0.0 or 1.0 and mean "one sample
    said so". Measured 2026-08-31 at 48 h, shrinking the lag from w=1.0 to
    w=0.7 improved its Brier from 0.202 to 0.168 -- a bigger move than most of
    what the model does.

    Shrinking makes the *baselines stronger*, which raises the bar the model has
    to clear. That is the point: an overconfident baseline is an easy baseline,
    and beating one is not evidence of anything.

    Fitted on the training rows of the fold and applied to test, so this is
    calibration and not a peek. Returns (weight, base_rate).
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
    """P(home) in the target slot, from the training rows sharing `keys`.

    Grouped on the TARGET slot's calendar, not the origin's, so this is a fair
    comparison against a model that also sees the target calendar.
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
    """Recover the target slot's weekday and slot index for grouping.

    The parquet stores the calendar as sin/cos pairs, which are right for the
    model and useless as a groupby key, so the two integers are rebuilt here
    from the timestamp rather than stored twice.
    """
    local = (table["time"] + pd.Timedelta(hours=horizon)).dt.tz_convert(config.TIMEZONE)
    return pd.DataFrame({
        "_dow": local.dt.dayofweek,
        "_slot": features.slot_of_day(local),
        "subject": table["subject"].to_numpy(),
    }, index=table.index)


def predictors(horizon: int, calendar: pd.DataFrame | None = None) -> dict:
    """The ladder, in increasing order of what it is allowed to know.

    `calendar` is `_target_calendar` over the WHOLE frame, computed once by the
    caller. It is row-wise arithmetic on `time`, so slicing it by a fold's index
    gives exactly what rebuilding it from that fold's rows gives -- and the
    three climatology rungs were each rebuilding it, per fold, from a
    `pd.concat` that copied the training frame. Three times a fold, nineteen
    folds, forty-eight horizons.

    Optional so the rungs stay usable on their own, which is what `main` and the
    tests do.
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
        # The nearest safe daily lag of the target slot. At 36-48 h "yesterday"
        # is in the future, so this falls back to two days -- see
        # features.safe_daily_lags.
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
    """Every column `predictors(horizon)` reads, and nothing else.

    Kept here, beside the rungs, rather than at the call site: a new rung that
    reaches for another column would otherwise silently outgrow a slice made
    somewhere that cannot see it. `train.train_all` fans the ladder out over
    horizons and hands each worker only these, which is the difference between
    pickling five columns and pickling a thousand-column table 48 times.

    The climatology rungs need no columns of their own -- they group on the
    target calendar, which `_target_calendar` rebuilds from `time`.
    """
    columns = ["time", "subject", f"y_{horizon}h", "state_now"]
    lags = features.safe_daily_lags(horizon)
    if lags:
        columns.append(f"tgt{horizon}h_lag{min(lags)}d")
    return columns


def run(table: pd.DataFrame, horizon: int, geometry: dict | None = None,
        windows: list | None = None,
        required: Iterable[str] = ()) -> dict[str, dict]:
    """Score every rung on the same folds a model would be scored on.

    `geometry` comes from the caller so the ladder and the model are cut on
    identical folds -- the model filters more strictly than this does, so
    letting each derive its own from its own frame is how they drift apart. On
    a short history it matters twice over: a ladder cut with the default
    geometry returns nothing at all, which reads downstream as "no baseline
    beat the model" rather than "no baseline ran".

    `windows` closes that gap properly. The ship gate counts folds the model
    won, indexing this function's `per_fold` list POSITIONALLY against the
    model's -- which is only correct if both cut the same calendar windows.
    Passing them in makes that an argument rather than a coincidence of two
    frames happening to start on the same day.

    `required` closes the other half of the same gap, the ROWS. The model
    drops any row missing a required origin feature; this dropped on the
    target alone, so its Brier was a mean over a superset of the model's rows
    and the gate compared two denominators. `train.required_origin_columns`
    is what the fits use, and `train_all` passes it here.
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

    # Once, for the whole frame and all six rungs -- it was being rebuilt three
    # times per fold from a concat of the fold's own rows, and `predictors`
    # itself was being rebuilt three times per fold on top of that.
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
