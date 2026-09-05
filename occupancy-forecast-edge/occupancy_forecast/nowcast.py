"""What is true RIGHT NOW, for the serving row only.

NEVER call this from `features.build`. The training table must stay on the
wall-clock grid, or every fold is scored against a definition it was not fitted
on.

**The problem.** A slot runs forward from its left edge, so the newest row of a
serving build is the IN-PROGRESS slot: at 19:47 it is `[19:30, 20:00)`, with the
present state carried forward through the thirteen minutes that have not
happened yet. That row is not stale -- it does move when somebody leaves -- but
how fast it moves depends on WHERE IN THE HALF HOUR they left. MEASURED, for a
departure inside the 19:30 slot:

    19:31 -> state_now 0.033   reads away immediately
    19:40 -> state_now 0.333   reads away immediately
    19:46 -> state_now 0.533   reads HOME for fourteen more minutes
    19:59 -> state_now 0.967   reads HOME for one more minute

Anything after the slot's midpoint keeps the house occupied until the slot
turns over. So the honest description is not "stale" but "correct within
0-15 minutes, and you do not get to know which" -- which is exactly the wrong
shape for something an automation acts on, and it is why an event-driven
re-predict was not worth having on its own.

**Why not simply shift the grid.** A trailing 30-minute window anchored at
`now` is strictly worse: one minute after a departure it reports 29/30 = 0.967.
A 30-minute average cannot be fast at any phase. Measured, not reasoned about.

**What this does instead.** Recompute the origin block -- and only the origin
block -- at `now`, over a much shorter window. Same definition as `home_frac`,
same integration code, five minutes instead of thirty. Worst case becomes a
constant 2.5 minutes rather than a phase-dependent 0-15.

Not a raw instantaneous binary, deliberately. That would reintroduce precisely
the GPS jitter `slot_fraction`'s time-weighting exists to remove -- see the
`14:22:11 -> 14:22:49 -> 14:26:16` example in features.py, and the measured 11
of 28 workplace episodes under two minutes. A five-minute weighted window
cannot be flipped by anything shorter than 2.5 minutes.

**Why the origin block is enough.** `train.RESIDUAL_BASE` is `state_now`, and
`train.predict` is `clip(model_residual + state_now)`. `state_now` is added to
every one of the 48 horizons, so moving it moves the whole curve at once. It is
the single highest-leverage value in the serving path.

**The accepted skew.** Serving draws `state_now` from a 5-minute window where
training drew it from 30, so the served distribution is slightly more bimodal
than the fitted one. The clean fix is to redefine `state_now` in
`features.build` as the trailing short window and retrain; that is a change to
every model and is deliberately not made here.

Left out on purpose, and worth doing next: the proximity columns are still read
off the 30-minute grid, so `distance_km` can be up to half an hour old while
`state_now` is seconds old. Distance is the strongest single feature group, and
its trace updates every few minutes while somebody drives, so there is real
resolution being discarded -- but it feeds the residual rather than the base,
which makes it a smaller error than the one this module fixes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features

# The window the origin state is measured over, ending at `now`.
#
# Five minutes is a window length, not a tuned dwell threshold: it rejects
# anything under 2.5 minutes because such a blip cannot reach half the window,
# and it admits a real change 2.5 minutes in. Shorter would start passing the
# jitter; longer would give back the responsiveness this exists to buy.
WINDOW_MINUTES = 5

# How far back to read events. Only needs to cover the window plus enough slack
# for `seeded_states` to find a prior value, which it does by seeding anyway.
LOOKBACK_MINUTES = 6 * 60


def presence_fraction(events: list[tuple[str, str]], at: pd.Timestamp,
                      window_min: int = WINDOW_MINUTES) -> float | None:
    """Fraction of the `window_min` minutes ENDING at `at` spent at home.

    Same definition and the same integration as `home_frac`, over a shorter
    window -- see `features.slot_fraction`, whose `minutes` argument exists for
    this caller alone.
    """
    if not events:
        return None
    window = pd.DatetimeIndex([at - pd.Timedelta(minutes=window_min)], tz="UTC")
    out = features.slot_fraction(events, window, config.HOME_STATE, window_min)
    value = out["frac"].iloc[0]
    return None if pd.isna(value) else float(value)


def apply(rows: pd.DataFrame, source, at: pd.Timestamp) -> pd.DataFrame:
    """Move the origin block of each serving row forward to `at`.

    Overrides a column only where there is a better answer for it. A subject
    whose tracker has said nothing keeps its grid values rather than being
    blanked: a missing nowcast must degrade to today's behaviour, never to NaN,
    because `predict.current_rows` drops any row without a `state_now` -- which
    deletes all 48 horizons for that subject, not merely the ones anchored on
    it.

    Everything outside the origin block is left exactly as built -- the daily
    lags, the climatology, the target calendar and `coverage`. Those are
    anchored on the slot and must stay that way; `test_predict.py` asserts they
    come through byte-identical.
    """
    if rows.empty:
        return rows

    start = (at - pd.Timedelta(minutes=LOOKBACK_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stop = at.strftime("%Y-%m-%dT%H:%M:%SZ")
    index = pd.DatetimeIndex([at], tz="UTC")

    fractions: dict[str, float] = {}
    minutes: dict[str, float] = {}
    for subject in config.SUBJECTS:
        try:
            events = features.presence_events(source, subject, start, stop)
        except Exception:  # noqa: BLE001
            # A source that cannot answer for one subject must not cost the
            # others their nowcast, nor the whole cycle its forecast.
            continue
        fraction = presence_fraction(events, at)
        if fraction is None:
            continue
        fractions[subject.slug] = fraction
        held = features.minutes_in_state(events, index)[0]
        if not np.isnan(held):
            minutes[subject.slug] = float(held)

    if not fractions:
        return rows

    out = rows.copy()
    out["current_at"] = at.isoformat()

    slugs = out["subject"].map(fractions)
    out["state_now"] = slugs.where(slugs.notna(), out["state_now"])

    held = out["subject"].map(minutes)
    out["minutes_in_state"] = held.where(held.notna(), out["minutes_in_state"])

    # Keep the row internally consistent. Without this `state_now` says "just
    # left" while `other_alice` still reports where they were half an hour ago,
    # and the cross-subject features contradict the origin they sit beside.
    for slug, fraction in fractions.items():
        column = f"other_{slug}"
        if column in out.columns:
            out.loc[out["subject"] != slug, column] = fraction

    return out
