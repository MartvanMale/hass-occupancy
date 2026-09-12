"""What is true RIGHT NOW, for the serving row only.

NEVER call this from `features.build`. The in-progress slot reacts 0-15 minutes
late, so this recomputes the ORIGIN BLOCK ONLY over five minutes; the accepted
skew is serving `state_now` from 5 minutes where training drew it from 30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features

# A window length, not a tuned dwell threshold: a blip under 2.5 minutes cannot
# reach half of it, and a real change shows 2.5 minutes in.
WINDOW_MINUTES = 5

# How far back to read events. Only needs to cover the window plus enough slack
# for `seeded_states` to find a prior value, which it does by seeding anyway.
LOOKBACK_MINUTES = 6 * 60


def presence_fraction(events: list[tuple[str, str]], at: pd.Timestamp,
                      window_min: int = WINDOW_MINUTES) -> float | None:
    """Fraction of the `window_min` minutes ENDING at `at` spent at home: the
    same definition and integration as `home_frac`, over a shorter window.
    """
    if not events:
        return None
    window = pd.DatetimeIndex([at - pd.Timedelta(minutes=window_min)], tz="UTC")
    out = features.slot_fraction(events, window, config.HOME_STATE, window_min)
    value = out["frac"].iloc[0]
    return None if pd.isna(value) else float(value)


def apply(rows: pd.DataFrame, source, at: pd.Timestamp) -> pd.DataFrame:
    """Move the origin block of each serving row forward to `at`. A subject with
    no nowcast keeps its grid values, never NaN, or it loses all 48 horizons;
    everything outside the origin block is left exactly as built.
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
            # One subject's failed read must not cost the others their nowcast.
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

    # Keep the row consistent, or `state_now` says "just left" while the
    # `other_*` columns report half an hour ago.
    for slug, fraction in fractions.items():
        column = f"other_{slug}"
        if column in out.columns:
            out.loc[out["subject"] != slug, column] = fraction

    return out
