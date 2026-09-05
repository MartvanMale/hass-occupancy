"""A household with a schedule we know, for asking what more data would buy.

The real archive holds ~100 departures per person, which is too few to tell
"this model is wrong" apart from "this model is starving". A household whose
routine is known by construction separates the two: generate a year, generate
five, and watch whether the model overtakes the baseline.

**Two worlds, and the control is not a formality.**

`realistic=False` is a weekday schedule plus jitter and nothing else. In that
world the per-weekday median IS the optimal predictor -- it is the maximum
likelihood estimate of the only structure present -- so a model CANNOT
legitimately beat it. That arm is the leak detector: if a model wins there, it
has seen something it should not have, and the result is about the harness
rather than the model.

`realistic=True` adds three things a weekday lookup structurally cannot express,
so that there is something for a model to find:

  * **holidays** -- nobody leaves, and the lookup has no notion of a calendar
  * **a change of routine part way through** -- one person's hour shifts, and an
    expanding median over all history adapts to that slowly
  * **partner coupling** -- one leaves earlier on the days the other works,
    which is a fact about the household rather than about the weekday

Alice and Bob, to match `conftest.settings()`. The whole point of a synthetic
household here is that it belongs to nobody, which is what stops one particular
installation creeping into the package.

Output is what `departure.label_days` reads: `subject`, `time`, `home_frac` on
the 30-minute grid, holes included -- plus `zone_work`, which says which of
those absences were to a TRACKED ZONE.

That last column matters more than it looks. A person leaves the house for the
workplace, the gym and the school run, and the hour is a different question
for each; the real archive's `departure_hour` is the first departure of ANY
kind, so its per-weekday median lands between the modes at an hour nobody
leaves. On a real archive that spread measured 3.50 h over all departures and
**0.74 h over days out to the work zone alone**. Splitting them is the
difference between an unanswerable question and a half-hour one, so the
generator has to be able to pose both.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .. import config

# Who works which days, and the hour they leave. Weekday index, Monday = 0.
SCHEDULES: dict[str, dict[int, float]] = {
    "alice": {0: 7.25, 1: 7.25, 3: 7.25},          # Mon, Tue, Thu
    "bob": {0: 8.0, 2: 8.0, 4: 8.0},               # Mon, Wed, Fri
}

WORK_JITTER_H = 0.35            # about twenty minutes either side
LEISURE_CHANCE = 0.40           # going out on a day they do not work
LEISURE_HOUR = 11.0
LEISURE_JITTER_H = 1.5
AWAY_HOURS = 8.0

# What `realistic=True` adds, and `realistic=False` deliberately does not.
ROUTINE_CHANGE_AT = 0.55        # fraction through the history
ROUTINE_SHIFT_H = 0.75          # alice starts leaving 45 minutes later
COUPLING_H = -0.25              # bob leaves 15 minutes early when alice works
HOLIDAYS = {(1, 1), (4, 27), (5, 5), (12, 25), (12, 26)}

# --- shifts and the alarm, for the "would a leaves-by model be feasible" study
#
# On a real archive the two housemates' alarms split into two cases, and the
# difference is the whole question:
#
#   one   07:00, 07:00, 07:00   constant -- says exactly what that person's
#                               weekday median already says, so it cannot help
#   other 06:15, 07:30, 06:15   varies by 1h15m night to night, so it MIGHT
#
# `shifts=True` gives alice two start times chosen per work day, which is the
# structure a per-weekday median cannot express by construction: the median of
# a bimodal weekday is a point neither mode sits on. The alarm is then the only
# thing that says which mode tomorrow is.
#
# Blocks rather than independent coin flips, because real shift rotas run in
# runs -- and a run is something a trailing window can partly learn, so this is
# the harder and more honest test.
SHIFT_HOURS = (6.75, 8.25)      # early and late, an hour and a half apart
SHIFT_BLOCK_DAYS = 4            # a rota holds for about this many work days
SHIFT_SUBJECT = "alice"

# How long before leaving the alarm goes off, and how tightly it tracks.
ALARM_LEAD_H = 1.5
ALARM_NOISE_H = 0.15
# The evening the alarm becomes visible. Set at 20:00 the night before, which
# is what the real sensor does -- it is why this is a LEADING indicator and the
# reason a 04:00 origin can read it at all.
ALARM_SET_HOUR = 20.0

# Days with a real hole, so the observability rule is exercised rather than
# assumed. The real archive runs at about 18%.
MISSING_DAY_CHANCE = 0.12
MISSING_RUN_SLOTS = (3, 10)


def _draws(rng) -> dict:
    """Every random number one person-day needs, drawn UNCONDITIONALLY.

    Drawn up front rather than inside the branches, so the two worlds consume
    the stream identically. Otherwise a holiday returns early in `realistic`,
    the generator falls out of step, and the control stops being the same
    household minus the extras -- it becomes a different roll of the dice, which
    is not a control at all.
    """
    return {
        "work_jitter": rng.normal(0, WORK_JITTER_H),
        "leisure_roll": rng.random(),
        "leisure_jitter": rng.normal(0, LEISURE_JITTER_H),
        "away_jitter": rng.normal(0, 3),
        "hole_roll": rng.random(),
        "hole_run": int(rng.integers(*MISSING_RUN_SLOTS)),
        "hole_at": rng.random(),
        "shift_roll": rng.random(),
        "alarm_noise": rng.normal(0, ALARM_NOISE_H),
        "alarm_reveal": rng.random(),
    }


def shift_start(subject: str, index: int, seed: int) -> float | None:
    """Which of two start times this work day is on, or None if not on a rota.

    Drawn from a stream keyed on the BLOCK rather than the day, and separate
    from the main one on purpose. Keyed on the block because a rota holds for a
    run of days; separate because `_draws` must stay in lockstep between the two
    worlds, and consuming a draw here only when shifts are on would break that.
    """
    if subject != SHIFT_SUBJECT:
        return None
    block = index // SHIFT_BLOCK_DAYS
    roll = np.random.default_rng([seed, 7919, block]).random()
    return SHIFT_HOURS[0] if roll < 0.5 else SHIFT_HOURS[1]


def departure_hour(draws: dict, subject: str, day: dt.date, index: int,
                   total: int, realistic: bool, shifts: bool = False,
                   seed: int = 0) -> float | None:
    """When this person leaves, or None if they stay in. The ground truth."""
    schedule = SCHEDULES[subject]
    if realistic and (day.month, day.day) in HOLIDAYS:
        return None
    if day.weekday() in schedule:
        base = schedule[day.weekday()]
        if shifts:
            rota = shift_start(subject, index, seed)
            if rota is not None:
                base = rota
        hour = base + draws["work_jitter"]
        if realistic:
            if subject == "alice" and index / max(total, 1) >= ROUTINE_CHANGE_AT:
                hour += ROUTINE_SHIFT_H
            if subject == "bob" and day.weekday() in SCHEDULES["alice"]:
                hour += COUPLING_H
        return float(hour)
    if draws["leisure_roll"] < LEISURE_CHANCE:
        return float(LEISURE_HOUR + draws["leisure_jitter"])
    return None


def alarm_hour(draws: dict, subject: str, day: dt.date, left: float | None,
               fidelity: float) -> float | None:
    """When the phone alarm is set for, or None if none is set.

    **`fidelity` is the whole experiment.** It blends the day's ACTUAL departure
    with that weekday's typical one:

        fidelity = 0   the alarm sits at the typical hour every time. That is
                       the constant case -- 07:00, 07:00, 07:00 -- and it says
                       exactly what that person's weekday median already says,
                       so a model can gain nothing from it however much history
                       it has.
        fidelity = 1   the alarm tracks the day being predicted. That is the
                       best a varying 06:15 / 07:30 / 06:15 could possibly be.

    Sweeping it answers the question the real archive cannot yet: how faithful
    does an alarm have to be before a model beats the lookup, and how many days
    of it are needed.

    No alarm on a day they do not go to work -- not for leisure, not on a
    holiday, not on a day they stay in. That absence is itself a signal, and it
    is the honest behaviour of the real sensor, which reads `absent`.
    """
    if left is None or day.weekday() not in SCHEDULES[subject]:
        return None
    typical = SCHEDULES[subject][day.weekday()]
    target = fidelity * left + (1.0 - fidelity) * typical
    return float(target - ALARM_LEAD_H + draws["alarm_noise"])


def household(days: int = 730, seed: int = 0, realistic: bool = True,
              start: str = "2024-01-01", missing: bool = True,
              shifts: bool = False,
              alarm_fidelity: float | None = None) -> pd.DataFrame:
    """`subject, time, home_frac` for two people over `days` days.

    With `alarm_fidelity` set, a `next_alarm_h` column comes too, in exactly the
    shape `features._add_next_alarm` produces on the real archive -- hours from
    each slot until the alarm, NaN when none is set. Same column name and same
    units, so a probe written against this runs unchanged against the real
    parquet the moment there is enough history.
    """
    rng = np.random.default_rng(seed)
    begin = dt.date.fromisoformat(start)
    slots = config.SLOTS_PER_DAY
    per_hour = 60 // config.GRID_MINUTES
    want_alarm = alarm_fidelity is not None
    rows = []
    for subject in SCHEDULES:
        home_all = np.empty(days * slots)
        zone_all = np.zeros(days * slots)
        alarm_all = np.full(days * slots, np.nan)
        for index in range(days):
            day = begin + dt.timedelta(days=index)
            draws = _draws(rng)
            left = departure_hour(draws, subject, day, index, days, realistic,
                                  shifts=shifts, seed=seed)
            home = np.ones(slots)
            if left is not None:
                out = max(0, int(round(left * per_hour)))
                back = min(slots, out + int(AWAY_HOURS * per_hour)
                           + int(draws["away_jitter"]))
                home[out:back] = 0.0
            if missing and draws["hole_roll"] < MISSING_DAY_CHANCE:
                run = draws["hole_run"]
                at = int(draws["hole_at"] * (slots - run))
                home[at:at + run] = np.nan
            home_all[index * slots:(index + 1) * slots] = home
            # The tracked zone, as distinct from any other reason to be out.
            # Only a
            # scheduled work day counts -- leisure is an absence too, and it is
            # exactly what must not be mixed in.
            if left is not None and day.weekday() in SCHEDULES[subject]:
                zone_all[index * slots:(index + 1) * slots] = np.nan_to_num(
                    1.0 - home)

            if not want_alarm:
                continue
            rings = alarm_hour(draws, subject, day, left, alarm_fidelity)
            if rings is None:
                continue
            # Visible from 20:00 the night before until it goes off. Written
            # onto the shared timeline rather than this day's block, because the
            # evening half of that window belongs to the PREVIOUS day -- which
            # is exactly what makes a 04:00 origin able to read it.
            fires = index * slots + int(round(rings * per_hour))
            set_at = (index - 1) * slots + int(round(ALARM_SET_HOUR * per_hour))
            lo, hi = max(0, set_at), min(days * slots, fires)
            for cell in range(lo, hi):
                alarm_all[cell] = (fires - cell) / per_hour

        # LOCAL midnight per day, and this is the one that has already been got
        # wrong once. `features.grid` is uniform in UTC, so matching it looks
        # like the faithful choice -- but that grid READS events that already
        # happened, while this WRITES a schedule expressed in local time. Anchor
        # the whole run to one UTC instant and every departure slides an hour at
        # the March transition, which shows up as seasonal drift that no
        # schedule contains: alice's 07:15 became a 07:52 mean spread over two
        # hours, the expanding weekday median could not track it, and a model
        # with trailing features "beat" it by 25-42% in the CONTROL world, where
        # beating it is supposed to be impossible.
        times = [(pd.Timestamp(begin + dt.timedelta(days=index),
                               tz=config.TIMEZONE)
                  + pd.Timedelta(minutes=config.GRID_MINUTES * slot)
                  ).tz_convert("UTC")
                 for index in range(days) for slot in range(slots)]
        block = {"subject": subject, "time": times, "home_frac": home_all,
                 "zone_work": zone_all}
        if want_alarm:
            block["next_alarm_h"] = alarm_all
        rows.append(pd.DataFrame(block))
    return pd.concat(rows, ignore_index=True)
