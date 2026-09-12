"""A two-person household with a known routine, belonging to nobody.
`realistic=False` is weekday-plus-noise and the leak detector: a model beating
the per-weekday median there has seen something it should not. `realistic=True`
adds holidays, drift and coupling; `irregular=True` errands, trips and a rota.
Output is what `departure.label_days` reads, plus `zone_work` (a TRACKED ZONE).
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

# `irregular=True`: errands, household trips and a loosely kept rota, because a
# timetable is anti-correlated at half a day and makes climatology unbeatable.
IRREGULAR_ERRANDS_PER_DAY = 1.5   # Poisson mean, per person
IRREGULAR_ERRAND_MEDIAN_H = 1.0   # lognormal median; p90 lands near 4 h
IRREGULAR_ERRAND_SIGMA = 1.05
IRREGULAR_ERRAND_HOURS = (9.0, 21.0)   # when an errand can start

# Errands cluster: a slow AR(1) gives busy stretches, which is dependence a
# per-weekday lookup cannot hold.
IRREGULAR_BUSY_RHO = 0.85
IRREGULAR_BUSY_SIGMA = 0.8
# Mostly the HOUSEHOLD's busyness, not one person's -- available in every fold,
# which the trips are not.
IRREGULAR_BUSY_SHARED = 0.6

# Many small trips rather than one long one, so the long-range structure lands
# in most folds instead of two, which `train.fold_record_allows` would refuse.
IRREGULAR_TRIPS_PER_YEAR = 8.0
IRREGULAR_TRIP_MEDIAN_DAYS = 4.5
IRREGULAR_TRIP_SIGMA = 0.6
IRREGULAR_TRIP_LEAVE_H = 9.0
# Weekend starts, so trip timing is predictable rather than uniform noise the
# long horizons cannot learn.
IRREGULAR_TRIP_WEEKEND = 0.75
IRREGULAR_WORKDAY_SKIP = 0.15     # a scheduled day not worked
IRREGULAR_WORKDAY_EXTRA = 0.10    # an unscheduled weekday worked
IRREGULAR_AWAY_SIGMA = 0.30       # the work day itself is not exactly 8 hours

# Its own stream, so adding this arm cannot shift a single draw in the other
# two. `_draws` is deliberately left alone for the same reason.
IRREGULAR_STREAM = 9001

# `shifts=True`: two start times per work day, bimodal by construction, so a
# per-weekday median cannot express it; only the alarm says which mode is next.
SHIFT_HOURS = (6.75, 8.25)      # early and late, an hour and a half apart
SHIFT_BLOCK_DAYS = 4            # a rota holds for about this many work days
SHIFT_SUBJECT = "alice"

# How long before leaving the alarm goes off, and how tightly it tracks.
ALARM_LEAD_H = 1.5
ALARM_NOISE_H = 0.15
# Set at 20:00 the night before, as the real sensor does -- which is why a 04:00
# origin can read it at all.
ALARM_SET_HOUR = 20.0

# Days with a real hole, so the observability rule is exercised, not assumed.
MISSING_DAY_CHANCE = 0.12
MISSING_RUN_SLOTS = (3, 10)


def _draws(rng) -> dict:
    """Every random number one person-day needs, drawn UNCONDITIONALLY so the
    two worlds consume the stream identically; an early return desyncs them."""
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
    """Which of two start times this work day is on, or None. Keyed on the
    BLOCK (a rota holds for a run) and on its own stream, so `_draws` stays in
    lockstep."""
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
    """When the phone alarm is set for, or None. `fidelity` blends the day's
    actual departure with that weekday's typical one: 0 adds nothing to the
    weekday median, 1 tracks the day being predicted. No alarm on a non-work
    day: that absence is itself a signal, as it is on the real sensor."""
    if left is None or day.weekday() not in SCHEDULES[subject]:
        return None
    typical = SCHEDULES[subject][day.weekday()]
    target = fidelity * left + (1.0 - fidelity) * typical
    return float(target - ALARM_LEAD_H + draws["alarm_noise"])


def _keep_schedule_loosely(rng, subject: str, day: dt.date,
                           left: float | None) -> tuple[float | None, bool]:
    """A rota kept loosely: `(departure hour or None, was it a work day)`. Both
    rolls drawn unconditionally, for the same reason `_draws` does it. This is the
    piece that flattens the weekday profile so climatology is not optimal."""
    skip_roll, extra_roll = rng.random(), rng.random()
    scheduled = day.weekday() in SCHEDULES[subject]
    if scheduled:
        if left is not None and skip_roll < IRREGULAR_WORKDAY_SKIP:
            return None, False           # did not go in today
        return left, left is not None
    # An unscheduled weekday worked anyway. Weekends are left out of this: a
    # Saturday shift is a different claim about the household than a busy week.
    if (day.weekday() < 5 and left is None
            and extra_roll < IRREGULAR_WORKDAY_EXTRA):
        return SCHEDULES[subject][min(SCHEDULES[subject])], True
    return left, False


def _lognormal_slots(rng, median_h: float, sigma: float, per_hour: int,
                     size: int) -> np.ndarray:
    """Durations in slots, heavy-tailed, at least one slot long. Lognormal because
    the real shape is a ~1 h median with a tail reaching a working day."""
    hours = median_h * np.exp(sigma * rng.standard_normal(size))
    return np.maximum(1, np.round(hours * per_hour).astype(int))


def trip_mask(rng, days: int, slots: int, per_hour: int,
              first_weekday: int = 0) -> np.ndarray:
    """Slots the whole household is away for a holiday. Household-level, not
    per-person: a shared block is a large part of why one person's presence
    predicts the other's days ahead."""
    mask = np.zeros(days * slots, dtype=bool)
    expected = IRREGULAR_TRIPS_PER_YEAR * days / 365.0
    for _ in range(rng.poisson(expected)):
        length_days = float(IRREGULAR_TRIP_MEDIAN_DAYS
                            * np.exp(IRREGULAR_TRIP_SIGMA * rng.standard_normal()))
        run = max(per_hour, int(round(length_days * slots)))
        start_day = int(rng.integers(0, max(1, days)))
        if rng.random() < IRREGULAR_TRIP_WEEKEND:
            # Forward to Saturday: a weekend start is truer, and learnable.
            ahead = (5 - (first_weekday + start_day) % 7) % 7
            start_day = min(days - 1, start_day + ahead)
        at = start_day * slots + int(IRREGULAR_TRIP_LEAVE_H * per_hour)
        mask[at:min(at + run, mask.size)] = True
    return mask


def busyness(rng, days: int) -> np.ndarray:
    """A slow AR(1) over days, standardised to unit variance so RHO changes the
    CLUSTERING and nothing else."""
    z = np.zeros(days)
    for index in range(1, days):
        z[index] = IRREGULAR_BUSY_RHO * z[index - 1] + rng.standard_normal()
    return z * np.sqrt(1.0 - IRREGULAR_BUSY_RHO ** 2)


def errand_mask(rng, days: int, slots: int, per_hour: int,
                shared: np.ndarray | None = None) -> np.ndarray:
    """Slots this person is out on something short and unscheduled. Poisson per
    day rather than a fixed rate, so some days have none and some have three."""
    mask = np.zeros(days * slots, dtype=bool)
    z = busyness(rng, days)
    if shared is not None:
        # Blended so the result still has unit variance: turning the shared
        # fraction up must change WHOSE busyness it is, not how much there is.
        z = (IRREGULAR_BUSY_SHARED * shared
             + np.sqrt(1.0 - IRREGULAR_BUSY_SHARED ** 2) * z)
    rate = IRREGULAR_ERRANDS_PER_DAY * np.exp(
        IRREGULAR_BUSY_SIGMA * z - IRREGULAR_BUSY_SIGMA ** 2 / 2)

    counts = rng.poisson(rate)
    total = int(counts.sum())
    if total == 0:
        return mask
    lo, hi = IRREGULAR_ERRAND_HOURS
    day_index = np.repeat(np.arange(days), counts)
    hour = rng.uniform(lo, hi, size=total)
    runs = _lognormal_slots(rng, IRREGULAR_ERRAND_MEDIAN_H,
                            IRREGULAR_ERRAND_SIGMA, per_hour, total)
    for index, at_hour, run in zip(day_index, hour, runs):
        at = int(index) * slots + int(round(at_hour * per_hour))
        mask[at:min(at + int(run), mask.size)] = True
    return mask


def household(days: int = 730, seed: int = 0, realistic: bool = True,
              start: str = "2024-01-01", missing: bool = True,
              shifts: bool = False, irregular: bool = False,
              alarm_fidelity: float | None = None) -> pd.DataFrame:
    """`subject, time, home_frac` for two people over `days` days. With
    `alarm_fidelity` set, a `next_alarm_h` column in exactly the shape
    `features._add_next_alarm` produces -- so a probe written here runs unchanged
    against the real parquet."""
    rng = np.random.default_rng(seed)
    begin = dt.date.fromisoformat(start)
    slots = config.SLOTS_PER_DAY
    per_hour = 60 // config.GRID_MINUTES
    want_alarm = alarm_fidelity is not None
    # One trip stream for the household, drawn before anyone's days so that
    # every subject gets the SAME holiday rather than one of their own.
    trips, shared_busy = None, None
    if irregular:
        house = np.random.default_rng([seed, IRREGULAR_STREAM])
        trips = trip_mask(house, days, slots, per_hour, begin.weekday())
        shared_busy = busyness(house, days)
    rows = []
    for position, subject in enumerate(SCHEDULES):
        home_all = np.empty(days * slots)
        zone_all = np.zeros(days * slots)
        alarm_all = np.full(days * slots, np.nan)
        # Per-subject and independent of `rng`, so `irregular` cannot move a
        # single draw in the control or realistic worlds.
        odd = np.random.default_rng([seed, IRREGULAR_STREAM, position])
        for index in range(days):
            day = begin + dt.timedelta(days=index)
            draws = _draws(rng)
            left = departure_hour(draws, subject, day, index, days, realistic,
                                  shifts=shifts, seed=seed)
            worked = left is not None and day.weekday() in SCHEDULES[subject]
            if irregular:
                left, worked = _keep_schedule_loosely(odd, subject, day, left)
            home = np.ones(slots)
            if left is not None:
                out = max(0, int(round(left * per_hour)))
                away = AWAY_HOURS
                if irregular:
                    # A work day is not exactly eight hours either.
                    away *= float(np.exp(IRREGULAR_AWAY_SIGMA
                                         * odd.standard_normal()))
                back = min(slots, out + int(away * per_hour)
                           + int(draws["away_jitter"]))
                home[out:back] = 0.0
            if missing and draws["hole_roll"] < MISSING_DAY_CHANCE:
                run = draws["hole_run"]
                at = int(draws["hole_at"] * (slots - run))
                home[at:at + run] = np.nan
            home_all[index * slots:(index + 1) * slots] = home
            # Only a day actually worked counts as a zone day -- leisure is an
            # absence too, and it is what must not be mixed in.
            if worked:
                zone_all[index * slots:(index + 1) * slots] = np.nan_to_num(
                    1.0 - home)

            if not want_alarm:
                continue
            rings = alarm_hour(draws, subject, day, left, alarm_fidelity)
            if rings is None:
                continue
            # Written on the shared timeline, not this day's block: the evening
            # half of the window belongs to the previous day.
            fires = index * slots + int(round(rings * per_hour))
            set_at = (index - 1) * slots + int(round(ALARM_SET_HOUR * per_hour))
            lo, hi = max(0, set_at), min(days * slots, fires)
            for cell in range(lo, hi):
                alarm_all[cell] = (fires - cell) / per_hour

        if irregular:
            # Applied to the whole timeline, because a trip does not respect
            # midnight; recorder holes are re-applied after.
            unknown = np.isnan(home_all)
            # `& was_home`: an errand inside a work absence is DISCARDED, not
            # merged -- it has to be its own short episode to count.
            was_home = home_all > 0.5
            errands = errand_mask(odd, days, slots, per_hour,
                                  shared=shared_busy) & was_home
            home_all[errands | trips] = 0.0
            # Only a trip clears the zone: an errand overlapping a work absence
            # must not carve a hole in the work signal.
            zone_all[trips] = 0.0
            home_all[unknown] = np.nan

        # LOCAL midnight per day: this writes a local-time schedule, and one UTC
        # anchor would slide every departure an hour at the March transition.
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
