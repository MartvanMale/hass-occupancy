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

# --- what `irregular=True` adds, and why the first two worlds are not enough
#
# Both worlds above are a TIMETABLE: one departure per person-day, away for a
# fixed eight hours, back before midnight. MEASURED 2026-09-05 against a real
# 175-day archive of two people, resampled onto this same 30-minute grid, that
# is not what a household looks like:
#
#                             real (two people)      realistic=True
#     away episodes / day     1.42 / 0.68            0.66
#     median away             1.0 h / 1.5 h          8.0 h
#     longest away            526 h (BOTH people)    11.5 h
#     away time in runs >24 h 49% / 61%              0%
#     autocorrelation  +8 h   +0.43 / +0.55          -0.21
#                     +12 h   +0.35 / +0.44          -0.25
#                     +36 h   +0.34 / +0.41          -0.24
#     days with any absence   84% / 72%              66%
#
# The SIGN of that autocorrelation is the whole story. A rigid eight-hour block
# repeated daily is anti-correlated at half a day -- if you are out now you are
# reliably in twelve hours from now -- so `persistence` and `same_slot_yesterday`
# carry no information and the per-weekday lookup is unbeatable by construction.
# The consequence is measurable: this generator at 400 days shipped **5 of 48
# horizons**, while the same code on 175 days of the real archive shipped **43**,
# and there the winning baseline was `persistence` rather than climatology.
#
# So this arm adds the three things a timetable structurally cannot express, and
# they are the three the real archive actually shows:
#
#   * **errands** -- short, irregular, several a week, at no fixed hour. This is
#     what makes the median absence one hour rather than eight.
#   * **trips** -- a holiday. Rare, days long, and taken by the WHOLE household
#     at once: in the real archive both people's longest absence is the same
#     526 hours, which is one trip and not two. Half of all away time is in
#     runs longer than a day, and nothing in a weekday lookup can reach it.
#   * **a schedule that is kept loosely** -- a work day skipped, an unscheduled
#     day worked. The real weekday profile spans 0.62..0.84, not the clean
#     0.67 / 0.87 split of a rota, and that flatness is what stops climatology
#     from being the optimal predictor.
#
# Deliberately a THIRD arm rather than a change to `realistic=True`. That world
# is pinned by tests -- on a non-working, non-holiday day it must agree with the
# control world exactly -- and it is the leak detector's other half. This layers
# on top of it, from its own random stream, so both existing worlds are
# untouched bit for bit.
IRREGULAR_ERRANDS_PER_DAY = 1.5   # Poisson mean, per person
IRREGULAR_ERRAND_MEDIAN_H = 1.0   # lognormal median; p90 lands near 4 h
IRREGULAR_ERRAND_SIGMA = 1.05
IRREGULAR_ERRAND_HOURS = (9.0, 21.0)   # when an errand can start

# Errands cluster. A slow AR(1) on the daily rate gives busy stretches and quiet
# ones, which is dependence a per-weekday lookup cannot hold: it is the reason
# yesterday tells you something about tomorrow beyond which weekday it is.
IRREGULAR_BUSY_RHO = 0.85
IRREGULAR_BUSY_SIGMA = 0.8
# ...and most of it belongs to the HOUSEHOLD rather than to one person. This is
# the everyday version of the shared trip: a busy week is busy for both of them,
# so one person's recent activity says something about the other's tomorrow that
# no per-weekday, per-person lookup can hold. It is available in every fold,
# which is what the trips are not.
IRREGULAR_BUSY_SHARED = 0.6

# Trips are the biggest single term and the easiest to get wrong in BOTH
# directions. Sized to the real archive's 12%-of-the-record they became one
# enormous holiday, and MEASURED 2026-09-05 that is worse than useless to the
# ship gate: skill was positive at every horizon (9.5..27.6%) and only 9 of 48
# horizons shipped, because the model won 18 folds of 51 -- it won hugely in the
# two folds the holiday touched and lost narrowly everywhere else, and
# `train.fold_record_allows` refuses exactly that. The real archive escapes the
# same fate only because 175 days is 19 folds, where a minority that size cannot
# be proven. So: the same share of away time, cut into many more trips, so the
# long-range structure appears in most folds instead of two.
IRREGULAR_TRIPS_PER_YEAR = 8.0
IRREGULAR_TRIP_MEDIAN_DAYS = 4.5
IRREGULAR_TRIP_SIGMA = 0.6
IRREGULAR_TRIP_LEAVE_H = 9.0
# Most trips start at a weekend, because that is when people go away. It also
# makes their timing PREDICTABLE rather than uniform noise, which is what gives
# the long horizons something learnable: a trip that can begin on any day with
# equal probability contributes variance and no signal.
IRREGULAR_TRIP_WEEKEND = 0.75
IRREGULAR_WORKDAY_SKIP = 0.15     # a scheduled day not worked
IRREGULAR_WORKDAY_EXTRA = 0.10    # an unscheduled weekday worked
IRREGULAR_AWAY_SIGMA = 0.30       # the work day itself is not exactly 8 hours

# Its own stream, so adding this arm cannot shift a single draw in the other
# two. `_draws` is deliberately left alone for the same reason.
IRREGULAR_STREAM = 9001

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


def _keep_schedule_loosely(rng, subject: str, day: dt.date,
                           left: float | None) -> tuple[float | None, bool]:
    """A rota kept loosely: `(departure hour or None, was it a work day)`.

    Both rolls are drawn unconditionally, for the same reason `_draws` does it:
    a branch that consumes a different number of numbers desynchronises every
    later day, and then a change to one constant silently rewrites the whole
    history rather than the part it names.

    This is the piece that flattens the weekday profile. The real archive's
    home-rate by weekday spans 0.62..0.84 with no clean split; a rota kept
    exactly gives 0.67 / 0.87, two flat levels, which is a lookup table and is
    precisely what makes climatology unbeatable.
    """
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
    """Durations in slots, heavy-tailed, at least one slot long.

    Lognormal because that is the shape the real archive has: a median of about
    an hour with a tail that reaches a working day. A normal fitted to the same
    mean would put almost no mass past three hours and the p90 would be wrong by
    a factor of three.
    """
    hours = median_h * np.exp(sigma * rng.standard_normal(size))
    return np.maximum(1, np.round(hours * per_hour).astype(int))


def trip_mask(rng, days: int, slots: int, per_hour: int,
              first_weekday: int = 0) -> np.ndarray:
    """Slots the whole household is away for a holiday.

    Household-level and not per-person on purpose. In the real archive both
    people's single longest absence is the same 526 hours -- one trip, taken
    together -- and that shared block is a large part of why one person's
    presence predicts the other's days ahead. Drawn per household and applied to
    everyone identically.
    """
    mask = np.zeros(days * slots, dtype=bool)
    expected = IRREGULAR_TRIPS_PER_YEAR * days / 365.0
    for _ in range(rng.poisson(expected)):
        length_days = float(IRREGULAR_TRIP_MEDIAN_DAYS
                            * np.exp(IRREGULAR_TRIP_SIGMA * rng.standard_normal()))
        run = max(per_hour, int(round(length_days * slots)))
        start_day = int(rng.integers(0, max(1, days)))
        if rng.random() < IRREGULAR_TRIP_WEEKEND:
            # Forward to the next Saturday. Trips that can start on any day with
            # equal probability are variance without signal; a household that
            # goes away at weekends is both truer and learnable.
            ahead = (5 - (first_weekday + start_day) % 7) % 7
            start_day = min(days - 1, start_day + ahead)
        at = start_day * slots + int(IRREGULAR_TRIP_LEAVE_H * per_hour)
        mask[at:min(at + run, mask.size)] = True
    return mask


def busyness(rng, days: int) -> np.ndarray:
    """A slow AR(1) over days, standardised to unit variance.

    Standardised so that changing RHO changes the CLUSTERING and nothing else.
    Left to itself an AR(1)'s variance grows with the correlation, so the two
    constants would fight and the household's overall rate would drift with a
    knob that is supposed to be about timing.
    """
    z = np.zeros(days)
    for index in range(1, days):
        z[index] = IRREGULAR_BUSY_RHO * z[index - 1] + rng.standard_normal()
    return z * np.sqrt(1.0 - IRREGULAR_BUSY_RHO ** 2)


def errand_mask(rng, days: int, slots: int, per_hour: int,
                shared: np.ndarray | None = None) -> np.ndarray:
    """Slots this person is out on something short and unscheduled.

    The count is Poisson per day rather than a fixed rate so that some days have
    none and some have three, which is what makes the daily absence rate 80-odd
    percent without making every day look the same.
    """
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
            # The tracked zone, as distinct from any other reason to be out.
            # Only a work day counts -- leisure is an absence too, and it is
            # exactly what must not be mixed in. Under `irregular` that means a
            # day actually worked rather than a day the rota says to work: a
            # skipped Monday is not a zone day, and an extra Wednesday is.
            if worked:
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

        if irregular:
            # Applied to the whole timeline rather than day by day, because a
            # trip does not respect midnight -- which is the entire point of it.
            # The recorder holes are re-applied afterwards: a hole is "we do not
            # know", and being out on an errand does not make it known.
            unknown = np.isnan(home_all)
            # `& was_home`: an errand that lands inside a work absence is
            # DISCARDED rather than merged into it. Merging is what kept the
            # median absence at five and a half hours when the real archive's is
            # one -- the errand has to be its own short episode to count.
            was_home = home_all > 0.5
            errands = errand_mask(odd, days, slots, per_hour,
                                  shared=shared_busy) & was_home
            home_all[errands | trips] = 0.0
            # Only the trip clears the zone. An errand that overlaps a work
            # absence must not carve a hole in the work-zone signal -- the
            # person is still at work -- and everywhere else the zone is already
            # zero, because it is only ever set where `home` is.
            zone_all[trips] = 0.0
            home_all[unknown] = np.nan

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
