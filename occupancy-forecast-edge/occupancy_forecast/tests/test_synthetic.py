"""The generator is measured against, so it has to be right.

A synthetic household with no tests is a liability: if it drifts, every
experiment run on it silently answers a different question than the one asked.
These pin the properties the experiments actually rely on.
"""
import numpy as np
import pandas as pd

from occupancy_forecast import departure
from occupancy_forecast.tests import synthetic


def _labelled(**kwargs):
    return departure.label_days(synthetic.household(**kwargs))


def test_the_control_world_contains_only_the_weekday_and_noise():
    """The leak detector's own premise. In `realistic=False` a person's work
    departures on one weekday must be one cluster -- no drift, no holidays, no
    partner effect -- because that is what makes the weekday median provably
    optimal there and a model beating it evidence of a leak."""
    days = _labelled(days=730, seed=3, realistic=False, missing=False)
    work = days[(days["subject"] == "alice") & (days["dow"] == 0)
                & days["left_today"]]
    assert len(work) > 40
    # Same cluster in both halves, on the MEAN: a median of a grid-quantised
    # variable lands on a grid point and cannot resolve the 0.75 h shift.
    first, second = np.array_split(work["departure_hour"].to_numpy(), 2)
    assert abs(np.mean(first) - np.mean(second)) < 0.25
    assert work["departure_hour"].std() < 2 * synthetic.WORK_JITTER_H


def test_the_realistic_world_moves_the_routine_part_way_through():
    """The thing an expanding median adapts to slowly and a model might not."""
    days = _labelled(days=730, seed=3, realistic=True, missing=False)
    work = days[(days["subject"] == "alice") & (days["dow"] == 0)
                & days["left_today"]].sort_values("date")
    first, second = np.array_split(work["departure_hour"].to_numpy(), 2)
    moved = np.median(second) - np.median(first)
    assert moved > 0.4, f"the routine change did not survive labelling: {moved:.2f} h"


def test_nobody_leaves_on_a_holiday_in_the_realistic_world():
    """The lookup has no calendar; this is one of the things a model could find."""
    days = _labelled(days=730, seed=3, realistic=True, missing=False)
    dates = pd.DatetimeIndex(days["date"])
    holiday = days[[(m, d) in synthetic.HOLIDAYS
                    for m, d in zip(dates.month, dates.day)]]
    assert len(holiday) > 4
    assert not holiday["left_today"].any()


def test_the_two_worlds_differ_only_where_they_are_meant_to():
    """Same seed, same draws: a non-working, non-holiday day is untouched by the
    realistic extras. Holidays are excluded because one can fall on a Saturday,
    and that difference is the point."""
    control = _labelled(days=365, seed=5, realistic=False, missing=False)
    realistic = _labelled(days=365, seed=5, realistic=True, missing=False)
    merged = control.merge(realistic, on=["subject", "date"], suffixes=("_c", "_r"))
    dates = pd.DatetimeIndex(merged["date"])
    holiday = [(m, d) in synthetic.HOLIDAYS for m, d in zip(dates.month, dates.day)]
    weekend = merged[(merged["dow_c"] == 5) & ~np.array(holiday)]
    assert len(weekend) > 40
    pd.testing.assert_series_equal(
        weekend["departure_hour_c"], weekend["departure_hour_r"],
        check_names=False)


def test_the_holes_are_real_enough_to_exercise_the_observability_rule():
    """The generator must produce days the label rule REFUSES, or an experiment
    on it never tests the part that protects against a recorder outage."""
    with_holes = _labelled(days=365, seed=7, realistic=True, missing=True)
    without = _labelled(days=365, seed=7, realistic=True, missing=False)
    assert without["candidate"].all()
    assert not with_holes["candidate"].all()


def _home(**kwargs) -> pd.DataFrame:
    """`home_frac` as (time x subject), holes filled the way a reader sees them."""
    frame = synthetic.household(**kwargs)
    # `pivot_table`, not `pivot`: on the spring-forward day an hour of the
    # timeline belongs to two local days and the timestamps repeat.
    wide = frame.pivot_table(index="time", columns="subject",
                             values="home_frac", aggfunc="last")
    return wide.ffill().bfill()


def _away_runs(values: np.ndarray) -> np.ndarray:
    """Lengths, in slots, of the runs where this person is out."""
    away = values <= 0.5
    edges = np.flatnonzero(np.diff(away.astype(int)))
    starts = np.r_[0, edges + 1]
    lengths = np.diff(np.r_[starts, away.size])
    return lengths[away[starts]]


def test_the_irregular_world_leaves_the_other_two_untouched():
    """The whole reason it is a third arm and not a change to the second:
    `realistic=True` is pinned against the control world, and a draw from the
    shared stream would change what every one of those comparisons asks."""
    plain = synthetic.household(days=120, seed=11, realistic=True, missing=False)
    also = synthetic.household(days=120, seed=11, realistic=True, missing=False,
                               irregular=False)
    pd.testing.assert_frame_equal(plain, also)

    control = synthetic.household(days=120, seed=11, realistic=False, missing=False)
    assert not control.equals(plain), "the two worlds must still differ"


def test_the_irregular_world_is_not_a_timetable():
    """The three properties the real archive has and a rota cannot produce: a
    short median absence, a multi-day longest one, and POSITIVE autocorrelation
    at every lag."""
    slots_per_hour = 2
    rota = _home(days=400, seed=7, realistic=True, missing=False)
    lived = _home(days=400, seed=7, realistic=True, missing=False, irregular=True)

    for subject in lived.columns:
        rota_runs = _away_runs(rota[subject].to_numpy())
        runs = _away_runs(lived[subject].to_numpy())

        # Short absences: errands, which a one-departure-a-day world has none of.
        assert np.median(runs) < np.median(rota_runs), subject

        # A trip. The timetable's longest absence is a single working day.
        assert runs.max() / slots_per_hour > 24, \
            f"{subject}: longest absence {runs.max()/slots_per_hour:.1f} h"

        # And it is a real share of the away time, not a rounding error.
        long = runs[runs > 24 * slots_per_hour]
        assert long.sum() / runs.sum() > 0.10, subject

        # The sign is the entire point: at half a day a timetable says "out now,
        # therefore in later", which makes persistence useless as a baseline.
        series = pd.Series(lived[subject].to_numpy())
        for lag_h in (8, 12, 24):
            assert series.autocorr(lag=lag_h * slots_per_hour) > 0.0, \
                f"{subject}: {lag_h} h autocorrelation is not positive"


def test_the_household_takes_its_trips_together():
    """One holiday, not one each. A shared away-block is a large part of why one
    person's presence predicts the other's days ahead; per-person trips would
    throw it away."""
    lived = _home(days=400, seed=7, realistic=True, missing=False, irregular=True)
    alice, bob = (lived[c].to_numpy() <= 0.5 for c in ("alice", "bob"))
    both = alice & bob
    runs = _away_runs(1.0 - both.astype(float))
    assert runs.max() > 24 * 2, \
        "no absence longer than a day is shared by the whole household"


def test_the_irregular_weekday_profile_is_not_two_flat_levels():
    """A rota kept exactly gives two flat levels -- work days and not -- and
    that gap is exactly what per-weekday climatology needs to be optimal."""
    def spread(frame):
        by_day = frame.groupby(pd.DatetimeIndex(frame.index).dayofweek).mean()
        return (by_day.max() - by_day.min()).max()

    rota = _home(days=400, seed=7, realistic=True, missing=False)
    lived = _home(days=400, seed=7, realistic=True, missing=False, irregular=True)
    assert spread(lived) < spread(rota)


def test_the_schedule_does_not_drift_across_a_dst_transition():
    """The generator writes a schedule in LOCAL time, so each day is anchored to
    its own local midnight. Anchoring the run to one UTC instant invents seasonal
    drift that a trailing feature can track and an expanding median cannot -- and
    the fake result looks exactly like a real one."""
    import numpy as np

    from occupancy_forecast.tests import synthetic

    days = _labelled(days=300, seed=4, realistic=False, missing=False)
    work = days[(days["subject"] == "alice") & (days["dow"] == 0)
                & days["left_today"]]
    winter = work[work["date"] < "2024-03-31"]["departure_hour"]
    summer = work[work["date"] > "2024-03-31"]["departure_hour"]
    assert len(winter) > 5 and len(summer) > 5

    scheduled = synthetic.SCHEDULES["alice"][0]
    for tag, half in (("winter", winter), ("summer", summer)):
        assert abs(half.mean() - scheduled) < 0.35, \
            f"{tag} departs at {half.mean():.2f}, scheduled {scheduled}"
