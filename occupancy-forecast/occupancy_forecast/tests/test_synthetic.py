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
    # Same cluster in both halves of the history: no routine change.
    #
    # Half a grid step, not a quarter. Departure hours are quantised to the
    # 30-minute grid, so two medians can differ by 0.25 h purely from which
    # side of a grid point the middle value falls -- that is the smallest
    # non-zero difference the data can express, and it is noise. The realistic
    # world's shift is 0.75 h, so this still separates the two cleanly.
    #
    # Compared on the MEAN rather than the median. A median of a variable
    # quantised to the 30-minute grid can only land on a grid point, so two
    # halves of one cluster differ by 0.0 or by a whole 0.5 h step and nothing
    # between -- there is no threshold that separates that from the realistic
    # world's 0.75 h shift. The mean is not quantised: with ~40 work days a half
    # and 0.35 h of jitter its standard error is about 0.06 h, so 0.25 h is four
    # standard errors of noise and a third of the shift being screened for.
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
    """Same seed, same draws: a non-working, non-holiday day is untouched by any
    of the three realistic extras, so the two worlds must agree there.

    This is what `_draws` buys. Drawing inside the branches let a holiday return
    early, desynchronise the generator and change every later day -- so the
    control was a different household rather than the same one without the
    extras. Holidays are excluded here because a holiday CAN fall on a Saturday
    (27 April 2024 does), and that difference is the point rather than a fault.
    """
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


def test_the_schedule_does_not_drift_across_a_dst_transition():
    """The generator writes a schedule in LOCAL time, so each day has to be
    anchored to its own local midnight.

    Anchoring the whole run to one UTC instant instead slides every departure by
    an hour at the March transition. That is not a cosmetic error: it invents
    seasonal drift, which a trailing feature can track and an expanding weekday
    median cannot, so a model appears to beat the baseline by 25-42% in the
    control world -- where beating it is supposed to be impossible. The fake
    result looked exactly like a real one."""
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
