"""When the forecast counts as a departure or an arrival.

`predict._crossing` reduces the whole 48 h curve to the two numbers a household
actually reads -- "hours until away", "hours until home" -- and it had no tests
at all. The rule it implements is not obvious: a threshold per direction, a run
requirement, a gate on the observed present, and three boundary decisions that
each go the less obvious way on purpose.

Everything here builds curves by hand. Nothing needs a model, a broker or a
Home Assistant.
"""

import pandas as pd
import pytest

from occupancy_forecast import config, predict


def row(state_now: float) -> pd.Series:
    return pd.Series({"subject": "alice", "state_now": state_now})


def curve(*values: float, start: int = 1) -> dict[int, float]:
    """`{start: values[0], start+1: values[1], ...}` -- an hourly curve."""
    return {start + i: v for i, v in enumerate(values)}


HOME, AWAY = row(1.0), row(0.0)


def departure(curve_, min_hours=2, threshold=0.5, at=HOME):
    return predict._crossing(at, curve_, going_home=False,
                             threshold=threshold, min_hours=min_hours)


def arrival(curve_, min_hours=2, threshold=0.5, at=AWAY):
    return predict._crossing(at, curve_, going_home=True,
                             threshold=threshold, min_hours=min_hours)


# --- the reason this change exists ---------------------------------------

def test_a_single_hour_across_the_line_is_not_a_departure():
    """The old bug. One dip does not make a departure.

    The forecast could not have meant a short absence even if it wanted to: its
    target is the fraction of a 30-minute slot spent home, so a walk round the
    block reads as ~0.5 in two adjacent slots and never crosses. A lone hour
    past the line is therefore noise by construction.
    """
    dip = curve(0.95, 0.9, 0.49, 0.88, 0.9, 0.92)
    assert departure(dip, min_hours=2) is None
    assert departure(dip, min_hours=1) == 3


def test_a_wobble_before_a_real_departure_is_stepped_over():
    both = curve(0.9, 0.9, 0.45, 0.9, 0.9, 0.2, 0.1, 0.1)
    assert departure(both, min_hours=2) == 6


def test_the_answer_is_the_first_hour_of_the_run_not_the_last():
    real = curve(0.9, 0.9, 0.9, 0.9, 0.2, 0.15, 0.1, 0.1)
    assert departure(real, min_hours=3) == 5


# --- the two cuts ---------------------------------------------------------

def test_each_direction_has_its_own_cut():
    settling = curve(0.9, 0.8, 0.45, 0.45, 0.45, 0.45)
    assert departure(settling, threshold=0.5) == 3
    assert departure(settling, threshold=0.35) is None

    rising = curve(0.1, 0.2, 0.55, 0.55, 0.55, 0.55)
    assert arrival(rising, threshold=0.5) == 3
    assert arrival(rising, threshold=0.65) is None


def test_a_forecast_inside_the_band_answers_neither():
    """The point of allowing the two cuts to differ.

    With a band of 0.4-0.6 a curve pinned at 0.5 is neither leaving nor
    arriving, whichever side the household is currently on.
    """
    pinned = curve(*[0.5] * 8)
    assert departure(pinned, threshold=0.4, at=HOME) is None
    assert arrival(pinned, threshold=0.6, at=AWAY) is None


# --- the gate on the observed present ------------------------------------

def test_nothing_is_reported_in_the_direction_they_are_already_in():
    away_curve = curve(*[0.1] * 8)
    assert departure(away_curve, at=AWAY) is None
    home_curve = curve(*[0.9] * 8)
    assert arrival(home_curve, at=HOME) is None


def test_the_present_is_judged_by_half_the_window_whatever_the_cut_is():
    """`state_now` is a time fraction, not a probability.

    Three of the last five minutes at home is 0.6, and that is somebody who is
    in the house -- so a `departure_threshold` of 0.7 must not reclassify them
    as already away. The gate stays at half the window in both directions.
    """
    leaving = curve(0.9, 0.9, 0.2, 0.1, 0.1, 0.1)
    assert departure(leaving, threshold=0.7, at=row(0.6)) == 3
    assert departure(leaving, threshold=0.3, at=row(0.4)) is None


# --- the three boundary decisions ----------------------------------------

def test_a_run_at_the_end_of_the_curve_is_whatever_is_left():
    """Requiring the run to fit would shorten the published horizon.

    A genuine departure at +47 h reported as "unknown" is a worse answer than
    reporting it, so the tail keeps whatever hours remain.
    """
    tail = {h: 0.9 for h in range(1, 47)} | {47: 0.2, 48: 0.1}
    assert departure(tail, min_hours=3) == 47

    last_only = {h: 0.9 for h in range(1, 48)} | {48: 0.1}
    assert departure(last_only, min_hours=2) == 48


def test_a_missing_horizon_breaks_the_run():
    """A NaN horizon is absent from the curve, not zero.

    Stepping over the gap would assert agreement across an hour nobody
    forecast, so the run restarts on the far side of it.
    """
    holed = {1: 0.9, 2: 0.9, 3: 0.2, 5: 0.2, 6: 0.1}
    assert departure(holed, min_hours=2) == 5


def test_a_gap_does_not_raise():
    """A run that never completes falls off the end rather than throwing.

    `curve[h]` on a horizon that is not there would be a KeyError, which is why
    the run is built from `h in curve` rather than indexed blindly.
    """
    assert departure({1: 0.9, 5: 0.2, 6: 0.9, 7: 0.9}, min_hours=2) is None


def test_the_tail_rule_is_measured_from_the_grid_not_from_the_curve():
    """The exemption is for the end of the GRID, not the end of the curve.

    It used to measure from `max(curve)`, which was defensible when a short
    curve meant a test had handed one over: every horizon was published, so the
    curve's end and the grid's end were the same thing.

    Since a horizon is published only where a model earned it, a curve that
    stops at +5 h is the normal shape of a household whose far horizons do not
    ship -- and measuring from it would report "leaving at +5 h" when what
    happened is that the forecast ran out. Worse, the sensor would then move
    whenever a `ships` flag flipped at a retrain, for reasons that have nothing
    to do with the household.
    """
    assert departure({1: 0.9, 5: 0.2}, min_hours=2) is None, \
        "the forecast running out is not a departure"
    # The real end of the grid still exempts itself, which is what the rule was
    # for: a genuine +48 h departure is better reported than turned into
    # "unknown" for want of a +49th hour that cannot exist.
    end = max(config.HORIZONS_H)
    assert departure({1: 0.9, end: 0.2}, min_hours=2) == end


def test_an_empty_curve_has_no_crossing():
    """A fresh install forecasts nothing at all, and `max({})` raises.

    Unreachable before: `predict_rows` dropped a subject whose curve was empty,
    so this function was never called with one. It now publishes the record
    anyway -- the entities exist and read `unknown` -- which brings the empty
    dict here for the first time.
    """
    assert departure({}) is None
    assert arrival({}) is None


# --- what the settings do to it ------------------------------------------

def test_the_configured_cuts_are_what_serving_uses():
    """`predict_rows` reads the module globals `configure()` populated."""
    assert config.DEPARTURE_THRESHOLD == config.DEFAULT_DEPARTURE_THRESHOLD
    assert config.ARRIVAL_THRESHOLD == config.DEFAULT_ARRIVAL_THRESHOLD
    assert config.CROSSING_MIN_HOURS == config.DEFAULT_CROSSING_MIN_HOURS


@pytest.mark.parametrize("value, expected", [
    # A number keeps its evident intent and moves to the nearest servable
    # value: somebody who typed 0 wanted the lowest cut there is, and 0.01
    # says that far better than the 0.5 default would.
    (0.0, 0.01),
    (-3, 0.01),
    (1.0, 0.99),
    (0.65, 0.65),
    # Not a number at all, so there is no intent to preserve. Fall back.
    ("nonsense", config.DEFAULT_DEPARTURE_THRESHOLD),
    (None, config.DEFAULT_DEPARTURE_THRESHOLD),
    # JSON `true` arrives as a Python int and would otherwise clamp to 1.0.
    (True, config.DEFAULT_DEPARTURE_THRESHOLD),
])
def test_a_hand_edited_config_is_clamped_rather_than_fatal(value, expected):
    """`/data/config.json` can be edited on the box, bypassing the API.

    A publisher whose job is to keep publishing degrades a nonsense cut to
    something servable and says so in the log. It does not refuse to boot.
    """
    from occupancy_forecast.tests.conftest import settings as make_settings
    config.configure(make_settings(departure_threshold=value))
    assert config.DEPARTURE_THRESHOLD == pytest.approx(expected)


def test_a_hand_edited_run_length_is_clamped_to_the_curve():
    from occupancy_forecast.tests.conftest import settings as make_settings
    config.configure(make_settings(crossing_min_hours=999))
    assert config.CROSSING_MIN_HOURS == max(config.HORIZONS_H)
    config.configure(make_settings(crossing_min_hours=0))
    assert config.CROSSING_MIN_HOURS == 1
