"""One answer: the model says whether and roughly when, the routine sharpens it.

The tests that earn their keep are the REFUSALS. A routine hour is a median off
other days, and published unguarded it once named a morning the forecast beside
it read as near-certainly home.
"""

import pandas as pd

from occupancy_forecast import config, predict


def _routine_for(weekday_hours: dict[int, tuple[float, float]],
                 departure_n: int = 8) -> dict:
    """A departure-routine artifact by hand: {weekday: (departure, return)}.

    `departure_n` is the lever the refusal tests pull: below
    `departure.MIN_WEEKDAY_SAMPLES` the weekday has no hour of its own.
    """
    by_weekday = {}
    for dow in range(7):
        hours = weekday_hours.get(dow)
        by_weekday[str(dow)] = {
            "n": 10, "n_left": 8 if hours else 0,
            "rate": 0.8 if hours else 0.0,
            "departure_hour": hours[0] if hours else None, "departure_sd": 0.3,
            "departure_n": departure_n if hours else 0,
            "return_hour": hours[1] if hours else None, "return_sd": 0.4,
            "return_n": departure_n if hours else 0,
        }
    return {"alice": {"subject": "alice", "fitted_at": None, "n_days": 70,
                      "n_left": 24, "base_rate": 0.3,
                      "shrink_weight": 1.0, "shrink_base": 0.3,
                      "by_weekday": by_weekday,
                      "overall": {"departure_hour": 8.0, "departure_sd": 0.5,
                                  "return_hour": 18.0, "return_sd": 0.5}}}


def _local(change: dict, key: str = "at") -> pd.Timestamp:
    return pd.Timestamp(change[key]).tz_convert(config.tzinfo())


def test_the_time_comes_from_the_day_the_change_falls_on_not_today():
    """A departure sixteen hours out lands TOMORROW, so reading today's routine
    would answer the wrong question."""
    # Thursday 2026-09-03 at 17:00 local. Nothing on Thursdays; 08:00 Fridays.
    thursday = pd.Timestamp("2026-09-03T17:00", tz=config.TIMEZONE).tz_convert("UTC")
    routine = _routine_for({4: (8.0, 18.0)})          # Friday only

    change = predict._next_change(routine, "alice", thursday,
                                  departure_h=16, arrival_h=None)
    assert change["direction"] == "leaving"
    assert change["at_from"] == "routine"
    at = _local(change)
    assert at.dayofweek == 4, "Friday's routine, not Thursday's"
    assert (at.hour, at.minute) == (8, 0)


def test_it_falls_back_to_the_crossing_where_that_day_has_no_hour():
    """A weekday they never leave on has no hour to give, but the model still
    says a change is coming -- so the crossing's own rounded hour, marked so."""
    thursday = pd.Timestamp("2026-09-03T17:00", tz=config.TIMEZONE).tz_convert("UTC")
    routine = _routine_for({4: (8.0, 18.0)})

    # Two hours out lands on the same Thursday, which has nothing.
    change = predict._next_change(routine, "alice", thursday,
                                  departure_h=2, arrival_h=None)
    assert change["at_from"] == "crossing"
    assert (_local(change).hour, _local(change).minute) == (19, 0)
    assert change["routine_at"] is None, "nothing was offered, so nothing is kept"


def test_a_routine_hour_far_from_the_crossing_is_refused():
    """The incident. The forecast dipped in the EVENING; the routine offered a
    MORNING hour for the same day, and the card named an hour its own chart
    read as home.
    """
    friday = pd.Timestamp("2026-03-06T15:30", tz=config.TIMEZONE).tz_convert("UTC")
    routine = _routine_for({5: (8.0, 18.0)})          # the next day, 08:00

    change = predict._next_change(routine, "alice", friday,
                                  departure_h=26, arrival_h=None)
    assert change["at_from"] == "crossing"
    at = _local(change)
    assert (at.dayofweek, at.hour, at.minute) == (5, 17, 30)
    # Refused, not discarded: the panel still says what the routine expected.
    offered = _local(change, "routine_at")
    assert (offered.dayofweek, offered.hour) == (5, 8)


def test_a_weekday_with_too_few_departures_never_overrules_the_crossing():
    """`departure_from` is "overall" here -- a median off the OTHER weekdays.
    Close enough to the crossing to pass the distance test, and still refused.
    """
    thursday = pd.Timestamp("2026-09-03T17:00", tz=config.TIMEZONE).tz_convert("UTC")
    # One measured Friday departure, so Friday falls back to the overall 08:00.
    routine = _routine_for({4: (9.0, 18.0)}, departure_n=1)

    change = predict._next_change(routine, "alice", thursday,
                                  departure_h=16, arrival_h=None)
    assert change["routine_day"]["departure_from"] == "overall"
    assert change["at_from"] == "crossing"
    assert (_local(change).hour, _local(change).minute) == (9, 0)


def test_an_arrival_is_timed_by_the_return_and_a_departure_by_the_departure():
    monday = pd.Timestamp("2026-09-07T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    routine = _routine_for({0: (7.0, 18.5)})

    leaving = predict._next_change(routine, "alice", monday, 1, None)
    assert _local(leaving).hour == 7

    arriving = predict._next_change(routine, "alice", monday, None, 12)
    back = _local(arriving)
    assert (back.hour, back.minute) == (18, 30), "the return, not the departure"


def test_the_evidence_behind_the_hour_travels_with_it():
    """A median off eight Mondays and one off thirty read identically on the
    sensor; `routine_day` is where the panel can tell them apart."""
    monday = pd.Timestamp("2026-09-07T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    change = predict._next_change(_routine_for({0: (7.0, 18.5)}), "alice",
                                  monday, 1, None)
    day = change["routine_day"]
    assert (day["weekday"], day["n_weekday"], day["n_left_weekday"]) == (0, 10, 8)
    assert day["departure_from"] == "weekday"


def test_no_crossing_means_no_answer_at_all():
    """The model decides whether. With no crossing there is nothing to time, and
    the routine does not get to volunteer one."""
    monday = pd.Timestamp("2026-09-07T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    change = predict._next_change(_routine_for({0: (7.0, 18.5)}), "alice",
                                  monday, None, None)
    assert change == {"direction": None, "in_hours": None, "at": None,
                      "at_from": None, "routine_at": None, "routine_day": None}


def test_a_subject_with_no_routine_still_gets_the_crossing():
    """Every install starts here, and a house had no routine at all until this
    change -- which is why the house row was the only one reading right."""
    monday = pd.Timestamp("2026-09-07T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    change = predict._next_change({}, "nobody", monday, 3, None)
    assert change["at_from"] == "crossing"
    assert change["routine_day"] is None
    assert _local(change).hour == 9
