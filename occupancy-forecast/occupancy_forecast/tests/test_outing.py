"""Is this person going out today.

The tests that earn their keep here are the causality ones. The per-weekday rate
is both a feature and the baseline the model must beat, so a leak inflates the
model and deflates nothing -- it would read as skill, which is exactly what it
must never be allowed to do.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast import config, departure, outing
from occupancy_forecast.tests.conftest import settings as make_settings


def _table(out_days: set[str], subject: str = "alice",
           zone: str = "zone_alice_office", start: str = "2026-01-01",
           days: int = 40) -> pd.DataFrame:
    """A grid where the person is out 08:00-17:00, in `zone` on the named days."""
    slots = config.SLOTS_PER_DAY
    per_hour = 60 // config.GRID_MINUTES
    rows = []
    begin = dt.date.fromisoformat(start)
    for index in range(days):
        day = begin + dt.timedelta(days=index)
        out = day.isoformat() in out_days
        home = np.ones(slots)
        if out:
            home[8 * per_hour:17 * per_hour] = 0.0
        inzone = np.zeros(slots)
        if out:
            inzone[9 * per_hour:16 * per_hour] = 1.0
        midnight = pd.Timestamp(day, tz=config.TIMEZONE)
        for slot in range(slots):
            rows.append({
                "subject": subject,
                "time": (midnight + pd.Timedelta(
                    minutes=config.GRID_MINUTES * slot)).tz_convert("UTC"),
                "home_frac": home[slot],
                zone: inzone[slot],
            })
    return pd.DataFrame(rows)


def _labelled(table: pd.DataFrame) -> pd.DataFrame:
    return outing.label_out_days(table, departure.label_days(table))


def test_a_day_in_a_configured_zone_is_an_out_day():
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    went = {"2026-01-05", "2026-01-07"}
    days = _labelled(_table(went))

    by_date = {d.date().isoformat(): o
               for d, o in zip(days["date"], days["out"])}
    assert by_date["2026-01-05"] == 1.0
    assert by_date["2026-01-07"] == 1.0
    assert by_date["2026-01-06"] == 0.0


def test_zone_other_does_not_count_as_somewhere_that_matters():
    """`zone_other` means "in some zone nobody named", which is the opposite of
    a household declaring that a place matters."""
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    assert "zone_other" not in outing.out_columns()
    assert "zone_alice_office" in outing.out_columns()


def test_an_installation_with_no_zones_answers_rather_than_raising():
    """A household that ticked no zones has no question to ask. The
    column goes NaN and `prequential` skips it -- it is not an error."""
    config.configure(make_settings(zones=[], zone_names={}))
    table = _table(set(), zone="zone_other")
    days = outing.label_out_days(table, departure.label_days(table))
    assert days["out"].isna().all()
    assert outing.prequential(outing.feature_frame(days)).empty


def test_the_weekday_rate_never_reads_the_day_it_predicts():
    """The leak that would matter most, because `wday_rate` is the baseline as
    well as a feature. Every Monday a day out: the FIRST Monday must still
    have no rate, and the rate on Monday k must be built from k-1 Mondays."""
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    mondays = {(dt.date(2026, 1, 5) + dt.timedelta(days=7 * k)).isoformat()
               for k in range(6)}
    frame = outing.feature_frame(_labelled(_table(mondays, days=45)))

    monday = frame[frame["dow"] == 0].sort_values("date")
    assert monday["out"].sum() >= 5
    # Below MIN_WEEKDAY_SAMPLES the rate is withheld rather than guessed.
    assert monday["wday_rate"].head(outing.MIN_WEEKDAY_SAMPLES).isna().all()
    assert monday["wday_n"].to_numpy()[0] == 0


def test_days_since_out_counts_backwards_only():
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    went = {"2026-01-05", "2026-01-09"}
    frame = outing.feature_frame(_labelled(_table(went)))
    by_date = {d.date().isoformat(): g
               for d, g in zip(frame["date"], frame["days_since_out"])}
    # Nothing has happened yet on the first office day itself.
    assert np.isnan(by_date["2026-01-05"]) or by_date["2026-01-05"] > 0
    # Four days later, having been in on the 5th.
    assert by_date["2026-01-09"] == pytest.approx(4.0)


def test_a_person_never_sees_their_own_out_column_as_a_partner():
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    frame = outing.feature_frame(_labelled(_table({"2026-01-05"})))
    assert frame["other_alice_out"].isna().all()


def test_the_baseline_rate_is_shrunk_toward_the_person_s_own_rate():
    """An unshrunk weekday rate emits 0.00 and 1.00 off three observations, and
    a baseline that makes confident mistakes is an easy thing for a model to
    beat for reasons that are not skill."""
    train = pd.DataFrame({
        "out": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "wday_rate": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    })
    weight, base = outing._fit_rate_shrink(train)
    assert 0.0 < weight <= 1.0
    assert base == pytest.approx(0.5)

    empty = pd.DataFrame({"out": [1.0, 0.0], "wday_rate": [np.nan, np.nan]})
    weight, base = outing._fit_rate_shrink(empty)
    assert weight == 1.0 and base == pytest.approx(0.5)


def test_the_gate_needs_the_effect_size_and_not_just_a_win():
    """A model a hair better than the baseline does not ship. Measured on this
    household, permuted labels produce skills up to +19%, so a bare win is
    inside the noise."""
    days = pd.DataFrame({"subject": ["alice"] * 4,
                         "candidate": [True] * 4,
                         "out": [1.0, 0.0, 1.0, 0.0]})
    scored = pd.DataFrame({
        "subject": ["alice"] * 4,
        "date": pd.date_range("2026-01-01", periods=4, freq="14D"),
        "out": [True, False, True, False],
        "p_model": [0.62, 0.38, 0.62, 0.38],
        "p_baseline": [0.60, 0.40, 0.60, 0.40],
        "shrink_weight": [0.9] * 4, "shrink_base": [0.5] * 4,
    })
    metrics = outing.score_subject("alice", days, scored)
    assert metrics.brier < metrics.baseline_brier, "it does win"
    assert metrics.skill_pct < outing.MIN_SKILL_PCT
    assert metrics.ships is False, "winning is not enough; the effect must be real"


# --- the routine that is actually served -----------------------------------

def _routine_days(weeks: int = 12) -> pd.DataFrame:
    """alice out every Tuesday and Friday, out 08:00, back 17:00."""
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    begin = dt.date(2026, 1, 5)                      # a Monday
    went = {(begin + dt.timedelta(days=7 * w + d)).isoformat()
            for w in range(weeks) for d in (1, 4)}
    return _labelled(_table(went, days=weeks * 7))


def test_the_routine_records_the_hours_and_what_they_are_built_from():
    routine = outing.fit_routine(_routine_days())
    alice = routine["alice"]
    assert alice["n_out"] >= 20

    tuesday = alice["by_weekday"]["1"]
    assert tuesday["rate"] == pytest.approx(1.0)
    assert tuesday["departure_hour"] == pytest.approx(8.0)
    # In the zone until 15:30 and home again at 17:00. The return is when they
    # got HOME, so 17.0 -- not the 16.0 they left the zone.
    assert tuesday["return_hour"] == pytest.approx(17.0)
    assert tuesday["n_out"] >= 10, "the count travels with the median"

    monday = alice["by_weekday"]["0"]
    assert monday["rate"] == pytest.approx(0.0)
    assert monday["departure_hour"] is None, "no office Mondays, so no hour"


def test_a_house_gets_no_office_routine():
    """A house does not go to an office, and a group's presence is not a
    person's commute."""
    days = _routine_days()
    days = pd.concat([days, days.assign(subject=config.HOUSE_SLUG)],
                     ignore_index=True)
    assert config.HOUSE_SLUG not in outing.fit_routine(days)


def test_too_little_history_publishes_no_routine_at_all():
    """Below the floor the per-weekday medians are single observations wearing
    a median's clothes."""
    assert outing.fit_routine(_routine_days(weeks=3)) == {}


def test_today_falls_back_to_the_overall_hours_on_a_thin_weekday():
    """Thin means SOME office days but too few to take a median from -- not
    none, which is a different answer entirely (see the `never` test below).

    Two Wednesdays in twelve weeks: enough to know she sometimes goes in, not
    enough for that weekday's own median to mean anything.
    """
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    begin = dt.date(2026, 1, 5)
    went = {(begin + dt.timedelta(days=7 * w + d)).isoformat()
            for w in range(12) for d in (1, 4)}
    went |= {(begin + dt.timedelta(days=7 * w + 2)).isoformat() for w in (0, 5)}
    routine = outing.fit_routine(_labelled(_table(went, days=84)))

    wednesday = pd.Timestamp("2026-04-01T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    answer = outing.today(routine, "alice", wednesday)
    assert answer["n_out_weekday"] == 2
    assert answer["departure_from"] == "overall"
    assert answer["departure_hour"] == pytest.approx(8.0)

    tuesday = pd.Timestamp("2026-03-31T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    answer = outing.today(routine, "alice", tuesday)
    assert answer["departure_from"] == "weekday"
    assert answer["probability"] > 0.7


def test_today_is_none_for_someone_with_no_routine():
    assert outing.today({}, "nobody") is None
    assert outing.today(outing.fit_routine(_routine_days()), "bob") is None


def test_an_hour_becomes_a_moment_on_the_local_date():
    """The sensors carry `device_class: timestamp`, so a fractional hour has to
    land on a real local moment -- including on the far side of a DST change,
    where an hour offset from UTC would be an hour wrong."""
    at = pd.Timestamp("2026-07-01T09:00", tz="UTC")
    assert outing.at_hour(at, None) is None
    stamp = pd.Timestamp(outing.at_hour(at, 8.5))
    local = stamp.tz_convert(config.tzinfo())
    assert (local.hour, local.minute) == (8, 30)
    assert local.date() == at.tz_convert(config.tzinfo()).date()


@pytest.mark.parametrize("day", ["2026-03-29", "2026-10-25"])
@pytest.mark.parametrize("hour", [8.0, 17.5])
def test_an_hour_is_still_that_hour_on_a_dst_transition_day(day, hour):
    """The two days a year the old arithmetic got wrong. `midnight + Timedelta`
    on a tz-aware stamp is absolute time, so 8.0 rendered as 09:00 on the
    spring-forward day and 07:00 on the fall-back day (Europe/Amsterdam, the
    fixture's zone). The sensors carry `device_class: timestamp`; an automation
    reading them would have fired an hour out."""
    at = pd.Timestamp(f"{day}T12:00", tz=config.TIMEZONE).tz_convert("UTC")
    local = pd.Timestamp(outing.at_hour(at, hour)).tz_convert(config.tzinfo())
    assert (local.hour, local.minute) == (int(hour), int(round((hour % 1) * 60)))
    assert str(local.date()) == day


def test_a_routine_survives_a_round_trip_and_a_corrupt_file(tmp_path):
    routine = outing.fit_routine(_routine_days())
    outing.save_routine(routine, tmp_path)
    assert outing.load_routine(tmp_path)["alice"]["n_out"] == \
        routine["alice"]["n_out"]

    (tmp_path / outing.ROUTINE_NAME).write_text("{ not json")
    assert outing.load_routine(tmp_path) == {}, \
        "a corrupt artifact publishes nothing rather than raising in a serve cycle"
    assert outing.load_routine(tmp_path / "nowhere") == {}


def test_a_weekday_they_never_go_in_on_publishes_no_hour_at_all():
    """The difference between "we cannot say" and "we know they do not".

    A real household had seventeen observed Thursdays and zero office days
    among them.
    Falling back to the overall median published `08:00` on those Thursdays --
    a number nobody earned, and one an automation reading the timestamp without
    the probability beside it would act on.
    """
    routine = outing.fit_routine(_routine_days())
    # A Monday: twelve of them observed, alice never in the outing.
    monday = pd.Timestamp("2026-03-30T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    answer = outing.today(routine, "alice", monday)

    assert answer["departure_from"] == "never"
    assert answer["departure_hour"] is None
    assert answer["return_hour"] is None
    assert answer["n_out_weekday"] == 0
    assert answer["probability"] < 0.2

    # And a weekday she DOES go in on still answers.
    tuesday = pd.Timestamp("2026-03-31T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    assert outing.today(routine, "alice", tuesday)["departure_hour"] is not None


# --- the return hour is when they got HOME ---------------------------------

def _table_with_return(zone_out_slot: int, home_back_slot: int | None,
                       day: str = "2026-01-06") -> pd.DataFrame:
    """One day: in a zone until `zone_out_slot`, home again at `home_back_slot`.

    `home_back_slot` of None means they were not home again before midnight.
    """
    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    slots = config.SLOTS_PER_DAY
    rows = []
    begin = dt.date.fromisoformat("2026-01-01")
    for index in range(12):
        d = begin + dt.timedelta(days=index)
        home = np.ones(slots)
        inzone = np.zeros(slots)
        if d.isoformat() == day:
            home[16:] = 0.0                       # out from 08:00
            inzone[18:zone_out_slot + 1] = 1.0
            if home_back_slot is not None:
                home[home_back_slot:] = 1.0
        midnight = pd.Timestamp(d, tz=config.TIMEZONE)
        for slot in range(slots):
            rows.append({
                "subject": "alice",
                "time": (midnight + pd.Timedelta(
                    minutes=config.GRID_MINUTES * slot)).tz_convert("UTC"),
                "home_frac": home[slot],
                "zone_alice_office": inzone[slot],
            })
    return pd.DataFrame(rows)


def test_the_return_hour_is_when_they_got_home_not_when_they_left_the_zone():
    """It used to be the last slot in the zone plus one, with a comment claiming
    that was "~20 minutes before they are home". That was a guess. Somebody who
    stops on the way home is home when they are home."""
    # In the zone until slot 33 (16:30), home again at slot 37 (18:30).
    days = _labelled(_table_with_return(zone_out_slot=33, home_back_slot=37))
    row = days[days["date"] == pd.Timestamp("2026-01-06")].iloc[0]
    assert row["out"] == 1.0
    assert row["out_return_hour"] == pytest.approx(18.5), \
        "the hour they reached home, not the 17.0 they left the zone"


def test_a_return_after_midnight_yields_no_hour_rather_than_wrapping():
    """A wrap put a nonsense 00:30 return on one Sunday in the first artifact.
    No hour is the honest answer: they did not get home that day."""
    days = _labelled(_table_with_return(zone_out_slot=33, home_back_slot=None))
    row = days[days["date"] == pd.Timestamp("2026-01-06")].iloc[0]
    assert row["out"] == 1.0, "they still went out"
    assert pd.isna(row["out_return_hour"])


# --- one answer: the model says whether, the routine says when -------------

def _routine_for(weekday_hours: dict[int, tuple[float, float]]) -> dict:
    """A routine artifact by hand: {weekday: (departure, return)}."""
    by_weekday = {}
    for dow in range(7):
        hours = weekday_hours.get(dow)
        by_weekday[str(dow)] = {
            "n": 10, "n_out": 8 if hours else 0,
            "rate": 0.8 if hours else 0.0,
            "departure_hour": hours[0] if hours else None, "departure_sd": 0.3,
            "departure_n": 8 if hours else 0,
            "return_hour": hours[1] if hours else None, "return_sd": 0.4,
        }
    return {"alice": {"subject": "alice", "fitted_at": None, "n_days": 70,
                      "n_out": 24, "base_rate": 0.3,
                      "shrink_weight": 1.0, "shrink_base": 0.3,
                      "by_weekday": by_weekday,
                      "overall": {"departure_hour": 8.0, "departure_sd": 0.5,
                                  "return_hour": 18.0, "return_sd": 0.5}}}


def test_the_time_comes_from_the_day_the_change_falls_on_not_today():
    """The wrinkle that makes this worth writing down. A departure sixteen hours
    out lands TOMORROW, and one person here never goes out on a Thursday while
    going out on 88% of Fridays -- reading today's routine would answer the
    wrong question."""
    from occupancy_forecast import predict

    # Thursday 2026-09-03 at 17:00 local. Nothing on Thursdays; 08:00 Fridays.
    thursday = pd.Timestamp("2026-09-03T17:00", tz=config.TIMEZONE).tz_convert("UTC")
    routine = _routine_for({4: (8.0, 18.0)})          # Friday only

    change = predict._next_change(routine, "alice", thursday,
                                  departure_h=16, arrival_h=None)
    assert change["direction"] == "leaving"
    assert change["at_from"] == "routine"
    at = pd.Timestamp(change["at"]).tz_convert(config.tzinfo())
    assert at.dayofweek == 4, "Friday's routine, not Thursday's"
    assert (at.hour, at.minute) == (8, 0)


def test_it_falls_back_to_the_crossing_where_that_day_has_no_hour():
    """A weekday they never go out on has no hour to give. The model still says
    a change is coming, so the row must still say something -- the crossing's
    own rounded hour, marked as such."""
    from occupancy_forecast import predict

    thursday = pd.Timestamp("2026-09-03T17:00", tz=config.TIMEZONE).tz_convert("UTC")
    routine = _routine_for({4: (8.0, 18.0)})

    # Two hours out lands on the same Thursday, which has nothing.
    change = predict._next_change(routine, "alice", thursday,
                                  departure_h=2, arrival_h=None)
    assert change["at_from"] == "crossing"
    at = pd.Timestamp(change["at"]).tz_convert(config.tzinfo())
    assert (at.hour, at.minute) == (19, 0), "the crossing's own moment"


def test_an_arrival_is_timed_by_the_return_and_a_departure_by_the_departure():
    from occupancy_forecast import predict

    monday = pd.Timestamp("2026-09-07T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    routine = _routine_for({0: (7.0, 18.5)})

    leaving = predict._next_change(routine, "alice", monday, 1, None)
    assert pd.Timestamp(leaving["at"]).tz_convert(config.tzinfo()).hour == 7

    arriving = predict._next_change(routine, "alice", monday, None, 12)
    back = pd.Timestamp(arriving["at"]).tz_convert(config.tzinfo())
    assert (back.hour, back.minute) == (18, 30), "the return, not the departure"


def test_no_crossing_means_no_answer_at_all():
    """The model decides whether. With no crossing there is nothing to time, and
    the routine does not get to volunteer one."""
    from occupancy_forecast import predict

    monday = pd.Timestamp("2026-09-07T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    change = predict._next_change(_routine_for({0: (7.0, 18.5)}), "alice",
                                  monday, None, None)
    assert change == {"direction": None, "in_hours": None,
                      "at": None, "at_from": None}
