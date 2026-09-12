"""The departure labels, the risky part of the family: a rule applied to a day,
where every way of getting it wrong is quiet -- a day nobody watched scored as
"stayed home", an evening errand scored as the commute.
"""
import datetime as dt

import pandas as pd
import pytest

from occupancy_forecast import config, departure
from occupancy_forecast.tests.conftest import settings as make_settings

SUBJECT = "alice"
DATE = "2026-03-05"          # a Thursday, well clear of any DST transition


@pytest.fixture(autouse=True)
def _configured():
    config.configure(make_settings())


def _table(spec: dict | None = None, fill: float = 1.0, date: str = DATE,
           subject: str = SUBJECT) -> pd.DataFrame:
    """One local day of `home_frac`, 48 slots, overridden by 'HH:MM'; `fill` is
    the rest of the day: 1.0 home, 0.0 away, None unobserved."""
    spec = spec or {}
    tz = config.tzinfo()
    midnight = dt.datetime.combine(dt.date.fromisoformat(date), dt.time(0, 0),
                                   tzinfo=tz)
    rows = []
    for slot in range(config.SLOTS_PER_DAY):
        at = midnight + dt.timedelta(minutes=config.GRID_MINUTES * slot)
        rows.append({"subject": subject,
                     "time": at.astimezone(dt.timezone.utc),
                     "home_frac": spec.get(at.strftime("%H:%M"), fill)})
    return pd.DataFrame(rows)


def test_the_repeated_hour_on_the_autumn_transition_keeps_both_observations():
    """On the fall-back day 02:00-02:59 happens twice, so two UTC slots land on
    one local slot; the mean of the two is the fair value, not the first."""
    times = pd.to_datetime(["2026-10-25T00:00Z", "2026-10-25T00:30Z",    # 02:00, 02:30 CEST
                            "2026-10-25T01:00Z", "2026-10-25T01:30Z"],   # 02:00, 02:30 CET
                           utc=True)
    frame = pd.DataFrame({"subject": SUBJECT, "time": times,
                          "home_frac": [1.0, 1.0, 0.0, 0.0]})
    grid = departure.day_grid(frame)
    day = grid.loc[(SUBJECT, dt.date(2026, 10, 25))]
    assert day[4] == 0.5 and day[5] == 0.5
    assert day.notna().sum() == 2, "only the two wall-clock slots that were observed"


def _one(table: pd.DataFrame) -> pd.Series:
    labelled = departure.label_days(table)
    assert len(labelled) == 1, labelled
    return labelled.iloc[0]


def _away(*times: str) -> dict:
    return {t: 0.0 for t in times}


def test_a_departure_needs_a_sustained_absence():
    """One slot away is a GPS blip, not a departure: an absence under about an
    hour has no representation in a 30-minute-slot target at all, the same line
    `config.CROSSING_MIN_HOURS` draws for the published sensor."""
    blip = _one(_table(_away("07:30")))
    assert blip["candidate"]
    assert not blip["left_today"]

    real = _one(_table(_away("07:30", "08:00")))
    assert real["left_today"]
    assert real["departure_hour"] == 7.5, "the FIRST slot of the run, not the last"


def test_a_hole_breaks_the_away_run():
    """Away, unobserved, away is not evidence of a sustained absence -- the same
    decision `_crossing` makes about a gap in the curve."""
    row = _one(_table({"07:30": 0.0, "08:00": None, "08:30": 0.0}))
    assert not row["left_today"], "a hole was treated as continued absence"


def test_a_morning_gap_makes_the_day_uncountable_rather_than_a_stay_at_home():
    """THE OUTAGE BUG, pinned. A day with a hole where the departure would be is
    not a day she stayed home -- it is a day nobody watched, and it must be
    dropped rather than labelled."""
    gap = {f"{h:02d}:{m:02d}": None
           for h in range(6, 10) for m in (0, 30)}      # 06:00-09:30 unobserved
    row = _one(_table(gap))
    assert not row["candidate"], "a four-hour hole over the morning was accepted"
    assert not row["left_today"]
    # The distinction this test exists for: dropped, not scored as staying home.
    assert departure.label_days(_table(gap))["candidate"].sum() == 0


def test_a_day_nobody_watched_is_not_a_day_she_stayed_home():
    """The same rule from the other direction: too little of the day observed."""
    sparse = {f"{h:02d}:{m:02d}": None
              for h in range(0, 12) for m in (0, 30)}
    assert not _one(_table(sparse))["candidate"]


def test_a_person_already_away_is_not_asked_the_question():
    """The conditional, enforced in ONE place. She left before the question was
    put, so the day was never a candidate and no model ever trained on it."""
    row = _one(_table(fill=0.0))
    assert not row["candidate"]
    assert not row["left_today"]


def test_a_departure_after_the_cap_is_not_leaving_today():
    """The MAX_LEAD_MIN analogue. Unbounded, an evening walk to the shop would
    teach a weekday departure time that is really a fact about shop hours."""
    late = _one(_table(_away("22:30", "23:00")))
    assert late["candidate"]
    assert not late["left_today"], "an absence after the cap counted as leaving"

    inside = _one(_table(_away("21:30", "22:00")))
    assert inside["left_today"]


def test_the_earliest_sustained_absence_is_the_departure():
    """Two absences in a day: the morning one is the departure."""
    row = _one(_table(_away("07:30", "08:00", "17:00", "17:30")))
    assert row["departure_hour"] == 7.5


def test_the_latest_return_is_the_return():
    """The mirror of the rule above, and deliberately the OTHER end: the walk at
    07:30 and the dinner at 17:00 bracket the day, and "back for the evening" is
    the one an automation waits on."""
    row = _one(_table(_away("07:30", "08:00", "17:00", "17:30")))
    assert row["departure_hour"] == 7.5
    assert row["return_hour"] == 18.0, "the first slot home after the LAST absence"


def test_a_return_after_midnight_yields_no_hour_rather_than_wrapping():
    """Out from 21:30 and not back before the day ends. A wrapped 00:30 would
    read as coming home before leaving."""
    row = _one(_table(_away("21:30", "22:00", "22:30", "23:00", "23:30")))
    assert row["left_today"] and row["departure_hour"] == 21.5
    assert pd.isna(row["return_hour"])


def test_a_day_she_never_left_has_no_return_either():
    row = _one(_table())
    assert not row["left_today"]
    assert pd.isna(row["return_hour"])


def test_the_labels_do_not_depend_on_row_order():
    """A guard against an unsorted groupby quietly deciding the answer."""
    table = _table(_away("07:30", "08:00"))
    shuffled = table.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert _one(shuffled)["departure_hour"] == _one(table)["departure_hour"]


def _history(hours: list[float | None], subject: str = SUBJECT,
             start: str = "2026-03-02") -> pd.DataFrame:
    """Consecutive days; `hours[i]` is that day's departure, None for staying in."""
    begin = dt.date.fromisoformat(start)
    frames = []
    for offset, hour in enumerate(hours):
        date = (begin + dt.timedelta(days=7 * offset)).isoformat()   # same weekday
        spec = {}
        if hour is not None:
            slot = int(hour * 2)
            for k in range(departure.MIN_AWAY_SLOTS):
                at = dt.time((slot + k) // 2, ((slot + k) % 2) * 30)
                spec[at.strftime("%H:%M")] = 0.0
        frames.append(_table(spec, date=date, subject=subject))
    return pd.concat(frames, ignore_index=True)


def test_the_weekday_lookup_never_sees_the_day_it_is_predicting():
    """Seven Thursdays at 06:00 and an eighth at 18:00: the eighth day's feature
    must read 6.0, the median of the seven BEFORE it. Anything pulled toward
    18:00 has seen its own answer -- the leak behind a beautiful offline number."""
    days = departure.feature_frame(
        departure.label_days(_history([6.0] * 7 + [18.0])))
    last = days.iloc[-1]
    assert last["departure_hour"] == 18.0, "the label itself"
    assert last["wday_hour"] == 6.0, "the feature saw its own day"
    # And it is NaN until there is enough history to mean anything.
    assert days["wday_hour"].isna().iloc[:departure.MIN_WEEKDAY_SAMPLES].all()


def test_truncating_the_future_does_not_change_the_past():
    """The strong form: a row's features must be the same when every later day
    is deleted, which is the only state a live install is ever in. Catches an
    expanding window that forgot to shift."""
    full = departure.feature_frame(
        departure.label_days(_history([6.0, 6.5, 6.0, 7.0, 18.0, 6.0, 6.5])))
    cut = 5
    truncated = departure.feature_frame(
        departure.label_days(_history([6.0, 6.5, 6.0, 7.0, 18.0][:cut])))

    columns = [c for c in departure.feature_columns() if c in full.columns]
    pd.testing.assert_frame_equal(
        full.iloc[:cut][columns].reset_index(drop=True),
        truncated[columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_a_subject_never_mirrors_itself_in_the_partner_column():
    """`other_{slug}_wday_hour` is about the OTHER people; a subject's own column
    would be `wday_hour` under a second name, and a tree will split on it twice."""
    days = departure.feature_frame(
        departure.label_days(_history([6.0] * 6)))
    own = f"other_{SUBJECT}_wday_hour"
    if own in days.columns:
        assert days[own].isna().all()


# --- the routine, which is what times the next-change row ------------------

MONDAY = "2026-03-02"


def _weeks(by_weekday: dict[int, tuple[float, float] | None], weeks: int = 9,
           skip: dict[int, set[int]] | None = None) -> pd.DataFrame:
    """Consecutive days from a Monday; `by_weekday[dow]` is (leave, return), or
    None for a day spent in. `skip[dow]` names week indices to stay in on."""
    skip = skip or {}
    begin = dt.date.fromisoformat(MONDAY)
    frames = []
    for offset in range(weeks * 7):
        date = begin + dt.timedelta(days=offset)
        hours = by_weekday.get(date.weekday())
        spec = {}
        if hours is not None and offset // 7 not in skip.get(date.weekday(), set()):
            leave, back = hours
            for slot in range(int(leave * 2), int(back * 2)):
                spec[f"{slot // 2:02d}:{(slot % 2) * 30:02d}"] = 0.0
        frames.append(_table(spec, date=date.isoformat()))
    return pd.concat(frames, ignore_index=True)


def _routine(**kwargs) -> dict:
    return departure.fit_routine(departure.label_days(_weeks(**kwargs)))


def test_the_routine_measures_each_weekday_on_its_own_days():
    """THE INCIDENT, from the fitting end. A weekend hour is a weekend fact; a
    routine reporting the working-day hour for it named an hour that weekday
    never earned."""
    routine = _routine(by_weekday={0: (8.0, 17.0), 1: (8.0, 17.0), 2: (8.0, 17.0),
                                   3: (8.0, 17.0), 4: (8.0, 17.0),
                                   5: (9.5, 12.0), 6: None})
    saturday = routine[SUBJECT]["by_weekday"]["5"]
    assert (saturday["n"], saturday["n_left"]) == (9, 9)
    assert saturday["departure_hour"] == 9.5 and saturday["departure_n"] == 9
    assert saturday["return_hour"] == 12.0
    # The overall median is dominated by the five working days, which is exactly
    # why a weekday is never allowed to borrow it -- see `today` below.
    assert routine[SUBJECT]["overall"]["departure_hour"] == 8.0


def test_a_weekday_she_never_leaves_on_publishes_no_hour_at_all():
    """An ANSWER, not a gap: the overall median here would be an hour nobody
    earned, which an automation cannot tell from a measured one."""
    routine = _routine(by_weekday={0: (8.0, 17.0), 6: None})
    sunday = departure.today(routine, SUBJECT,
                             pd.Timestamp("2026-03-08T06:00Z"))
    assert sunday["weekday"] == 6
    assert sunday["departure_from"] == "never"
    assert sunday["departure_hour"] is None


def test_a_weekday_with_too_few_departures_is_labelled_overall_not_measured():
    """A weekday with a couple of departures against nine observed. The hour
    still has to come from somewhere, so it comes from the overall median AND
    says so -- the label is what stops `_next_change` acting on it."""
    routine = _routine(by_weekday={0: (8.0, 17.0), 5: (9.5, 12.0)},
                       skip={5: {0, 1, 2, 3, 4, 5, 6}})       # 2 Saturdays out of 9
    saturday = departure.today(routine, SUBJECT,
                               pd.Timestamp("2026-03-07T06:00Z"))
    assert saturday["n_weekday"] == 9 and saturday["n_left_weekday"] == 2
    assert saturday["departure_from"] == "overall"
    assert saturday["departure_hour"] == 8.0, "the working-day median, flagged as such"


def test_the_house_gets_a_routine_unlike_the_out_routine():
    """A house goes to no office, but it does empty and fill -- and the
    next-change row draws it."""
    table = pd.concat([_weeks(by_weekday={0: (8.0, 17.0), 5: (9.5, 12.0)}),
                       _weeks(by_weekday={0: (8.0, 17.0), 5: (9.5, 12.0)})
                       .assign(subject=config.HOUSE_SLUG)], ignore_index=True)
    routine = departure.fit_routine(departure.label_days(table))
    assert config.HOUSE_SLUG in routine


def test_too_little_history_publishes_no_routine_at_all():
    """Below the floor the per-weekday medians are single observations wearing a
    median's clothes."""
    assert _routine(by_weekday={0: (8.0, 17.0)}, weeks=4) == {}


def test_a_routine_survives_a_round_trip_and_a_corrupt_file(tmp_path):
    routine = _routine(by_weekday={0: (8.0, 17.0), 5: (9.5, 12.0)})
    departure.save_routine(routine, tmp_path)
    assert departure.load_routine(tmp_path)[SUBJECT]["n_left"] == \
        routine[SUBJECT]["n_left"]

    (tmp_path / departure.ROUTINE_NAME).write_text("{not json")
    assert departure.load_routine(tmp_path) == {}, \
        "a truncated write must not take the add-on down"
    assert departure.load_routine(tmp_path / "nowhere") == {}
