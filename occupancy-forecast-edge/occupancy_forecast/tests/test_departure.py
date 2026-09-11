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
