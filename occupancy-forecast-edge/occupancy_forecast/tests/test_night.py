"""The night shading: a weekly pattern recovered from a week of history.

Decoration, but the failure modes are the interesting kind. Inventing a night
out of missing history would be worse than no shading at all, because the band
exists to explain a dip.
"""
import datetime as dt

import pytest

from occupancy_forecast import config, night
from occupancy_forecast.tests.conftest import settings as make_settings

UTC = dt.timezone.utc


@pytest.fixture(autouse=True)
def _configured():
    config.configure(make_settings())


def _changes(now, awake_from=7, awake_to=23):
    """A schedule that has been on 07:00-23:00 local, every day, for a week."""
    out = []
    start = now - dt.timedelta(days=night.WEEK_DAYS)
    for day in range(night.WEEK_DAYS + 1):
        midnight = (start + dt.timedelta(days=day)).astimezone(
            config.tzinfo()).replace(hour=0, minute=0, second=0, microsecond=0)
        for hour, state in ((awake_from, "on"), (awake_to, "off")):
            out.append({"last_changed": (midnight + dt.timedelta(hours=hour)).isoformat(),
                        "state": state})
    return sorted(out, key=lambda r: r["last_changed"])


def test_a_week_of_history_becomes_the_week():
    now = dt.datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    pattern = night.weekly_pattern(_changes(now), now)
    assert pattern, "a full week of changes produced no pattern at all"
    # Midday awake, small hours asleep, on every weekday the pattern saw.
    for weekday in range(7):
        assert pattern.get((weekday, 12 * 4)) is True
        assert pattern.get((weekday, 3 * 4)) is False


def test_the_bands_are_offsets_from_now_not_clock_times():
    """The chart plots horizon, not clock time. A band in clock time would land
    in the wrong place the moment the two disagree."""
    now = dt.datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    bands = night.bands(night.weekly_pattern(_changes(now), now), now, 48)
    assert bands, "a schedule that sleeps every night produced no bands"
    assert all(0 <= b["from"] < b["to"] <= 48 for b in bands)
    # Two nights in 48 hours from midday.
    assert len(bands) == 2
    for b in bands:
        assert 7 <= b["to"] - b["from"] <= 9, "a night is about eight hours"


def test_history_it_never_saw_is_not_shaded():
    """The one failure mode worth avoiding. A slot with no evidence is drawn as
    awake, because inventing a night is how a chart explains a dip that never
    happened."""
    now = dt.datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    assert night.bands({}, now, 48) == []
    assert night.weekly_pattern([], now) == {}


def test_an_unconfigured_or_unreachable_schedule_is_simply_no_shading():
    """Never raises. A chart that cannot be shaded is a chart without shading;
    refusing to serve the forecast over a decoration would be a poor trade."""
    now = dt.datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    assert night.night_bands(None, now, 48) == []

    class Exploding:
        def history(self, *_args, **_kwargs):
            raise RuntimeError("home assistant said no")

    config.DAY_SCHEDULE = "schedule.day_time"
    try:
        assert night.night_bands(Exploding(), now, 48) == []
    finally:
        config.DAY_SCHEDULE = None
