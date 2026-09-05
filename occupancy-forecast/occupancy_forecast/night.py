"""When the household is asleep, for shading the forecast chart.

**Decoration, and deliberately isolated.** Nothing here touches a feature, a
model or a published entity: it exists so the 48-hour chart can grey out the
hours nobody is expected to be awake, which is what makes a dip at 03:00 read
differently from a dip at 15:00. If it fails, or if no schedule is configured,
the chart simply has no bands.

**Why the pattern is recovered from history rather than read off the entity.**
A `schedule.*` entity publishes its current state and its next event, not its
week. The chart needs the next 48 HOURS, which is future state, and Home
Assistant will not tell anyone what a schedule is going to do. A weekly
schedule does repeat, though, so a week of its own history is the schedule --
sampled on a grid and projected forward. A household that changes its schedule
sees the chart follow within a week, which for shading is soon enough.
"""
from __future__ import annotations

import datetime as dt

from . import config, log

_log = log.get(__name__)

# The sampling grid. Fifteen minutes is finer than any bedtime is meaningful
# and keeps a week to 672 samples.
STEP_MIN = 15
WEEK_DAYS = 7


def _sample(changes: list[dict], at: dt.datetime) -> str | None:
    """The state in force at `at`, from a list of state CHANGES."""
    seen = None
    for row in changes:
        stamp = row.get("last_changed") or row.get("last_updated")
        if not stamp:
            continue
        when = dt.datetime.fromisoformat(stamp)
        if when > at:
            break
        seen = row.get("state")
    return seen


def weekly_pattern(changes: list[dict], now: dt.datetime) -> dict[tuple[int, int], bool]:
    """`(weekday, slot) -> is the household awake`, from a week of history.

    Slots are `STEP_MIN` minutes from local midnight. Keyed on local time
    because a schedule is a local thing and the rest of this package is UTC.
    """
    pattern: dict[tuple[int, int], bool] = {}
    if not changes:
        return pattern
    start = now - dt.timedelta(days=WEEK_DAYS)
    steps = WEEK_DAYS * 24 * 60 // STEP_MIN
    for i in range(steps):
        at = start + dt.timedelta(minutes=i * STEP_MIN)
        state = _sample(changes, at)
        if state in (None, "unavailable", "unknown"):
            continue
        local = at.astimezone(config.tzinfo())
        slot = (local.hour * 60 + local.minute) // STEP_MIN
        pattern[(local.weekday(), slot)] = state == "on"
    return pattern


def bands(pattern: dict[tuple[int, int], bool], now: dt.datetime,
          hours: int) -> list[dict]:
    """Contiguous asleep runs over the next `hours`, as offsets FROM NOW.

    Hours rather than timestamps because that is the chart's x-axis: it plots
    horizon, not clock time, so a band has to be expressed the same way or it
    would land in the wrong place the moment the two disagree.

    A slot the pattern never saw is treated as awake -- no shading. Inventing a
    night from missing history would be the one failure mode worth avoiding,
    since the whole point of the band is to explain a dip.
    """
    if not pattern:
        return []
    out: list[dict] = []
    steps = int(hours * 60 / STEP_MIN)
    open_at: float | None = None
    for i in range(steps + 1):
        at = (now + dt.timedelta(minutes=i * STEP_MIN)).astimezone(config.tzinfo())
        slot = (at.hour * 60 + at.minute) // STEP_MIN
        asleep = pattern.get((at.weekday(), slot), True) is False
        offset = i * STEP_MIN / 60
        if asleep and open_at is None:
            open_at = offset
        elif not asleep and open_at is not None:
            out.append({"from": open_at, "to": offset})
            open_at = None
    if open_at is not None:
        out.append({"from": open_at, "to": steps * STEP_MIN / 60})
    return out


def night_bands(ha, now: dt.datetime, hours: int) -> list[dict]:
    """The bands for the configured schedule, or nothing at all.

    Never raises. A chart that cannot be shaded is a chart without shading, and
    an add-on that refused to serve its forecast because a decoration failed
    would be a poor trade.
    """
    entity = config.DAY_SCHEDULE
    if not entity or ha is None:
        return []
    try:
        start = (now - dt.timedelta(days=WEEK_DAYS)).isoformat(timespec="seconds")
        series = ha.history([entity], start) or [[]]
        return bands(weekly_pattern(series[0], now), now, hours)
    except Exception as err:  # noqa: BLE001
        _log.warning("could not read %s for the night shading: %s", entity, err)
        return []
