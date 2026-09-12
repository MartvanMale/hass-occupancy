"""When the household is asleep, for shading the forecast chart.

Decoration: nothing here touches a feature, a model or a published entity, and
on failure the chart has no bands. A `schedule.*` entity publishes its current
state, never its week, so the pattern is recovered from a week of its history.
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
    Slots count `STEP_MIN` from LOCAL midnight: a schedule is a local thing.
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
    """Contiguous asleep runs over the next `hours`, as hour offsets FROM NOW
    because the chart plots horizon. A slot the pattern never saw counts as
    awake: inventing a night from missing history would defeat the band.
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
    """The bands for the configured schedule, or nothing at all. Never raises:
    a failed decoration must not cost the forecast.
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
