"""Composition root: assemble settings, Home Assistant and a history source.

Everything else takes what it needs as an argument, so a test can build its own
combination without touching a network.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config, discover
from .sources import HistoryStore, HomeAssistant, InfluxSource, StoreSource


def home_assistant() -> HomeAssistant:
    return HomeAssistant()


def load_settings(ha: HomeAssistant | None = None,
                  path: Path = config.CONFIG_PATH) -> config.Settings:
    """Saved settings, or a proposal from what Home Assistant currently has.

    First run has no config.json, so the add-on proposes something workable for
    the user to confirm rather than refusing to start.
    """
    settings = config.Settings.load(path)
    if settings is None:
        settings = discover.propose(ha or home_assistant())
        settings.save(path)
    return settings


def refresh_environment(settings: config.Settings, ha: HomeAssistant) -> config.Settings:
    """Re-read what is Home Assistant's to change; never the holiday calendar.

    HA's country only seeds it; re-reading would silently undo the user's pick.
    """
    core = ha.config()
    settings.timezone = core.get("time_zone") or "UTC"
    settings.country = core.get("country")
    if settings.holiday_country is None:
        settings.holiday_country = core.get("country")
    settings.home_latitude = core.get("latitude")
    settings.home_longitude = core.get("longitude")

    # The add-on option wins: a history source is infrastructure, not identity.
    settings.source = os.environ.get("OCCUPANCY_SOURCE") or settings.source

    numeric = [pair[0] for pair in settings.proximity.values() if pair and pair[0]]
    states = ha.states() if (numeric or settings.next_alarm is None
                             or settings.zones) else None
    if numeric:
        settings.units = discover.units_for(states, numeric)

    # A zone's name is the only per-person zone signal in history, and HA's to
    # rename, so the snapshot is re-read like the timezone.
    if settings.zones:
        settings.zone_names = discover.zone_names(states, settings.zones)

    # `is None`, not falsiness: an empty dict means somebody looked and found
    # nothing, and re-running discovery over that would undo the choice.
    if states is not None and settings.next_alarm is None:
        found = {p: discover.match_next_alarm(p, states) for p in settings.people}
        settings.next_alarm = {p: e for p, e in found.items() if e}
    return settings


def build_source(settings: config.Settings, ha: HomeAssistant,
                 store: HistoryStore | None = None):
    """The history source named by the settings.

    `influx` lets an install that already archives HA keep the months it has.
    """
    if settings.source == "influx":
        url = os.environ.get("INFLUX_URL")
        token = os.environ.get("INFLUX_TOKEN")
        org = os.environ.get("INFLUX_ORG")
        if not (url and token and org):
            raise RuntimeError(
                "source is 'influx' but INFLUX_URL / INFLUX_TOKEN / INFLUX_ORG "
                "are not all set in the add-on options")
        return InfluxSource(url, token, org,
                            bucket=os.environ.get("INFLUX_BUCKET", "homeassistant"),
                            units=settings.units)
    return StoreSource(store or forecast_log(), ha)


def forecast_log() -> HistoryStore:
    """Where the add-on records what it published, whatever the source is.

    Never reached through the source: `.store` is what says "not on Influx".
    """
    return HistoryStore(config.HISTORY_DB)


def tracked_entities(settings: config.Settings) -> list[str]:
    """Everything the collector should be pulling into the store."""
    wanted: list[str] = list(settings.people)
    if settings.house_entity:
        wanted.append(settings.house_entity)
    # No feature reads a zone's count, but uncollected history is gone forever.
    wanted.extend(settings.zones)
    for pair in settings.proximity.values():
        wanted.extend(e for e in (pair or []) if e)
    # Collected but not served yet (features.BUILT_NOT_SHIPPED): collection has
    # to start well before the feature does.
    wanted.extend(e for e in (settings.next_alarm or {}).values() if e)
    return sorted(set(wanted))


def absence_entities(settings: config.Settings) -> list[str]:
    """Entities whose `unavailable` is a reading rather than a gap.

    Next-alarm sensors, which read `unavailable` exactly when no alarm is set.
    """
    return sorted(e for e in (settings.next_alarm or {}).values() if e)


def presence_entities(settings: config.Settings) -> list[str]:
    """Entities whose `unknown` ends the previous state rather than being a gap.

    Listed, not prefix-matched: the house entity is not always a `group.*`.
    """
    wanted = list(settings.people)
    if settings.house_entity:
        wanted.append(settings.house_entity)
    return sorted(set(wanted))


def trigger_entities(settings: config.Settings) -> list[str]:
    """The subset of `tracked_entities` whose change is worth re-predicting for.

    Not proximity: it rewrites every few minutes, and a slot averages it anyway.
    """
    wanted: list[str] = list(settings.people)
    if settings.house_entity:
        wanted.append(settings.house_entity)
    wanted.extend(settings.zones)
    return sorted(set(e for e in wanted if e))


def bootstrap(path: Path = config.CONFIG_PATH):
    """Everything, wired. Returns (settings, ha, source, forecast_log)."""
    ha = home_assistant()
    settings = refresh_environment(load_settings(ha, path), ha)
    settings.save(path)
    config.configure(settings)
    log = forecast_log()
    return settings, ha, build_source(settings, ha, log), log
