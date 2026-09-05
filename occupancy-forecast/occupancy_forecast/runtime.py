"""Composition root: assemble settings, Home Assistant and a history source.

Everything else takes what it needs as an argument. This is the one place that
knows how the pieces fit together, so a test can build its own combination
without touching a network.
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

    First run has no config.json, so rather than refusing to start, the add-on
    proposes something workable and shows it on the configuration page for the
    user to confirm. An install with one person and nothing else still gets a
    running system.
    """
    settings = config.Settings.load(path)
    if settings is None:
        settings = discover.propose(ha or home_assistant())
        settings.save(path)
    return settings


def refresh_environment(settings: config.Settings, ha: HomeAssistant) -> config.Settings:
    """Re-read the things that belong to Home Assistant, not to the user.

    Timezone, country, home coordinates and sensor units are HA's to tell us,
    and all four can change without anyone touching this add-on's settings.
    Re-reading them every startup is cheaper than a support thread.

    The holiday calendar is the one that is *not* HA's, though it starts there.
    HA's country is a fine first guess and seeds the setting on a fresh install,
    so nobody who never opens the panel notices a change. Once chosen it is the
    user's, and re-reading it here would silently undo their pick on the next
    restart -- which is the whole reason the setting was unusable before.
    """
    core = ha.config()
    settings.timezone = core.get("time_zone") or "UTC"
    settings.country = core.get("country")
    if settings.holiday_country is None:
        settings.holiday_country = core.get("country")
    settings.home_latitude = core.get("latitude")
    settings.home_longitude = core.get("longitude")

    # The add-on option wins over the persisted value: which history source to
    # read is infrastructure, set in Supervisor, not part of the household
    # identity the user edits on the panel.
    settings.source = os.environ.get("OCCUPANCY_SOURCE") or settings.source

    numeric = [pair[0] for pair in settings.proximity.values() if pair and pair[0]]
    states = ha.states() if (numeric or settings.next_alarm is None
                             or settings.zones) else None
    if numeric:
        settings.units = discover.units_for(states, numeric)

    # A zone's friendly name is the only per-person zone signal that exists in
    # history (features._resolve_zone_events), so the snapshot has to track
    # renames. Re-read here for the same reason as the timezone: it is Home
    # Assistant's to change, and nobody will think to come and re-save.
    if settings.zones:
        settings.zone_names = discover.zone_names(states, settings.zones)

    # Discovered once, for anyone who has not looked yet rather than only on a
    # fresh install: the setting arrived after the add-on shipped, so every
    # existing config.json is missing it, and a phone sensor nobody notices is
    # one that never accumulates the history it needs.
    #
    # `is None`, not falsiness. An empty dict means somebody looked and found
    # nothing, or cleared the list on purpose; re-running discovery over that
    # would undo the choice on the next restart. Same three-state reasoning as
    # `holiday_country` above.
    if states is not None and settings.next_alarm is None:
        found = {p: discover.match_next_alarm(p, states) for p in settings.people}
        settings.next_alarm = {p: e for p, e in found.items() if e}
    return settings


def build_source(settings: config.Settings, ha: HomeAssistant,
                 store: HistoryStore | None = None):
    """The history source named by the settings.

    `store` is the default and works anywhere. `influx` exists for installs that
    already archive Home Assistant and would otherwise discard months of history
    and wait six weeks for a model.
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
    return StoreSource(store or HistoryStore(config.HISTORY_DB), ha)


def tracked_entities(settings: config.Settings) -> list[str]:
    """Everything the collector should be pulling into the store."""
    wanted: list[str] = list(settings.people)
    if settings.house_entity:
        wanted.append(settings.house_entity)
    # Kept even though no feature reads a zone's own count any more: history
    # not collected is history gone forever, and the count is the fallback if
    # the name join ever has to be replaced.
    wanted.extend(settings.zones)
    for pair in settings.proximity.values():
        wanted.extend(e for e in (pair or []) if e)
    # Collected but not served yet -- see features.BUILT_NOT_SHIPPED. A phone
    # sensor is typically enabled long after the recorder started, so the only
    # way it is ever useful is for collection to start well before the feature
    # does.
    wanted.extend(e for e in (settings.next_alarm or {}).values() if e)
    return sorted(set(wanted))


def absence_entities(settings: config.Settings) -> list[str]:
    """Entities whose `unavailable` is a reading rather than a gap.

    Only the next-alarm sensors so far: they read `unavailable` exactly when no
    alarm is set, which is the commoner state and every bit as informative as a
    time. See `sources.ha.HistoryStore.collect`.
    """
    return sorted(e for e in (settings.next_alarm or {}).values() if e)


def trigger_entities(settings: config.Settings) -> list[str]:
    """The subset of `tracked_entities` whose change is worth re-predicting for.

    Deliberately NOT the full tracked set. That includes the proximity distance
    and direction-of-travel sensors, which rewrite every few minutes for as
    long as somebody is driving -- and whose contribution to the forecast is
    averaged over a 30-minute slot anyway, so re-running on each one would buy
    nothing and cost a feature rebuild every time. Presence and the zones are
    the signals where a change actually moves the answer.
    """
    wanted: list[str] = list(settings.people)
    if settings.house_entity:
        wanted.append(settings.house_entity)
    wanted.extend(settings.zones)
    return sorted(set(e for e in wanted if e))


def bootstrap(path: Path = config.CONFIG_PATH):
    """Everything, wired. Returns (settings, ha, source)."""
    ha = home_assistant()
    settings = refresh_environment(load_settings(ha, path), ha)
    settings.save(path)
    config.configure(settings)
    return settings, ha, build_source(settings, ha)
