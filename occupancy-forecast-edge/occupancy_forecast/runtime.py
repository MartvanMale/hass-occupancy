"""Composition root: assemble settings, Home Assistant and a history source.

Everything else takes what it needs as an argument, so a test can build its own
combination without touching a network.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

from . import config, discover
from .sources import HistoryStore, HomeAssistant, InfluxSource, StoreSource

_log = logging.getLogger(__name__)


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


# The add-on options these settings used to be read from. Kept in config.yaml for
# a release or two so an existing install carries its values across; removing a
# key makes Supervisor drop the stored value before any of our code runs.
_LEGACY_OPTIONS = {
    "source": ("OCCUPANCY_SOURCE", str),
    "influx_url": ("INFLUX_URL", str),
    "influx_org": ("INFLUX_ORG", str),
    "influx_bucket": ("INFLUX_BUCKET", str),
    "influx_token": ("INFLUX_TOKEN", str),
    # The LEGACY_ names, not MQTT_*: those also carry Supervisor's discovered
    # broker, and importing that would freeze today's address into config.json
    # where it would then win over the discovery that is meant to track it.
    "mqtt_host": ("OCCUPANCY_LEGACY_MQTT_HOST", str),
    "mqtt_port": ("OCCUPANCY_LEGACY_MQTT_PORT", int),
    "mqtt_user": ("OCCUPANCY_LEGACY_MQTT_USER", str),
    "mqtt_password": ("OCCUPANCY_LEGACY_MQTT_PASSWORD", str),
    "mqtt_ssl": ("OCCUPANCY_LEGACY_MQTT_SSL", bool),
}


def import_legacy_options(settings: config.Settings) -> bool:
    """Adopt the add-on options these settings moved out of. Once, ever.

    Guarded on the marker rather than on each field being empty: "every boot"
    would re-apply the option over a panel edit, which is the bug this move
    exists to fix, and "only the first version" would miss anyone who updates
    straight past it.
    """
    if settings.migrated_from_options:
        return False
    taken: list[str] = []
    for field_name, (variable, kind) in _LEGACY_OPTIONS.items():
        raw = os.environ.get(variable)
        if raw is None or raw == "":
            continue
        if kind is bool:
            value = raw.strip().lower() == "true"
        elif kind is int:
            try:
                value = int(raw)
            except ValueError:
                continue
        else:
            value = raw
        setattr(settings, field_name, value)
        taken.append(field_name)
    settings.migrated_from_options = dt.datetime.now(
        dt.timezone.utc).isoformat(timespec="seconds")
    if taken:
        # Named, without values: one of them is a token.
        _log.info("imported %s from the add-on options into the panel's "
                  "settings; change them on the Connections tab from now on",
                  ", ".join(sorted(taken)))
    return bool(taken)


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
        if not (settings.influx_url and settings.influx_token and settings.influx_org):
            # Refused, never a quiet fall back to `store`: that would start a
            # fresh empty archive and look healthy for the ten days it takes to
            # notice.
            raise RuntimeError(
                "the history source is 'influx' but the URL, token and org are "
                "not all set. Fill them in on the add-on's Connections tab -- "
                "they moved there from the add-on options in 0.4.0.")
        return InfluxSource(settings.influx_url, settings.influx_token,
                            settings.influx_org,
                            bucket=settings.influx_bucket or "homeassistant",
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
    settings = load_settings(ha, path)
    # Before anything reads them, and only here: `refresh_environment` also runs
    # on every save, where re-importing would undo the edit being saved.
    import_legacy_options(settings)
    settings = refresh_environment(settings, ha)
    settings.save(path)
    config.configure(settings)
    log = forecast_log()
    return settings, ha, build_source(settings, ha, log), log
