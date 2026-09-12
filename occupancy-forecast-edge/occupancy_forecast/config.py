"""What this installation looks like, and the constants that do not vary.

Modelling constants are universal; identity is per-install and discovered; the
crossing cuts are per-install and DECLARED, as no history can measure what a
wrong one costs. `configure()` loads them once and `require()` raises if not.
"""

from __future__ import annotations

import zoneinfo
import json
import re
import os
import time

from . import log

_log = log.get(__name__)
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Universal: measured, not configured
# ---------------------------------------------------------------------------

# The resample puts people on a common grid and averages away GPS jitter; it
# does not invent resolution.
GRID_MINUTES = 30
SLOTS_PER_DAY = 24 * 60 // GRID_MINUTES  # 48

# A slot needs this much of its duration observed, or it is NaN, not a guess.
MIN_SLOT_COVERAGE = 0.5

# Longest silence still counted as observed: above the ~5-6 h nightly phone doze
# (blanking it deletes every night), far below a real multi-day outage.
MAX_SILENCE_H = 12

# Two families are fitted over this range and the gate picks per horizon.
HORIZONS_H = tuple(range(1, 49))

# The subset with its own entity; the full curve rides as a JSON attribute, so
# a subject does not get 48 near-identical sensors.
SENSOR_HORIZONS_H = (1, 2, 3, 6, 12, 24, 36, 48)

# Default for `Settings.forecast_retention_days`. Not training data, so keeping
# it forever would be storage for nobody.
FORECAST_RETENTION_DAYS = 30

# No time cap: Proximity fires only when the distance CHANGES, so its silence
# means "has not moved", and a cap mostly blanks people sitting at home.
DISTANCE_STALE_MIN = None

HOUSE_SLUG = "house"

# The state Home Assistant uses for "in the home zone". Universal.
HOME_STATE = "home"

# One definition on purpose: the collector, the trigger filter and the slot
# integrator have to agree on what "no reading" looks like.
EMPTY_STATES = frozenset({"unknown", "unavailable", "", "none"})


def is_empty(value) -> bool:
    """True when a state means "no reading" rather than a reading."""
    return value is None or str(value).strip().lower() in EMPTY_STATES

DATA_DIR = Path("/data")
CONFIG_PATH = DATA_DIR / "config.json"
MODELS_DIR = DATA_DIR / "models"
FEATURES_PATH = DATA_DIR / "features.parquet"
HISTORY_DB = DATA_DIR / "history.db"


# ---------------------------------------------------------------------------
# Per-installation identity
# ---------------------------------------------------------------------------

def slugify(entity_id: str) -> str:
    """`person.alice_smith` -> `alice_smith`."""
    return re.sub(r"[^a-z0-9_]+", "_", entity_id.split(".", 1)[-1].lower()).strip("_")


@dataclass
class Subject:
    """One thing whose occupancy is forecast. `slug` is load-bearing in five
    places, so it is derived once, de-duplicated, and never equals HOUSE_SLUG.
    """
    slug: str
    entity_id: str          # person.* or the house group
    is_person: bool = True
    distance_entity: str | None = None  # sensor.*_distance, optional
    direction_entity: str | None = None # sensor.*_direction_of_travel, optional
    next_alarm_entity: str | None = None # sensor.*_next_alarm, optional


@dataclass(frozen=True)
class Zone:
    """One place the user ticked, deliberately roleless: which mean "work" is
    the model's problem. `name` is snapshotted because HA writes a zone's NAME
    into a person's state; see `features._resolve_zone_events`.
    """
    slug: str          # slugify(entity_id): zone.alice_office -> alice_office
    entity_id: str
    name: str


# `crossing_min_hours` is 2, not 1: a 30-minute-slot target cannot represent an
# absence under about an hour, so one hour past the line is noise.
DEFAULT_DEPARTURE_THRESHOLD = 0.5
DEFAULT_ARRIVAL_THRESHOLD = 0.5
DEFAULT_CROSSING_MIN_HOURS = 2

# Write-only settings: stored and used, never served back. See `Settings.public`.
SECRETS = ("influx_token", "mqtt_password")


@dataclass
class Settings:
    """Everything that differs between installations, persisted to config.json.
    `holiday_country` None falls back to HA's `country`, "" means no holidays;
    the crossing cuts are a band, `departure_threshold <= arrival_threshold`.
    """
    people: list[str] = field(default_factory=list)      # person.* entity ids
    zones: list[str] = field(default_factory=list)       # zone.* entity ids, enabled
    # zone entity -> friendly name, refreshed every boot: the feature build
    # never talks to Home Assistant, so it cannot look these up itself.
    zone_names: dict[str, str] = field(default_factory=dict)
    house_entity: str | None = None                      # group.* or None -> OR over people
    proximity: dict[str, list[str]] = field(default_factory=dict)  # person -> [distance, direction]
    # person -> sensor.*_next_alarm. None is "never looked", {} is "looked and
    # there are none"; only None is re-discovered.
    next_alarm: dict[str, str] | None = None
    timezone: str = "UTC"
    country: str | None = None                           # Home Assistant's. Never the user's.
    holiday_country: str | None = None                   # The user's. See the docstring.
    units: dict[str, str] = field(default_factory=dict)  # entity id -> unit_of_measurement
    home_latitude: float | None = None
    home_longitude: float | None = None
    source: str = "store"                                # "store" | "influx"
    # Where the history and the broker live. Panel-owned since 0.4.0; the
    # matching add-on options are read once, by `runtime.import_legacy_options`.
    influx_url: str = ""
    influx_org: str = ""
    influx_bucket: str = "homeassistant"
    influx_token: str = ""                               # secret; see SECRETS
    mqtt_host: str = ""                                  # empty -> Supervisor's broker
    mqtt_port: int = 1883
    mqtt_user: str = ""
    mqtt_password: str = ""                              # secret; see SECRETS
    mqtt_ssl: bool = False
    # When the add-on options were imported, so "where did my token go" has an
    # answer. Its presence is also what stops the import running twice.
    migrated_from_options: str | None = None
    # A `schedule.*` entity shading the night on the chart; decoration only.
    day_schedule: str | None = None
    # How the curve is reduced to "hours until away/home". See the docstring.
    departure_threshold: float = DEFAULT_DEPARTURE_THRESHOLD
    arrival_threshold: float = DEFAULT_ARRIVAL_THRESHOLD
    crossing_min_hours: int = DEFAULT_CROSSING_MIN_HOURS
    # Days; 0 never prunes, and lowering it deletes on the next cycle, for good.
    forecast_retention_days: int = FORECAST_RETENTION_DAYS

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def public(self) -> dict:
        """Everything `GET /api/config` may serve. Never a secret's VALUE.

        That GET is deliberately open -- the panel needs it on load -- so a
        secret here would be readable by anyone who can open the panel, not only
        by `admin_users`. A `_set` flag says whether one is stored; a mask would
        be written straight back as the password by the next save.
        """
        out = asdict(self)
        for name in SECRETS:
            out[f"{name}_set"] = bool(out.pop(name))
        return out

    @classmethod
    def from_json(cls, text: str) -> "Settings":
        raw = cls._migrate(json.loads(text))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def _migrate(raw: dict) -> dict:
        """Bring an older config.json forward; runs BEFORE the field filter.
        `office_zones`'s values become the enabled zones and the key is dropped,
        because two spellings of one setting drift apart.
        """
        if "office_zones" in raw:
            legacy = raw.pop("office_zones") or {}
            if "zones" not in raw:
                raw["zones"] = sorted({z for z in legacy.values() if z})
        return raw

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        # Flushed BEFORE the rename: the rename is atomic, the bytes are not.
        with tmp.open("w") as fh:
            fh.write(self.to_json())
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings | None":
        if not path.exists():
            return None
        return cls.from_json(path.read_text())


# Populated by configure(). Read by every other module.
SETTINGS: Settings | None = None
SUBJECTS: tuple[Subject, ...] = ()
PEOPLE: tuple[Subject, ...] = ()
ZONES: tuple[Zone, ...] = ()
TIMEZONE: str = "UTC"


def tzinfo() -> "zoneinfo.ZoneInfo":
    """`TIMEZONE` as a real tzinfo, for the code that is not pandas. Falls back
    to UTC: a bad timezone should cost an hour of accuracy, not the start-up.
    """
    global _tz_warned
    try:
        return zoneinfo.ZoneInfo(TIMEZONE)
    except Exception as err:  # noqa: BLE001
        # Said once: silent, this shifted labels by an hour in one half of the
        # stack while `features._localise` raised on the same zone.
        if _tz_warned != TIMEZONE:
            _tz_warned = TIMEZONE
            _log.warning("timezone %r is not usable (%s); local dates and "
                         "hours are being computed in UTC", TIMEZONE, err)
        return zoneinfo.ZoneInfo("UTC")


_tz_warned: str | None = None
HOLIDAY_COUNTRY: str | None = None
HOME_COORDS: tuple[float, float] | None = None
DAY_SCHEDULE: str | None = None
DEPARTURE_THRESHOLD: float = DEFAULT_DEPARTURE_THRESHOLD
ARRIVAL_THRESHOLD: float = DEFAULT_ARRIVAL_THRESHOLD
CROSSING_MIN_HOURS: int = DEFAULT_CROSSING_MIN_HOURS


def _crossing_cut(value, default: float | int, low: float, high: float):
    """One crossing cut, clamped to something servable; never raises. A
    hand-edited config.json reaches `configure()` unvalidated, so this clamps
    and leaves rejecting a bad value loudly to the API.
    """
    try:
        # bool is an int in Python and json.loads turns `true` into one, so it
        # would otherwise clamp to 1.0 rather than being caught as nonsense.
        if isinstance(value, bool):
            raise TypeError(value)
        number = type(default)(value)
    except (TypeError, ValueError):
        _log.warning("ignoring unusable crossing cut %r, using %s", value, default)
        return default
    clamped = type(default)(min(max(number, low), high))
    if clamped != number:
        _log.warning("clamping crossing cut %s to %s", number, clamped)
    return clamped


def configure(settings: Settings) -> tuple[Subject, ...]:
    """Apply an installation's identity. Call once at startup."""
    global SETTINGS, SUBJECTS, PEOPLE, ZONES, TIMEZONE, HOLIDAY_COUNTRY, HOME_COORDS
    global DEPARTURE_THRESHOLD, ARRIVAL_THRESHOLD, CROSSING_MIN_HOURS, DAY_SCHEDULE

    if not settings.people:
        raise ValueError(
            "no people configured: pick at least one person.* entity. Occupancy "
            "is the one thing this cannot be built without.")

    seen: set[str] = {HOUSE_SLUG}
    people: list[Subject] = []
    for entity_id in settings.people:
        slug = slugify(entity_id)
        # A slug equal to "house" or another person's would give duplicate
        # (subject, time) keys and blow up much later in a pandas reindex.
        base, n = slug, 2
        while slug in seen:
            slug, n = f"{base}_{n}", n + 1
        seen.add(slug)
        proximity = settings.proximity.get(entity_id) or [None, None]
        people.append(Subject(
            slug=slug, entity_id=entity_id, is_person=True,
            distance_entity=proximity[0], direction_entity=proximity[1],
            next_alarm_entity=(settings.next_alarm or {}).get(entity_id)))

    house = Subject(slug=HOUSE_SLUG, entity_id=settings.house_entity or "",
                    is_person=False)

    # Two zones that slugify alike would silently merge into one column.
    seen_zones: set[str] = set()
    zones: list[Zone] = []
    for entity_id in settings.zones:
        slug = slugify(entity_id)
        base, n = slug, 2
        while slug in seen_zones:
            slug, n = f"{base}_{n}", n + 1
        seen_zones.add(slug)
        zones.append(Zone(slug=slug, entity_id=entity_id,
                          name=settings.zone_names.get(entity_id, "")))

    SETTINGS = settings
    PEOPLE = tuple(people)
    ZONES = tuple(zones)
    SUBJECTS = (*people, house)
    TIMEZONE = settings.timezone or "UTC"
    HOLIDAY_COUNTRY = (settings.holiday_country if settings.holiday_country is not None
                       else settings.country)
    HOME_COORDS = ((settings.home_latitude, settings.home_longitude)
                   if settings.home_latitude is not None else None)
    DAY_SCHEDULE = (settings.day_schedule or None) if settings else None
    # Open at both ends: a rounded curve never meets a cut of exactly 0 or 1.
    DEPARTURE_THRESHOLD = _crossing_cut(
        settings.departure_threshold, DEFAULT_DEPARTURE_THRESHOLD, 0.01, 0.99)
    ARRIVAL_THRESHOLD = _crossing_cut(
        settings.arrival_threshold, DEFAULT_ARRIVAL_THRESHOLD, 0.01, 0.99)
    CROSSING_MIN_HOURS = _crossing_cut(
        settings.crossing_min_hours, DEFAULT_CROSSING_MIN_HOURS,
        1, max(HORIZONS_H))
    return SUBJECTS


def require() -> Settings:
    if SETTINGS is None:
        raise RuntimeError(
            "occupancy_forecast.config.configure() has not been called -- the add-on "
            "loads its settings from /data/config.json at startup, and the "
            "feature builder cannot know which entities to read without them.")
    return SETTINGS


def subject(slug: str) -> Subject:
    for item in SUBJECTS:
        if item.slug == slug:
            return item
    raise KeyError(slug)


def all_slugs() -> tuple[str, ...]:
    return tuple(s.slug for s in SUBJECTS)


def zone_slugs() -> tuple[str, ...]:
    """The enabled zones. Empty is fine -- there are simply no zone columns."""
    return tuple(z.slug for z in ZONES)


def zone_name_map() -> dict[str, str]:
    """Lowercased friendly name -> zone entity id, for the enabled zones. A zone
    with no name adds nothing, not a ""-keyed entry that swallows blank states.
    """
    return {z.name.strip().lower(): z.entity_id for z in ZONES if z.name.strip()}


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------

# Deliberately NO host default: an unset variable must fail, not silently dial
# somebody else's broker.

def mqtt_settings(settings: "Settings | None" = None) -> dict:
    """A broker set in the panel wins; otherwise Supervisor's own mqtt service.

    The service is DISCOVERY, not configuration -- it is how an ordinary
    Mosquitto install works with the panel's broker card left empty -- so it
    stays in the environment rather than moving into config.json.
    """
    chosen = settings if settings is not None else SETTINGS
    if chosen is not None and chosen.mqtt_host:
        return {
            "host": chosen.mqtt_host,
            "port": int(chosen.mqtt_port or 1883),
            "username": chosen.mqtt_user or None,
            "password": chosen.mqtt_password or None,
            "ssl": bool(chosen.mqtt_ssl),
        }
    host = os.environ.get("MQTT_HOST")
    if not host:
        raise RuntimeError(
            "no MQTT broker: nothing in the panel's broker card and no mqtt "
            "service from Supervisor. Fill in the broker on the add-on's "
            "Connections tab, or install the Mosquitto add-on.")
    return {
        "host": host,
        "port": int(os.environ.get("MQTT_PORT", "1883")),
        "username": os.environ.get("MQTT_USER") or None,
        "password": os.environ.get("MQTT_PASSWORD") or None,
        # bashio prints the service's boolean as the word; anything else is no.
        "ssl": (os.environ.get("MQTT_SSL") or "").strip().lower() == "true",
    }


# Ingress authenticates but does not authorise, and forwards no admin flag.
# Supervisor sets `X-Remote-User-Id` and strips any client copy, so it cannot be
# forged. EMPTY MEANS EVERYONE, or an upgrade would lock the owner out.
def admin_users() -> frozenset[str]:
    """Home Assistant user ids allowed to POST. Empty set means unrestricted."""
    raw = os.environ.get("OCCUPANCY_ADMIN_USERS", "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


# Topic root and client id derive from the slug minus Supervisor's repository
# token, so stable and edge never share a client id and a move renames nothing.
DEFAULT_TOPIC_PREFIX = "occupancy_forecast"

# Cached ONLY once Supervisor has answered: caching the default on failure made
# edge a second stable build on MQTT until its next restart.
_topic_prefix: str | None = None
_topic_prefix_error: str | None = None


def _supervisor_slug(token: str, timeout: float) -> str:
    import json
    import urllib.request
    request = urllib.request.Request(
        "http://supervisor/addons/self/info",
        headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["data"]["slug"]


def resolve_topic_prefix(attempts: int = 1, delay: float = 0.0,
                         timeout: float = 5.0) -> bool:
    """Ask Supervisor for this add-on's slug; True once the prefix is known.
    The only network call for it. Outside an add-on the default is cached at
    once; inside one a failure caches NOTHING, so the next call asks again.
    """
    global _topic_prefix, _topic_prefix_error
    if _topic_prefix is not None:
        return True
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        _topic_prefix = DEFAULT_TOPIC_PREFIX
        return True
    for attempt in range(1, attempts + 1):
        try:
            slug = _supervisor_slug(token, timeout)
            _, _, name = slug.partition("_")     # drop the repository prefix
            _topic_prefix = name or DEFAULT_TOPIC_PREFIX
            _topic_prefix_error = None
            return True
        except Exception as err:  # noqa: BLE001
            _topic_prefix_error = str(err)
            _log.error("could not read this add-on's slug from Supervisor "
                       "(attempt %d of %d: %s). Nothing is published to MQTT "
                       "until it can be read: a guessed prefix would collide "
                       "with any other build of this add-on.",
                       attempt, attempts, err)
            if attempt < attempts and delay > 0:
                time.sleep(delay)
    return False


def topic_prefix_resolved() -> bool:
    """Whether `topic_prefix()` is this add-on's own, rather than a guess."""
    return _topic_prefix is not None or not os.environ.get("SUPERVISOR_TOKEN")


def topic_prefix_error() -> str | None:
    return None if topic_prefix_resolved() else _topic_prefix_error


def topic_prefix() -> str:
    """The MQTT topic root, never via the network. Unresolved, it returns the
    default without remembering it; writers ask `topic_prefix_resolved()` first.
    """
    if _topic_prefix is None and not os.environ.get("SUPERVISOR_TOKEN"):
        resolve_topic_prefix()      # no network outside an add-on; caches
    return _topic_prefix or DEFAULT_TOPIC_PREFIX


def display_name() -> str:
    """`occupancy_forecast_edge` -> "Occupancy Forecast Edge", so a message says
    which build wrote it. It also feeds the MQTT device name, which HA builds
    entity ids from: a casing change here silently orphans every entity.
    """
    return topic_prefix().replace("_", " ").title()
