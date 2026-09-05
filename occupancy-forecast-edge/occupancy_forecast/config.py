"""What this installation looks like, and the constants that do not vary.

Split into two halves on purpose.

**The modelling constants are universal.** Grid size, horizons, the coverage
floor, the fold geometry -- these were measured, they are not preferences, and
they are the same on every install. They stay module constants.

**The identity is per-installation and discovered, never hardcoded.** Which
people, which zones, which proximity sensors, the timezone, the country, the
distance unit. In the original this was a block of literals naming two specific
people and six specific entity ids, which is precisely what made the thing
unportable.

**A decision cut is per-installation and declared, not discovered.** The
forecast is a probability; turning it into "they leave in 6 h" needs a line, and
where that line goes depends on what the answer is wired to rather than on
anything in the history. Pre-heating an hour early is cheap; dropping the house
out of comfort on a false departure is not, and no amount of history can measure
that asymmetry. Those live in `Settings` beside the identity, and for the same
reason: nothing here can measure them. They change no model, no feature and no
forecast -- only how the finished curve is read. See `predict._crossing`.

Identity is loaded once at startup by `configure()` and read from module globals
afterwards, because every other module was written against `config.X` and
threading a settings object through all of them would be a large, risky diff for
no behavioural gain. `configure()` raising when it has not been called is the
guard against action-at-a-distance.
"""

from __future__ import annotations

import zoneinfo
import json
import re
import os

from . import log

_log = log.get(__name__)
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Universal: measured, not configured
# ---------------------------------------------------------------------------

# 30-minute slots. The presence sources update far more often (a person entity
# writes every few minutes while moving), so the resample exists to put people
# on a common grid and average away GPS jitter, not to invent resolution.
GRID_MINUTES = 30
SLOTS_PER_DAY = 24 * 60 // GRID_MINUTES  # 48

# A slot needs this much of its duration observed, or it is NaN rather than a
# guess. Costs ~1% of slots.
MIN_SLOT_COVERAGE = 0.5

# Longest tracker silence that still counts as observation.
#
# MEASURED on a two-person household: the gap distribution is bimodal -- a ~5-6 h
# gap every night as phones doze (worst observed 10.3 h), and rare multi-day
# holes when recording actually stopped (one was 653 h). 12 h sits an order of
# magnitude below the outage and comfortably above the doze.
#
# Blanking the nightly doze would delete a quarter of the history including
# every night, which is when occupancy is most predictable. Not blanking a real
# outage invented three straight weeks of "home" in an early build.
MAX_SILENCE_H = 12

# Hourly out to two days. Two families are fitted over this range -- one model
# per horizon, and one pooled model with `horizon_h` as a feature -- and the
# gate picks per horizon. See train.py.
HORIZONS_H = tuple(range(1, 49))

# The subset that gets its own Home Assistant entity. The full curve rides as a
# JSON attribute, so this is about not creating 48 near-identical sensors per
# subject, not about what the model produces.
SENSOR_HORIZONS_H = (1, 2, 3, 6, 12, 24, 36, 48)

# How long the record of what was forecast is kept, for the verification chart.
#
# All 48 horizons are stored, not just SENSOR_HORIZONS_H: the point of that card
# is a slider across the whole range, and the difference is a few megabytes.
# 48 horizons x 48 slots x ~3 subjects is ~6.9k rows a day, so 30 days caps the
# table around 200k rows -- the same order as the archive it sits beside, which
# is ~2.3 MB a year and never pruned. Unlike the archive this is not training
# data: nothing refits from it, and a chart of the recent past is its only
# reader, so keeping it forever would be paying storage for nobody.
FORECAST_RETENTION_DAYS = 30

# Distance is carried forward with no time cap -- see sources/store.py and the
# note in features.numeric_on_grid. Proximity only fires when the distance
# CHANGES, so its silence means "has not moved": measured p50 gap 0.5-3 min while
# moving against a p90 of 322-1341 min while parked at home. A one-hour cap
# blanked 67-78% of the column, almost all of it people sitting at home.
DISTANCE_STALE_MIN = None

HOUSE_SLUG = "house"

# The state Home Assistant uses for "in the home zone". Universal.
HOME_STATE = "home"

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
    """One thing whose occupancy is forecast.

    `slug` is load-bearing in five places -- the subject key, the one-hot ML
    category, the `other_*` column names, the MQTT topic segment and the ETA
    model filename -- so it is derived once, de-duplicated, and never allowed to
    collide with HOUSE_SLUG.
    """
    slug: str
    entity_id: str          # person.* or the house group
    is_person: bool = True
    distance_entity: str | None = None  # sensor.*_distance, optional
    direction_entity: str | None = None # sensor.*_direction_of_travel, optional
    next_alarm_entity: str | None = None # sensor.*_next_alarm, optional


@dataclass(frozen=True)
class Zone:
    """One place the user ticked, and nothing more.

    Deliberately roleless. The previous design hung a single "office zone" off
    each person, which could not express a second workplace nor a place nobody
    works -- a supermarket. A zone is now just a place; which of them mean
    "work" for which person is the model's problem, learned from the columns
    rather than declared here.

    `name` is the zone's friendly name, snapshotted from the live entity by
    `runtime.refresh_environment`. It is here because Home Assistant writes a
    zone's NAME into a person's state, so the name is the only per-person zone
    signal that exists in history -- see `features._resolve_zone_events`, which
    is the one place it is read. Snapshotted rather than hardcoded so a rename
    is picked up on the next boot.
    """
    slug: str          # slugify(entity_id): zone.alice_office -> alice_office
    entity_id: str
    name: str


# Defaults for the crossing cuts, quoted by the API's error messages and by the
# panel's copy so that three places cannot drift.
#
# `crossing_min_hours` is 2 rather than 1 because 1 is the old bug written down:
# the forecast grid is hourly and its target is the fraction of a 30-minute slot
# spent home, so an absence shorter than about an hour cannot be represented at
# all -- a walk round the block reads as `home_frac` near 0.5 in two adjacent
# slots and never becomes a departure. A single hour past the line is therefore
# noise by construction, not a short trip, and there is no real signal for this
# to suppress. `eta.MIN_JOURNEY_KM` draws the same line in distance.
DEFAULT_DEPARTURE_THRESHOLD = 0.5
DEFAULT_ARRIVAL_THRESHOLD = 0.5
DEFAULT_CROSSING_MIN_HOURS = 2


@dataclass
class Settings:
    """Everything that differs between installations. Persisted to /data/config.json.

    `country` and `holiday_country` look redundant and are not. `country` is
    Home Assistant's, re-read on every boot, and says where the house *is*.
    `holiday_country` is the user's, and says which calendar the household
    actually *keeps* -- an Indian family living in NL may well be at home on
    Diwali and at work on Koningsdag. Three states, and the difference between
    the last two is what lets the user's pick survive a restart:

        None  never chosen. Fall back to `country`, which is exactly how this
              behaved before the setting existed, so every config.json written
              by an older build keeps working without a migration.
        ""    chosen, and the choice is "no holidays at all".
        "IN"  chosen.

    The three crossing cuts are a pair of thresholds and a dwell, and they are
    settings rather than constants because the cost of a wrong answer is not
    symmetric and is not knowable from here -- see the module docstring. They
    reduce the 48 h curve to the two "hours until" sensors and touch nothing
    else: the curve is identical whatever they are set to. Read them together as
    a hysteresis band, `departure_threshold <= arrival_threshold`, so that a
    forecast between the two counts as neither leaving nor arriving.
    """
    people: list[str] = field(default_factory=list)      # person.* entity ids
    zones: list[str] = field(default_factory=list)       # zone.* entity ids, enabled
    # zone entity -> friendly name, refreshed from the live entities on every
    # boot by runtime.refresh_environment. The feature build never talks to Home
    # Assistant, so it cannot look these up for itself.
    zone_names: dict[str, str] = field(default_factory=dict)
    house_entity: str | None = None                      # group.* or None -> OR over people
    proximity: dict[str, list[str]] = field(default_factory=dict)  # person -> [distance, direction]
    # person -> sensor.*_next_alarm. None/{} carry the same distinction as
    # `holiday_country`: None is "never looked", {} is "looked, and there are
    # none" or "the user cleared them". Only None is re-discovered.
    next_alarm: dict[str, str] | None = None
    timezone: str = "UTC"
    country: str | None = None                           # Home Assistant's. Never the user's.
    holiday_country: str | None = None                   # The user's. See the docstring.
    units: dict[str, str] = field(default_factory=dict)  # entity id -> unit_of_measurement
    home_latitude: float | None = None
    home_longitude: float | None = None
    source: str = "store"                                # "store" | "influx"
    # How the curve is reduced to "hours until away/home". See the docstring.
    # Optional, and decoration only: a `schedule.*` entity the household already
    # keeps for its waking hours, used to shade the night on the forecast chart.
    # It touches no feature, no model and no published entity -- unset, the
    # chart simply has no grey bands.
    day_schedule: str | None = None
    departure_threshold: float = DEFAULT_DEPARTURE_THRESHOLD
    arrival_threshold: float = DEFAULT_ARRIVAL_THRESHOLD
    crossing_min_hours: int = DEFAULT_CROSSING_MIN_HOURS

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Settings":
        raw = cls._migrate(json.loads(text))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def _migrate(raw: dict) -> dict:
        """Bring an older config.json forward. Runs BEFORE the field filter.

        `office_zones` was `{person entity: zone entity}` -- one workplace each,
        with the role baked in. Its values are exactly the zones that household
        cared about, so they become the enabled list and nobody has to re-pick
        them. The key is dropped rather than kept: two spellings of the same
        setting is how they drift apart.
        """
        if "office_zones" in raw:
            legacy = raw.pop("office_zones") or {}
            if "zones" not in raw:
                raw["zones"] = sorted({z for z in legacy.values() if z})
        return raw

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.to_json())
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
    """`TIMEZONE` as a real tzinfo, for the code that is not pandas.

    Most of this package converts through pandas, which takes the string
    directly. `night.py` walks a 48-hour grid with stdlib datetimes and needs
    the object; falls back to UTC rather than raising, for the same reason
    `features` does -- a bad timezone from Home Assistant is a reason to be
    wrong by an hour, not a reason to refuse to start.
    """
    try:
        return zoneinfo.ZoneInfo(TIMEZONE)
    except Exception:  # noqa: BLE001
        return zoneinfo.ZoneInfo("UTC")
HOLIDAY_COUNTRY: str | None = None
HOME_COORDS: tuple[float, float] | None = None
DAY_SCHEDULE: str | None = None
DEPARTURE_THRESHOLD: float = DEFAULT_DEPARTURE_THRESHOLD
ARRIVAL_THRESHOLD: float = DEFAULT_ARRIVAL_THRESHOLD
CROSSING_MIN_HOURS: int = DEFAULT_CROSSING_MIN_HOURS


def _crossing_cut(value, default: float | int, low: float, high: float):
    """One crossing cut, clamped to something servable. Never raises.

    Unlike the `no people` ValueError below, a nonsense cut must not stop the
    add-on booting. `/data/config.json` can be hand-edited on the box, and that
    path reaches `configure()` without passing the API's validation at all -- so
    a publisher whose job is to keep publishing degrades to the nearest sane
    value and says so, rather than bricking itself. The API rejects the same
    value loudly, which is where a person actually finds out.
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
        # A person whose entity id slugifies to "house", or two people who
        # collide, would otherwise produce duplicate (subject, time) keys and
        # blow up much later inside a pandas reindex.
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

    # Same de-dup as the subjects, and for the same reason: two zones that
    # slugify alike would mint one column between them and silently merge two
    # places into one feature.
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
    # Clamped, never raised. A cut of exactly 0 or 1 can never be met by a
    # rounded curve and would leave the sensor permanently unknown, so the
    # bounds are open at both ends.
    DAY_SCHEDULE = (settings.day_schedule or None) if settings else None
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
    """Lowercased friendly name -> zone entity id, for the enabled zones.

    The lookup table behind the one friendly-name match this package makes. A
    zone with no snapshotted name contributes nothing rather than a ""-keyed
    entry that would swallow every blank state.
    """
    return {z.name.strip().lower(): z.entity_id for z in ZONES if z.name.strip()}


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
# Supervisor injects the broker's details when the add-on declares
# `services: ["mqtt:need"]`, and run.sh exports them. There is deliberately NO
# host default: the original defaulted to a specific LAN address, so an unset
# variable silently dialled somebody else's broker instead of failing.

def mqtt_settings() -> dict:
    host = os.environ.get("MQTT_HOST")
    if not host:
        raise RuntimeError(
            "no MQTT broker: expected MQTT_HOST from Supervisor's mqtt service. "
            "Install the Mosquitto add-on, or set it in the add-on options.")
    return {
        "host": host,
        "port": int(os.environ.get("MQTT_PORT", "1883")),
        "username": os.environ.get("MQTT_USER") or None,
        "password": os.environ.get("MQTT_PASSWORD") or None,
    }


# MQTT topic root, and the MQTT client id derived from it in predict.py.
#
# DERIVED FROM THE ADD-ON'S OWN SLUG, so the stable and edge builds separate
# themselves and cannot be made to collide by editing one and forgetting the
# other. Supervisor prefixes the slug with its repository (`local_occupancy_forecast`,
# `a1b2c3d4_occupancy_forecast_edge`), and that first token is dropped -- otherwise
# moving an add-on from the local folder to the published repository would
# silently rename every entity it owns.
#
# Two of these MUST NOT share a client id: MQTT requires the broker to hand a
# duplicated id to whoever connected last and silently disconnect the previous
# holder, so they would fight forever with no error surfacing anywhere. That is
# exactly what running stable and edge side by side would do.
DEFAULT_TOPIC_PREFIX = "occupancy_forecast"

_topic_prefix: str | None = None


def topic_prefix() -> str:
    global _topic_prefix
    if _topic_prefix is not None:
        return _topic_prefix

    _topic_prefix = DEFAULT_TOPIC_PREFIX
    token = os.environ.get("SUPERVISOR_TOKEN")
    if token:
        # Only attempted inside an add-on; outside one there is no supervisor to
        # ask and no second instance to collide with.
        try:
            import json
            import urllib.request
            request = urllib.request.Request(
                "http://supervisor/addons/self/info",
                headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(request, timeout=15) as response:
                slug = json.load(response)["data"]["slug"]
            _, _, name = slug.partition("_")     # drop the repository prefix
            if name:
                _topic_prefix = name
        except Exception as err:  # noqa: BLE001
            # For the stable add-on the default IS the right answer. For any
            # second instance it is actively wrong -- it silently adopts
            # stable's topic root and client id, and two MQTT clients sharing
            # an id take turns kicking each other off with nothing in the log
            # to say why. Never silent, therefore -- and ERROR rather than
            # warning, because it is silent data loss and not a degradation.
            _log.error("could not read this add-on's slug from Supervisor "
                       "(%s); falling back to the topic prefix %r. If more "
                       "than one build of this add-on is installed, they will "
                       "now COLLIDE on MQTT.", err, DEFAULT_TOPIC_PREFIX)
    return _topic_prefix


def display_name() -> str:
    """This add-on's name for human eyes: `occupancy_forecast_edge` -> "Occupancy Forecast Edge".

    Anywhere a message is addressed to the user -- a notification title, a log
    line -- it has to say WHICH build wrote it, or two add-ons produce identical
    text and the reader cannot tell which one is complaining. Derived from the
    same slug as everything else so there is one thing to get right, not two.
    """
    return topic_prefix().replace("_", " ").title()
