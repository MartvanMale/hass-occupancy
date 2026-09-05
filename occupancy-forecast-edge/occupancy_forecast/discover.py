"""Work out what this Home Assistant actually has, and propose a configuration.

The original named two people and six sensors as literals. This asks.

Everything optional degrades to an all-NaN feature column, which
HistGradientBoosting handles natively and the ship gate prices -- so a house
with one person, no work zones and no Proximity integration still trains and
still serves. It just gets less skill, and the status page says so.

**Synthesised proximity is the one that is worth the trouble.** Distance-to-home
was the single biggest feature win (+17% Brier at 1 h), and it comes from the
Proximity integration, which many installs will not have. It can be computed
instead from `person.*` GPS attributes against the home coordinates -- verified
to within 4-10 m of the real sensors.

The catch, and it is a real one: **synthesised distance can only be recorded
going forward.** A person moving across town does not change their *state*, so
the recorder holds no intermediate positions to backfill from -- their history
is a handful of zone transitions with attributes attached. Installs with
Proximity get months of distance history immediately; installs without it start
accumulating from the moment the add-on is installed.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from .config import Settings, slugify

# Proximity's entity ids look like sensor.<zone>_<person>_distance. Rather than
# parse that, match on the suffix and on the person's slug appearing in the id.
DISTANCE_SUFFIX = "_distance"
DIRECTION_SUFFIX = "_direction_of_travel"

# The companion app's own naming, which is stable across platforms.
NEXT_ALARM_SUFFIX = "_next_alarm"


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    return 2 * 6_371_000 * asin(sqrt(
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2))


def zone_names(states: list[dict] | None, wanted: list[str]) -> dict[str, str]:
    """Zone entity -> friendly name, for the enabled zones.

    A zone that has vanished from Home Assistant keeps whatever name was last
    snapshotted rather than dropping out: its history is still full of that
    name, and forgetting it would strand those rows in `zone_other`.
    """
    if not states:
        return {}
    by_id = {s["entity_id"]: s for s in states}
    found = {}
    for entity_id in wanted:
        state = by_id.get(entity_id)
        if state is None:
            continue
        name = state.get("attributes", {}).get("friendly_name")
        if name:
            found[entity_id] = name
    return found


def candidates(states: list[dict]) -> dict:
    """Everything the user could plausibly pick, for the configuration page."""
    by_domain: dict[str, list[dict]] = {}
    for state in states:
        domain = state["entity_id"].split(".", 1)[0]
        by_domain.setdefault(domain, []).append(state)

    def named(items):
        return sorted(
            ({"entity_id": s["entity_id"],
              "name": s.get("attributes", {}).get("friendly_name") or s["entity_id"]}
             for s in items), key=lambda x: x["name"].lower())

    zones = [z for z in by_domain.get("zone", []) if z["entity_id"] != "zone.home"]
    # A person group, not a light group: its members are the thing that matters.
    groups = [g for g in by_domain.get("group", [])
              if any(str(m).startswith("person.")
                     for m in g.get("attributes", {}).get("entity_id", []) or [])]

    return {
        "people": named(by_domain.get("person", [])),
        "zones": named(zones),
        "groups": named(groups),
        # Optional, and only ever decoration: a schedule the household already
        # keeps for "when are we awake" lets the forecast chart shade the night.
        # Nothing about the model reads it.
        "schedules": named(by_domain.get("schedule", [])),
        "has_proximity": bool(_proximity_sensors(states)),
        "countries": holiday_countries(),
    }


# "and"/"of" inside a country name should not be capitalised by .title().
_LOWER_WORDS = {"and", "of", "the"}


def holiday_countries() -> list[dict]:
    """Every calendar the `holidays` package can supply, for the panel's picker.

    Read from the registry rather than `list_supported_countries()` because the
    registry carries a name alongside each code, and because the supported list
    also contains alias codes (`UK` for `GB`) that would show up as a duplicate
    entry the user cannot tell apart.

    An empty list is a usable answer: the panel then says the picker is
    unavailable, which beats a 500 on the status page over a cosmetic list.
    """
    try:
        from holidays.registry import COUNTRIES
    except ImportError:
        return []

    out = []
    for key, entry in COUNTRIES.items():
        words = key.replace("_", " ").split()
        name = " ".join(w.title() if i == 0 or w not in _LOWER_WORDS else w
                        for i, w in enumerate(words))
        out.append({"code": entry[1], "name": name})
    return sorted(out, key=lambda c: c["name"])


def is_supported_country(code: str) -> bool:
    """Whether `holidays` covers this code. Unknown-because-uninstalled is True.

    The check exists to give the user a 400 instead of a silently flat
    `is_holiday` column. If the library cannot be imported at all we cannot
    honestly say the code is wrong, so we let it through -- `_holiday_flags`
    already degrades to zeros rather than crashing the feature build.
    """
    try:
        from holidays.utils import list_supported_countries
    except ImportError:
        return True
    return code in list_supported_countries()


def _proximity_sensors(states: list[dict]) -> dict[str, str]:
    out = {}
    for state in states:
        entity_id = state["entity_id"]
        if not entity_id.startswith("sensor."):
            continue
        if entity_id.endswith(DISTANCE_SUFFIX) or entity_id.endswith(DIRECTION_SUFFIX):
            out[entity_id] = state.get("attributes", {}).get("unit_of_measurement", "")
    return out


def match_proximity(person_entity: str, states: list[dict]) -> list[str | None]:
    """Find this person's Proximity pair, or [None, None].

    Matched on the person's slug appearing in the sensor id, which is how the
    Proximity integration names them (`sensor.home_alice_distance`). A miss is
    not a problem -- it just means the distance gets synthesised.
    """
    slug = slugify(person_entity)
    sensors = _proximity_sensors(states)
    distance = next((e for e in sensors
                     if e.endswith(DISTANCE_SUFFIX) and slug in e), None)
    direction = next((e for e in sensors
                      if e.endswith(DIRECTION_SUFFIX) and slug in e), None)
    return [distance, direction]


def match_next_alarm(person_entity: str, states: list[dict]) -> str | None:
    """Find this person's companion-app next-alarm sensor, or None.

    Matched on the person's slug appearing in the sensor id, the same way
    `match_proximity` works -- the companion app names its sensors after the
    device (`sensor.alices_phone_next_alarm`), which usually carries the
    person's name but is not guaranteed to. A miss is not a problem: the column
    goes NaN and the ship gate prices it at nothing.

    Nothing is served off this yet -- see `features.BUILT_NOT_SHIPPED`. It is
    discovered and collected now so that the history exists later.
    """
    slug = slugify(person_entity)
    return next((s["entity_id"] for s in states
                 if s["entity_id"].startswith("sensor.")
                 and s["entity_id"].endswith(NEXT_ALARM_SUFFIX)
                 and slug in s["entity_id"]), None)


def units_for(states: list[dict], entity_ids: list[str]) -> dict[str, str]:
    """entity_id -> unit_of_measurement.

    Needed by the Influx source, which addresses numeric sensors by their UNIT
    rather than their entity id. Hardcoding this to "m" is why the original
    silently produced an all-NaN distance column on an imperial install.
    """
    wanted = set(entity_ids)
    return {s["entity_id"]: s["attributes"]["unit_of_measurement"]
            for s in states
            if s["entity_id"] in wanted
            and s.get("attributes", {}).get("unit_of_measurement")}


def propose(ha) -> Settings:
    """A configuration that would work, for the user to confirm or edit."""
    config = ha.config()
    states = ha.states()

    people = [s["entity_id"] for s in states if s["entity_id"].startswith("person.")]
    groups = candidates(states)["groups"]

    proximity, next_alarm = {}, {}
    for person in people:
        pair = match_proximity(person, states)
        if any(pair):
            proximity[person] = pair
        alarm = match_next_alarm(person, states)
        if alarm:
            next_alarm[person] = alarm

    numeric = [p[0] for p in proximity.values() if p[0]]

    return Settings(
        people=sorted(people),
        zones=[],                     # cannot be guessed; the user ticks
        house_entity=groups[0]["entity_id"] if groups else None,
        proximity=proximity,
        next_alarm=next_alarm,
        timezone=config.get("time_zone") or "UTC",
        country=config.get("country"),
        # Seeded from HA, then the user's to change. See Settings' docstring.
        holiday_country=config.get("country"),
        units=units_for(states, numeric),
        home_latitude=config.get("latitude"),
        home_longitude=config.get("longitude"),
    )


def synthetic_distance_entity(slug: str) -> str:
    """Where a synthesised distance is stored, so it cannot collide with a real one."""
    return f"occupancy_ml.{slug}_distance"


def sample_distances(ha, settings: Settings) -> list[tuple[str, int, str]]:
    """Current distance-to-home per person, as store rows.

    Called on every collection pass. Only produces rows for people who have no
    real Proximity sensor -- there is no point shadowing a better source -- and
    only when the person entity is currently reporting GPS.
    """
    import datetime as dt

    if settings.home_latitude is None:
        return []
    home = (settings.home_latitude, settings.home_longitude)
    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    rows: list[tuple[str, int, str]] = []
    states = {s["entity_id"]: s for s in ha.states()}
    for person in settings.people:
        if (settings.proximity.get(person) or [None])[0]:
            continue
        attrs = (states.get(person) or {}).get("attributes", {})
        lat, lon = attrs.get("latitude"), attrs.get("longitude")
        if lat is None or lon is None:
            continue
        entity = synthetic_distance_entity(slugify(person))
        rows.append((entity, now_ms, f"{haversine_m(home, (lat, lon)):.1f}"))
    return rows
