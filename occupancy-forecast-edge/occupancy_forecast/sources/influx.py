"""InfluxDB v2 as a history source, read-only; worth it only if you have one.

HA names a measurement after the entity id, except a sensor with a unit, which
lands under its UNIT with the object_id in an `entity_id` tag. So the caller
passes `units` from HA's state; hardcoding "m" blanked imperial distances.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import urllib.request


class InfluxSource:
    def __init__(self, url: str, token: str, org: str,
                 bucket: str = "homeassistant",
                 units: dict[str, str] | None = None, timeout: int = 300):
        self.url = url.rstrip("/")
        self.token = token
        self.org = org
        self.bucket = bucket
        self.units = units or {}
        self.timeout = timeout

    # -- transport ----------------------------------------------------------

    def _query(self, flux: str, timed: bool = True) -> list[dict]:
        """`timed=False` for an aggregate: count() returns no `_time` column at
        all, so the usual filter would discard every row of it.
        """
        request = urllib.request.Request(
            f"{self.url}/api/v2/query?org={self.org}",
            data=flux.encode(),
            headers={"Authorization": f"Token {self.token}",
                     "Content-Type": "application/vnd.flux",
                     "Accept": "application/csv"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode()
        rows = []
        for row in csv.DictReader(io.StringIO(body)):
            # A `union` of two schemas repeats the header mid-body, and
            # DictReader hands that back as data.
            if row.get("result") == "result" or row.get("_time") == "_time":
                continue
            if timed and not row.get("_time"):
                continue
            rows.append(row)
        return rows

    @staticmethod
    def _window(stop: str | None) -> str:
        # Always pass an explicit stop when given one: Influx's implicit stop is
        # now(), which silently truncates anything stamped in the future.
        return f", stop: {stop}" if stop else ""

    def first_seen(self, entity_ids: list[str]) -> str | None:
        """Earliest timestamp across these entities, or None if there is nothing.

        Guessing too far back builds a mostly-NaN frame, which OOMs a small box.
        """
        selectors = " or ".join(
            f'r._measurement == "{e}"' for e in entity_ids if e)
        if not selectors:
            return None
        flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: 0)
  |> filter(fn: (r) => {selectors})
  |> first()
  |> keep(columns: ["_time"])
'''
        try:
            rows = self._query(flux)
        except Exception:  # noqa: BLE001
            return None
        times = sorted(r["_time"] for r in rows if r.get("_time"))
        return times[0] if times else None

    # -- Source -------------------------------------------------------------

    def states(self, entity_id: str, start: str,
               stop: str | None = None) -> list[tuple[str, str]]:
        flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: {start}{self._window(stop)})
  |> filter(fn: (r) => r._measurement == "{entity_id}" and r._field == "state")
  |> keep(columns: ["_time", "_value"])
  |> sort(columns: ["_time"])
'''
        return [(r["_time"], r["_value"]) for r in self._query(flux) if r.get("_value")]

    def seeded_states(self, entity_id: str, start: str, stop: str | None = None,
                      seed_days: int = 14) -> list[tuple[str, str]]:
        rows = self.states(entity_id, start, stop)
        begin = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        seed_start = (begin - dt.timedelta(days=seed_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        seed = self.states(entity_id, seed_start, start)
        if seed:
            rows = [(begin.strftime("%Y-%m-%dT%H:%M:%SZ"), seed[-1][1]), *rows]
        return rows

    def numeric(self, entity_id: str, start: str,
                stop: str | None = None) -> list[tuple[str, float]]:
        unit = self.units.get(entity_id)
        if unit:
            # Measurement named after the unit, object_id in a tag.
            object_id = entity_id.split(".", 1)[-1]
            selector = (f'r._measurement == "{unit}" and r.entity_id == "{object_id}" '
                        f'and r._field == "value"')
        else:
            # No unit: the entity gets its own measurement.
            selector = f'r._measurement == "{entity_id}" and r._field == "value"'

        flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: {start}{self._window(stop)})
  |> filter(fn: (r) => {selector})
  |> keep(columns: ["_time", "_value"])
  |> sort(columns: ["_time"])
'''
        out: list[tuple[str, float]] = []
        for row in self._query(flux):
            try:
                out.append((row["_time"], float(row["_value"])))
            except (TypeError, ValueError):
                continue  # 'unavailable' during a restart is absence, not an error
        return out

    def seeded_numeric(self, entity_id: str, start: str, stop: str | None = None,
                       seed_days: int = 14) -> list[tuple[str, float]]:
        """`numeric`, seeded the way `seeded_states` seeds the string path.

        `DISTANCE_STALE_MIN` is None, so a reading carries forward for as long
        as it takes. Without the seed an entity that changes twice a week draws
        nothing at all rather than the flat line it actually held.
        """
        rows = self.numeric(entity_id, start, stop)
        begin = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        seed_start = (begin - dt.timedelta(days=seed_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        seed = self.numeric(entity_id, seed_start, start)
        if seed:
            rows = [(begin.strftime("%Y-%m-%dT%H:%M:%SZ"), seed[-1][1]), *rows]
        return rows

    # -- inventory, for the Data tab ----------------------------------------

    def _tags(self, entity_ids: list[str]) -> str:
        """Select these entities by their `domain`/`entity_id` TAGS.

        Not by `_measurement`: a sensor with a unit is stored under the unit,
        so a measurement filter misses it. Both tags are on every point.
        """
        pairs = []
        for entity_id in entity_ids:
            domain, _, object_id = entity_id.partition(".")
            if domain and object_id:
                pairs.append(f'(r.domain == "{domain}" and r.entity_id == "{object_id}")')
        return " or ".join(pairs)

    def archive(self, entity_ids: list[str]) -> dict:
        """What Influx holds for these entities: the Data tab's archive card.

        Two queries for the whole set, not two per entity -- the bucket holds
        every entity in the house, so this is only ever asked about the
        configured ones. `field` says which shape the reader should ask for.
        """
        selector = self._tags(entity_ids)
        if not selector:
            return {"span": _empty_span(), "entities": []}
        window = (f'  |> range(start: 0)\n'
                  f'  |> filter(fn: (r) => ({selector}) and '
                  f'(r._field == "state" or r._field == "value"))\n')

        counts: dict[tuple[str, str], int] = {}
        flux = (f'from(bucket: "{self.bucket}")\n{window}'
                '  |> group(columns: ["domain", "entity_id", "_field"])\n'
                '  |> count()\n')
        for row in self._query(flux, timed=False):
            key = (f'{row.get("domain")}.{row.get("entity_id")}', row.get("_field") or "")
            try:
                counts[key] = int(row["_value"])
            except (TypeError, ValueError, KeyError):
                continue

        bounds: dict[tuple[str, str], dict] = {}
        flux = (f'base = from(bucket: "{self.bucket}")\n{window}'
                '  |> group(columns: ["domain", "entity_id", "_field"])\n'
                '  |> keep(columns: ["domain", "entity_id", "_field", "_time", "_value"])\n'
                'union(tables: [base |> first(), base |> last()])\n')
        for row in self._query(flux):
            key = (f'{row.get("domain")}.{row.get("entity_id")}', row.get("_field") or "")
            seen = bounds.setdefault(key, {"first": None, "last": None, "sample": []})
            when = row["_time"]
            if seen["first"] is None or when < seen["first"]:
                seen["first"] = when
            if seen["last"] is None or when > seen["last"]:
                seen["last"] = when
            if row.get("_value") is not None:
                seen["sample"].append(row["_value"])

        entities = []
        for entity_id in entity_ids:
            # `state` first: HA writes BOTH for a person -- the string and its
            # numeric twin -- so summing the pair would double the archive, and
            # the numeric twin would classify presence as a numeric series.
            field = next((f for f in ("state", "value")
                          if counts.get((entity_id, f))), None)
            edge = bounds.get((entity_id, field or ""), {})
            entities.append({
                "entity_id": entity_id,
                "rows": counts.get((entity_id, field or ""), 0),
                "first": edge.get("first"),
                "last": edge.get("last"),
                "field": field,
                "sample": edge.get("sample") or [],
            })
        return {"span": _span_of(entities), "entities": entities}


def _empty_span() -> dict:
    return {"first": None, "last": None, "rows": 0, "days": 0.0, "bytes": None}


def _span_of(entities: list[dict]) -> dict:
    """`bytes` is None, not 0: the bucket is shared with the rest of the house,
    so there is no honest "on disk" figure for this add-on's share of it."""
    import datetime as dt

    firsts = [e["first"] for e in entities if e["first"]]
    lasts = [e["last"] for e in entities if e["last"]]
    first, last = (min(firsts) if firsts else None), (max(lasts) if lasts else None)
    days = 0.0
    if first and last:
        began = dt.datetime.fromisoformat(first.replace("Z", "+00:00"))
        ended = dt.datetime.fromisoformat(last.replace("Z", "+00:00"))
        days = round((ended - began).total_seconds() / 86_400, 1)
    return {"first": first, "last": last,
            "rows": sum(e["rows"] for e in entities),
            "days": days, "bytes": None}
