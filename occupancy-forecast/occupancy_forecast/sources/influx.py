"""InfluxDB v2 as a history source. Optional, and only worth it if you already have one.

The default source is the local store; this exists for installs that have been
archiving Home Assistant to Influx for months and would otherwise throw that
history away and start over from the ten days it takes to train at all.

**The awkward part is that Home Assistant writes three different shapes**, and
which one an entity lands in depends on its domain and whether it has a unit:

  person.alice                         measurement `person.alice`, field `state`  (string)
  zone.alice_office                    measurement `zone.alice_office`, field `value` (number)
  sensor.home_alice_distance           measurement `m` -- THE UNIT -- with the
                                       object_id in an `entity_id` tag, field `value`
  sensor.home_..._direction_of_travel  measurement `sensor.home_...`, field `state`

The first, second and fourth are addressable from the entity id alone. The third
is not: you have to know the unit. So the caller passes `units`, resolved from
Home Assistant's own current state.

That indirection also fixes a real bug in an earlier version: the unit was hardcoded
to `"m"`, so on an imperial install (where HA writes `"mi"`) the distance
columns silently went all-NaN and the highest-value feature group vanished
without a word.

Read-only. `homeassistant` is Home Assistant's source of truth and nothing here
writes to it.
"""

from __future__ import annotations

import csv
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

    def _query(self, flux: str) -> list[dict]:
        request = urllib.request.Request(
            f"{self.url}/api/v2/query?org={self.org}",
            data=flux.encode(),
            headers={"Authorization": f"Token {self.token}",
                     "Content-Type": "application/vnd.flux",
                     "Accept": "application/csv"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode()
        return [row for row in csv.DictReader(io.StringIO(body)) if row.get("_time")]

    @staticmethod
    def _window(stop: str | None) -> str:
        # Always pass an explicit stop when given one: Influx's implicit stop is
        # now(), which silently truncates anything stamped in the future.
        return f", stop: {stop}" if stop else ""

    def first_seen(self, entity_ids: list[str]) -> str | None:
        """Earliest timestamp across these entities, or None if there is nothing.

        Without this the feature builder has no idea how far back an Influx goes
        and has to guess. Guessing high is expensive, not merely wasteful: the
        table is one row per subject per 30-minute slot across ~700 columns, so
        a two-year guess against six months of data is a ~600 MB frame of mostly
        NaN, which on a small box is the difference between training and being
        killed by the OOM reaper.
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
        import datetime as dt
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
            # Shape 3: measurement named after the unit, object_id in a tag.
            object_id = entity_id.split(".", 1)[-1]
            selector = (f'r._measurement == "{unit}" and r.entity_id == "{object_id}" '
                        f'and r._field == "value"')
        else:
            # Shapes 1/2: the entity gets its own measurement.
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
