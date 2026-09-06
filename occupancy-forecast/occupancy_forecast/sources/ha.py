"""Talking to Home Assistant, and filling the store from it.

Two things live here:

  `HomeAssistant`  a thin REST client. Inside an add-on it goes through the
                   Supervisor proxy at http://supervisor/core/api with
                   SUPERVISOR_TOKEN, which Supervisor injects -- no token for
                   the user to create, no host to configure. Outside one (tests,
                   development on a laptop) it falls back to HA_URL + HA_TOKEN.

  `StoreSource`    a `Source` backed by `HistoryStore`, plus the `collect()`
                   that keeps the store fed.

Deliberately stdlib-only (urllib + json). The add-on already carries pandas and
scikit-learn; adding aiohttp to make four requests a minute would be silly, and
stdlib means `sources` imports cleanly in a bare test environment.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from .store import HistoryStore, _ms

SUPERVISOR_API = "http://supervisor/core/api"

# The collector's own pulse, written on every successful pass.
#
# This exists because "nothing was recorded" and "nothing changed" are the same
# thing in Home Assistant's history, and telling them apart is the whole job of
# the observability mask. On Influx -- which stores every write, not just
# changes -- a long silence really does mean the recorder stopped. Here it
# usually means everyone was asleep: MEASURED, the person entities' gaps have a
# p95 of 9.2 h and a max of 15.4 h with nothing wrong at all. Judging those by
# the same 12 h threshold blanked 11% of the timeline as a fake outage.
#
# A heartbeat is not a heuristic: if it is there, the add-on was running and
# Home Assistant was answering.
HEARTBEAT_ENTITY = "occupancy_ml.collector"

# How far back to reach on the very first collection. Recorder will usually have
# far less than this -- 10 days is the stock default -- but asking for more
# costs nothing and an install that has been recording for months should get all
# of it. Measured: a 100-day request returned in 0.8 s and 0.2 MB.
BOOTSTRAP_DAYS = 400

# Re-fetch this much on every poll. Writes are idempotent (primary key on
# entity_id + ts), so overlapping is free and it means a missed poll, a restart
# or a clock skew heals itself instead of leaving a hole.
OVERLAP_MINUTES = 90

# What gets stored for an entity whose absence is itself a reading. Its own
# word rather than `unavailable` or `unknown`, because Home Assistant uses both
# for this and a reader should not have to handle two spellings of one fact.
ABSENT = "absent"


class HomeAssistant:
    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: int = 120):
        supervisor = os.environ.get("SUPERVISOR_TOKEN")
        if base_url is None and supervisor:
            base_url, token = SUPERVISOR_API, supervisor
        self.base_url = (base_url or os.environ.get("HA_URL", "")).rstrip("/")
        self.token = token or os.environ.get("HA_TOKEN", "")
        self.timeout = timeout
        if not self.base_url:
            raise RuntimeError(
                "no Home Assistant to talk to: expected SUPERVISOR_TOKEN (inside an "
                "add-on) or HA_URL + HA_TOKEN (outside one)")

    def _get(self, path: str) -> object:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    def _post(self, path: str, payload: dict) -> object:
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode()
        return json.loads(body) if body.strip() else None

    # -- reads --------------------------------------------------------------

    def config(self) -> dict:
        """`/api/config` -- the source of truth for timezone, country and units.

        All three used to be module constants, and two of them crashed the
        feature build outright when wrong (`tz_convert` on a bad zone,
        `holidays.country_holidays` on an unsupported country).
        """
        return self._get("/config")

    def states(self) -> list[dict]:
        return self._get("/states")

    def history(self, entity_ids: list[str], start: str,
                stop: str | None = None) -> list[list[dict]]:
        """`/api/history/period` with `minimal_response`.

        `minimal_response` collapses each series to its state CHANGES -- which
        is exactly the step function the feature builder wants, and a fraction
        of the payload. Only the first entry of each series carries attributes.

        **`end_time` is always sent, and that is not tidiness.** Without it Home
        Assistant returns ONE DAY from `start` and says nothing about having
        done so -- no error, no flag, just a short series that looks complete.
        Asking for a week and silently getting the first day of it is the kind
        of wrong that survives review: `night.py` recovered a weekly pattern
        from what it thought was seven days and was really sixteen hours, so six
        weekdays had no evidence and the chart shaded one night out of two. The
        same call bootstraps the archive on a fresh install, where the failure
        would have been a quietly truncated backfill nobody could recover later.
        """
        if stop is None:
            stop = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = {"filter_entity_id": ",".join(entity_ids), "minimal_response": "",
                 "end_time": stop}
        return self._get(f"/history/period/{start}?{urllib.parse.urlencode(query)}")

    # -- writes -------------------------------------------------------------

    def notify(self, title: str, message: str, notification_id: str) -> None:
        """Raise a persistent notification.

        An add-on cannot create a repair issue -- `issue_registry` is Core-only
        -- so this is the closest it can reach for "you should know something".
        Re-using the same `notification_id` replaces rather than stacks.
        """
        self._post("/services/persistent_notification/create",
                   {"title": title, "message": message,
                    "notification_id": notification_id})

    def dismiss(self, notification_id: str) -> None:
        try:
            self._post("/services/persistent_notification/dismiss",
                       {"notification_id": notification_id})
        except urllib.error.HTTPError:
            pass  # never raised, or already gone


class StoreSource:
    """A `Source` reading the local store, with `collect()` to keep it fed."""

    def __init__(self, store: HistoryStore, ha: HomeAssistant):
        self.store = store
        self.ha = ha

    # -- Source -------------------------------------------------------------

    def states(self, entity_id, start, stop=None):
        return self.store.states(entity_id, start, stop)

    def seeded_states(self, entity_id, start, stop=None, seed_days=14):
        return self.store.seeded_states(entity_id, start, stop, seed_days)

    def numeric(self, entity_id, start, stop=None):
        return self.store.numeric(entity_id, start, stop)

    # -- collection ---------------------------------------------------------

    def collect(self, entity_ids: list[str],
                absence_is_a_reading: Iterable[str] = ()) -> dict:
        """Pull everything new for `entity_ids` from HA into the store.

        The window starts at the oldest per-entity watermark minus an overlap,
        so an entity added to the config later gets backfilled with whatever
        recorder still holds rather than starting from now.

        `absence_is_a_reading` names the entities for which `unavailable` and
        `unknown` are DATA rather than a gap, and are therefore stored instead
        of dropped. There is exactly one shape of sensor like that so far: a
        next-alarm sensor reads `unavailable` precisely when no alarm is set,
        which is the more common state and at least as informative as a time.
        Dropping it would leave an archive that says nothing at all on the days
        somebody had no alarm -- indistinguishable from the days the sensor was
        broken, and unrecoverable later, because Home Assistant's recorder will
        long since have discarded the difference.
        """
        if not entity_ids:
            return {"added": 0, "entities": 0}

        now = dt.datetime.now(dt.timezone.utc)
        earliest = None
        for entity_id in entity_ids:
            seen = self.store.last_seen(entity_id)
            begin = (dt.datetime.fromtimestamp(seen / 1000, dt.timezone.utc)
                     - dt.timedelta(minutes=OVERLAP_MINUTES)) if seen else (
                     now - dt.timedelta(days=BOOTSTRAP_DAYS))
            earliest = begin if earliest is None else min(earliest, begin)

        series = self.ha.history(
            entity_ids, earliest.strftime("%Y-%m-%dT%H:%M:%SZ"),
            now.strftime("%Y-%m-%dT%H:%M:%SZ"))

        keep_absence = set(absence_is_a_reading)
        rows: list[tuple[str, int, str]] = []
        for entries in series or []:
            entity_id = None
            for entry in entries:
                # Only the first entry of a minimal_response series names itself.
                entity_id = entry.get("entity_id") or entity_id
                state = entry.get("state")
                when = entry.get("last_changed") or entry.get("last_updated")
                if not (entity_id and when):
                    continue
                if state in (None, "", "unknown", "unavailable"):
                    if entity_id not in keep_absence:
                        continue
                    # Normalised, because HA uses both words for the same thing
                    # and the reader should not have to know which one it got.
                    state = ABSENT
                rows.append((entity_id, _ms(when), str(state)))

        added = self.store.append(rows)
        self.store.append([(HEARTBEAT_ENTITY, int(now.timestamp() * 1000), "ok")])
        return {"added": added, "entities": len(series or []),
                "since": earliest.isoformat()}

    def liveness_times(self, start: str, stop: str | None = None) -> list[str]:
        """When we know history was being captured.

        Heartbeats cover everything since the add-on was installed. The window
        before that was backfilled from the recorder and has none, so the
        tracked entities' own changes stand in for it -- which is the best that
        can be said about a period nobody was watching.
        """
        return [when for when, _ in self.store.states(HEARTBEAT_ENTITY, start, stop)]
