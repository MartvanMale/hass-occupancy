"""Talking to Home Assistant, and filling the store from it.

`HomeAssistant` is a thin REST client: the Supervisor proxy inside an add-on,
HA_URL + HA_TOKEN outside one; stdlib-only, so a bare test environment imports
it. `StoreSource` is a `Source` over `HistoryStore`, plus `collect()` to feed it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from .. import config
from .store import HistoryStore, _ms

SUPERVISOR_API = "http://supervisor/core/api"

# Written every successful pass: "nothing recorded" and "nothing changed" look
# alike in HA's history, and this says the add-on ran and HA was answering.
HEARTBEAT_ENTITY = "occupancy_ml.collector"

# First collection: the recorder usually holds far less, but an install that
# has recorded for months should get all of it.
BOOTSTRAP_DAYS = 400

# ...asked for in windows this wide, newest first: a response is parsed whole,
# and a deep archive in one request would be held in memory on a Pi.
BOOTSTRAP_CHUNK_DAYS = 30

# Re-fetched every poll: writes are idempotent, so overlap is free and a missed
# poll or a restart heals itself instead of leaving a hole.
OVERLAP_MINUTES = 90

# One history request per bucket of watermark age, so a quiet entity's stale
# watermark does not drag every other entity's window out to weeks.
WINDOW_BUCKET_HOURS = 6

# An entity with NO rows is re-asked at most this often, and only for the
# stretch since the last ask rather than the whole bootstrap window.
EMPTY_RETRY_SECONDS = 3600

# Stored where absence is itself a reading: its own word, because HA uses both
# `unavailable` and `unknown` for it and a reader should not handle two.
ABSENT = "absent"

# Stored where a reading is merely MISSING, a different fact from ABSENT; a row
# rather than silence, because the row is what ENDS the preceding state.
UNKNOWN = "unknown"

# Bumped when the store needs one refill. 1 re-pulls the presence `unknown`s
# older releases dropped, while the recorder still holds them.
STORE_VERSION = 1


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
        """`/api/config`, the source of truth for timezone, country, units."""
        return self._get("/config")

    def states(self) -> list[dict]:
        return self._get("/states")

    def history(self, entity_ids: list[str], start: str,
                stop: str | None = None) -> list[list[dict]]:
        """`/api/history/period` with `minimal_response`: state CHANGES only.

        Always sends `end_time`: without it HA silently returns only ONE DAY.
        """
        if stop is None:
            stop = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = {"filter_entity_id": ",".join(entity_ids), "minimal_response": "",
                 "end_time": stop}
        return self._get(f"/history/period/{start}?{urllib.parse.urlencode(query)}")

    # -- writes -------------------------------------------------------------

    def notify(self, title: str, message: str, notification_id: str) -> None:
        """Raise a persistent notification; an add-on cannot raise a repair.

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


def _utc(when: dt.datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _rows(series: list, keep_gap: set[str], keep_absence: set[str]
          ) -> tuple[list[tuple[str, int, str]], set[str]]:
    """Flatten a `/history/period` response into store rows, and who they name."""
    rows: list[tuple[str, int, str]] = []
    named: set[str] = set()
    for entries in series or []:
        entity_id = None
        for entry in entries:
            # Only the first entry of a minimal_response series names itself.
            entity_id = entry.get("entity_id") or entity_id
            state = entry.get("state")
            when = entry.get("last_changed") or entry.get("last_updated")
            if not (entity_id and when):
                continue
            named.add(entity_id)
            if config.is_empty(state):
                # Normalised in both cases, because HA uses both words for each
                # and the reader should not have to know which it got.
                if entity_id in keep_gap:
                    state = UNKNOWN
                elif entity_id in keep_absence:
                    state = ABSENT
                else:
                    continue
            rows.append((entity_id, _ms(when), str(state)))
    return rows, named


def _windows(begin: dt.datetime, now: dt.datetime,
             chunk_days: int = BOOTSTRAP_CHUNK_DAYS):
    """`[begin, now]` as request windows, NEWEST FIRST.

    So a walk can stop: past the one purge horizon, every window is empty.
    """
    span = dt.timedelta(days=chunk_days)
    until = now
    while until > begin:
        start = max(begin, until - span)
        yield start, until
        until = start


class StoreSource:
    """A `Source` reading the local store, with `collect()` to keep it fed."""

    def __init__(self, store: HistoryStore, ha: HomeAssistant):
        self.store = store
        self.ha = ha
        # entity -> (monotonic, wall clock) of the last ask that found nothing.
        self._asked_empty: dict[str, tuple[float, dt.datetime]] = {}

    # -- Source -------------------------------------------------------------

    def states(self, entity_id, start, stop=None):
        return self.store.states(entity_id, start, stop)

    def seeded_states(self, entity_id, start, stop=None, seed_days=14):
        return self.store.seeded_states(entity_id, start, stop, seed_days)

    def numeric(self, entity_id, start, stop=None):
        return self.store.numeric(entity_id, start, stop)

    # -- collection ---------------------------------------------------------

    def collect(self, entity_ids: list[str],
                absence_is_a_reading: Iterable[str] = (),
                gap_is_a_boundary: Iterable[str] = ()) -> dict:
        """Pull everything new for `entity_ids` from HA into the store.

        `absence_is_a_reading` (next-alarm) stores `unavailable`/`unknown` as
        ABSENT, a reading; `gap_is_a_boundary` (presence) stores them as
        UNKNOWN, which ends the state, or a silent phone reads as everybody-out.
        """
        if not entity_ids:
            return {"added": 0, "entities": 0}

        now = dt.datetime.now(dt.timezone.utc)
        mono = time.monotonic()
        keep_gap = set(gap_is_a_boundary)
        # One-shot, keyed in the store rather than per process: a flag on the
        # instance would re-pull the whole bootstrap window on every restart.
        refill = (keep_gap if self.store.user_version() < STORE_VERSION
                  else set())
        begins: dict[str, dt.datetime] = {}
        bootstrap: set[str] = set()
        for entity_id in entity_ids:
            if entity_id in refill:
                begins[entity_id] = now - dt.timedelta(days=BOOTSTRAP_DAYS)
                bootstrap.add(entity_id)
                continue
            seen = self.store.last_seen(entity_id)
            if seen:
                begins[entity_id] = (dt.datetime.fromtimestamp(seen / 1000, dt.timezone.utc)
                                     - dt.timedelta(minutes=OVERLAP_MINUTES))
                continue
            asked = self._asked_empty.get(entity_id)
            if asked is not None and mono - asked[0] < EMPTY_RETRY_SECONDS:
                continue
            # First ask reaches back the whole bootstrap window; a later one
            # only covers the time since the previous ask found nothing.
            begins[entity_id] = (asked[1] - dt.timedelta(minutes=OVERLAP_MINUTES)
                                 if asked else now - dt.timedelta(days=BOOTSTRAP_DAYS))
            if asked is None:
                bootstrap.add(entity_id)
            self._asked_empty[entity_id] = (mono, now)

        # Bucket by HOW FAR BACK a window reaches, not where its start lands on
        # the clock, or the grouping depends on the time of day the cycle ran.
        groups: dict[int, list[str]] = {}
        for entity_id, begin in begins.items():
            key = int((now - begin).total_seconds()) // (WINDOW_BUCKET_HOURS * 3600)
            groups.setdefault(key, []).append(entity_id)

        keep_absence = set(absence_is_a_reading)
        added = requests = 0
        seen_entities: set[str] = set()
        earliest = now
        for _key, ids in sorted(groups.items(), reverse=True):   # oldest window first
            begin = min(begins[e] for e in ids)
            # Only the bootstrap is chunked: a months-old watermark on a quiet
            # entity would otherwise be nine requests every cycle, forever.
            spans = (_windows(begin, now) if any(e in bootstrap for e in ids)
                     else [(begin, now)])
            for start, until in spans:
                series = self.ha.history(ids, _utc(start), _utc(until)) or []
                requests += 1
                earliest = min(earliest, start)
                rows, named = _rows(series, keep_gap, keep_absence)
                # Per window, not once at the end: an interrupted walk then
                # keeps what it got, and resuming re-asks only for the rest.
                added += self.store.append(rows)
                seen_entities |= named
                if not any(series):
                    break

        # After the append, so an interrupted refill runs again. Re-running it
        # is free: the (entity_id, ts) primary key makes the rows idempotent.
        if refill:
            self.store.set_user_version(STORE_VERSION)
        self.store.append([(HEARTBEAT_ENTITY, int(now.timestamp() * 1000), "ok")])
        return {"added": added, "entities": len(seen_entities),
                "since": earliest.isoformat(), "requests": requests}

    def liveness_times(self, start: str, stop: str | None = None) -> list[str]:
        """When we know history was being captured: the collector's heartbeats.

        The backfilled prefix has none; `features._liveness` fills it in.
        """
        return [when for when, _ in self.store.states(HEARTBEAT_ENTITY, start, stop)]
