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
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from .. import config
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
# far less than this -- 10 days is the stock default -- but an install that has
# been recording for months should get all of it.
BOOTSTRAP_DAYS = 400

# ...asked for in windows this wide, newest first, rather than in one request.
# The response is parsed whole before a row is stored, and a deep archive is
# large: measured, six proximity entities cost 0.5 MB per 30 days, so three
# years of them would be one ~18 MB response to hold in memory on a Pi.
BOOTSTRAP_CHUNK_DAYS = 30

# Re-fetch this much on every poll. Writes are idempotent (primary key on
# entity_id + ts), so overlapping is free and it means a missed poll, a restart
# or a clock skew heals itself instead of leaving a hole.
OVERLAP_MINUTES = 90

# Entities are fetched in GROUPS by how far back their watermark reaches,
# bucketed to this many hours, one history request per group. It used to be
# one request for all of them from the OLDEST watermark: a work zone nobody
# entered during a three-week trip dragged every proximity sensor's window out
# to three weeks, every five minutes, on the box that also runs the recorder.
# Six hours puts everything that reported today into one request and gives a
# quiet entity its own, short one.
WINDOW_BUCKET_HOURS = 6

# An entity with NO rows at all -- excluded from the recorder, or a sensor that
# has never reported -- asked for BOOTSTRAP_DAYS on every cycle, forever. Now
# it is asked once, then only for the stretch since it was last asked, and no
# more often than this.
EMPTY_RETRY_SECONDS = 3600

# What gets stored for an entity whose absence is itself a reading. Its own
# word rather than `unavailable` or `unknown`, because Home Assistant uses both
# for this and a reader should not have to handle two spellings of one fact.
ABSENT = "absent"

# What gets stored where a reading is merely MISSING. A different fact from
# ABSENT and so a different word: "no alarm is set" is something we know, and
# this is something we do not. It exists as a row rather than as silence
# because the row is what ends the preceding state -- without it the last
# known state carries forward for as long as the tracker stays quiet.
UNKNOWN = "unknown"

# Bumped when a release needs the store rewritten or refilled once. 1 re-pulls
# presence over the bootstrap window: releases before this dropped every
# `unknown` from a person or group, so those transitions are missing from
# every archive already written and the recorder is the only place left that
# still has them.
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

    Newest first is what lets a walk stop: history runs out at the recorder's
    purge horizon and there is only one of those, so the first empty window
    means every window below it is empty too.
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

        The window starts at the oldest per-entity watermark minus an overlap,
        so an entity added to the config later gets backfilled with whatever
        recorder still holds rather than starting from now. A window wider than
        `BOOTSTRAP_CHUNK_DAYS` is walked backwards in chunks and stops at the
        first empty one, which is the recorder's purge horizon.

        Two kinds of entity keep what would otherwise be dropped, for two
        different reasons.

        `absence_is_a_reading` names the entities for which `unavailable` and
        `unknown` are DATA. There is exactly one shape of sensor like that: a
        next-alarm sensor reads `unavailable` precisely when no alarm is set,
        which is the more common state and at least as informative as a time.
        Dropping it would leave an archive that says nothing at all on the days
        somebody had no alarm -- indistinguishable from the days the sensor was
        broken, and unrecoverable later, because Home Assistant's recorder will
        long since have discarded the difference.

        `gap_is_a_boundary` names the presence entities, where the same words
        mean the opposite: not a reading, but the END of one. Dropped, they
        left the previous state to be carried forward indefinitely, and a phone
        that stopped reporting read as everybody-out for as long as it stayed
        quiet -- straight into the training labels. Stored as UNKNOWN, the
        interval is uncovered instead, which is what it is.
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

        # Bucket by HOW FAR BACK a window reaches, not by where its start lands
        # on the clock. Flooring the absolute hour splits two entities either
        # side of a boundary -- one seen five minutes ago and one seen two hours
        # ago land in different six-hour blocks whenever the older one happens
        # to cross it -- so the grouping would depend on the time of day the
        # cycle ran, and "everything that reported today" would be one request
        # only sometimes.
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
            # Only the bootstrap is chunked. A watermark can be months old on an
            # entity that rarely changes -- a zone nobody visits -- and chunking
            # that would turn one request every five minutes into nine, forever.
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
        """When we know history was being captured.

        Heartbeats cover everything since the add-on was installed. The window
        before that was backfilled from the recorder and has none, so the
        tracked entities' own changes stand in for it -- which is the best that
        can be said about a period nobody was watching.
        """
        return [when for when, _ in self.store.states(HEARTBEAT_ENTITY, start, stop)]
