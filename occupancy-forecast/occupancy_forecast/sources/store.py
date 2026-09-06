"""Append-only SQLite history, the add-on's own archive.

This exists because **Home Assistant cannot supply training history.** Measured
on a well-configured instance: a 100-day request reached back 21 days despite
`purge_keep_days: 100`, and stock Home Assistant defaults to 10. There is no
long-term-statistics shortcut either -- LTS covers only numeric entities with a
`state_class`, and presence is a string while the proximity distance sensors
carry `state_class: None`.

So the add-on accumulates its own. Measured at 158 state changes per day across
the seven entities a two-person household tracks, that is **~2.3 MB per year**,
and the add-on's `/data` survives restarts and updates. Which means this store
keeps history *forever* -- strictly better than the recorder it reads from.

Storage is deliberately dumb: one row per state change, primary key
(entity_id, ts). Re-importing an overlapping window is therefore idempotent, so
the collector can be sloppy about its watermark and simply re-fetch the last
hour on every poll.

A second table, `forecasts`, keeps what the add-on *said* rather than what the
house did. Nothing else records it: `predict.py` writes no file, the server's
in-memory forecast is one slot overwritten every cycle, and a retained MQTT
message is a last value rather than a history. Without it there is no way to ask
"what did we forecast for 07:00, and what actually happened at 07:00" -- every
score the add-on reports otherwise is cross-validation at *training* time, which
answers a different question and cannot see the serving path at all. It is
pruned; the archive is not. See `config.FORECAST_RETENTION_DAYS`.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS states (
    entity_id TEXT NOT NULL,
    ts        INTEGER NOT NULL,     -- epoch milliseconds, UTC
    value     TEXT NOT NULL,
    PRIMARY KEY (entity_id, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS states_entity_ts ON states (entity_id, ts);
CREATE TABLE IF NOT EXISTS forecasts (
    subject   TEXT    NOT NULL,
    target_ts INTEGER NOT NULL,     -- epoch milliseconds, UTC, on the slot grid
    horizon_h INTEGER NOT NULL,
    p         REAL    NOT NULL,
    PRIMARY KEY (subject, target_ts, horizon_h)
) WITHOUT ROWID;
"""

# Deliberately NO secondary index on `forecasts`. The primary key IS the
# storage order in a WITHOUT ROWID table, and every read here is a prefix of
# it; the redundant index on `states` above is already noted as a mistake in
# this module's own history.


def _ms(when: str | dt.datetime) -> int:
    if isinstance(when, dt.datetime):
        moment = when
    else:
        text = when.replace("Z", "+00:00")
        moment = dt.datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return int(moment.timestamp() * 1000)


def _iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat()


class HistoryStore:
    def __init__(self, path: Path | str = "/data/history.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        # WAL so a long feature build reading the store does not block the
        # collector appending to it.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.commit()

    # -- writing ------------------------------------------------------------

    def append(self, rows: Iterable[tuple[str, int, str]]) -> int:
        """Insert (entity_id, epoch_ms, value). Duplicates are ignored, not errors."""
        rows = list(rows)
        if not rows:
            return 0
        before = self.count()
        self._db.executemany(
            "INSERT OR IGNORE INTO states (entity_id, ts, value) VALUES (?, ?, ?)", rows)
        self._db.commit()
        return self.count() - before

    def append_forecasts(self, rows: Iterable[tuple[str, int, int, float]]) -> int:
        """Insert (subject, target_ts_ms, horizon_h, p). Last write wins.

        `INSERT OR REPLACE`, not `OR IGNORE` as `append` uses, and the
        difference matters. A 30-minute slot is covered by several five-minute
        serve cycles, each with a fresher feature row, so the same
        (subject, target_ts, horizon) is written repeatedly with different
        numbers. The last one is the one the sensor was actually holding when
        the slot arrived, which is what this table exists to be scored against.
        """
        rows = list(rows)
        if not rows:
            return 0
        self._db.executemany(
            "INSERT OR REPLACE INTO forecasts (subject, target_ts, horizon_h, p) "
            "VALUES (?, ?, ?, ?)", rows)
        self._db.commit()
        return len(rows)

    # -- reading ------------------------------------------------------------

    def states(self, entity_id: str, start: str,
               stop: str | None = None) -> list[tuple[str, str]]:
        sql = "SELECT ts, value FROM states WHERE entity_id = ? AND ts >= ?"
        args: list = [entity_id, _ms(start)]
        if stop:
            sql += " AND ts < ?"
            args.append(_ms(stop))
        sql += " ORDER BY ts"
        return [(_iso(ts), value) for ts, value in self._db.execute(sql, args)]

    def seeded_states(self, entity_id: str, start: str, stop: str | None = None,
                      seed_days: int = 14) -> list[tuple[str, str]]:
        rows = self.states(entity_id, start, stop)
        seed = self._db.execute(
            "SELECT value FROM states WHERE entity_id = ? AND ts < ? "
            "ORDER BY ts DESC LIMIT 1", (entity_id, _ms(start))).fetchone()
        if seed:
            rows = [(_iso(_ms(start)), seed[0]), *rows]
        return rows

    def numeric(self, entity_id: str, start: str,
                stop: str | None = None) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for when, value in self.states(entity_id, start, stop):
            try:
                out.append((when, float(value)))
            except (TypeError, ValueError):
                continue
        return out

    def forecast_series(self, subject: str, horizon_h: int, start: str,
                        stop: str | None = None) -> list[tuple[int, float]]:
        """What was forecast for each slot at one horizon, oldest first.

        Epoch milliseconds rather than the ISO strings `states` returns, on
        purpose: the caller joins this against the observed grid by equality on
        the slot, and an integer key makes that a dict lookup rather than a
        string comparison that any timezone spelling could break.
        """
        sql = ("SELECT target_ts, p FROM forecasts "
               "WHERE subject = ? AND horizon_h = ? AND target_ts >= ?")
        args: list = [subject, int(horizon_h), _ms(start)]
        if stop:
            sql += " AND target_ts < ?"
            args.append(_ms(stop))
        sql += " ORDER BY target_ts"
        return [(ts, p) for ts, p in self._db.execute(sql, args)]

    def forecast_count(self, subject: str | None = None) -> int:
        if subject is None:
            return self._db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        return self._db.execute(
            "SELECT COUNT(*) FROM forecasts WHERE subject = ?", (subject,)).fetchone()[0]

    # -- housekeeping -------------------------------------------------------

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM states").fetchone()[0]

    def last_seen(self, entity_id: str) -> int | None:
        row = self._db.execute(
            "SELECT MAX(ts) FROM states WHERE entity_id = ?", (entity_id,)).fetchone()
        return row[0] if row and row[0] is not None else None

    def span(self) -> dict:
        row = self._db.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM states").fetchone()
        first, last, n = row
        return {
            "first": _iso(first) if first else None,
            "last": _iso(last) if last else None,
            "rows": n,
            "days": round((last - first) / 86_400_000, 1) if first and last else 0.0,
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def entities(self) -> list[str]:
        return [r[0] for r in self._db.execute(
            "SELECT DISTINCT entity_id FROM states ORDER BY entity_id")]

    def inventory(self) -> list[dict]:
        """One row per entity: how much of it there is, and what it spans.

        What `entities()` gives, with the numbers that make it worth looking at
        -- an entity present with a plausible row count is a signal that arrived,
        and one present with a `last` from three weeks ago is a signal that
        stopped. Neither was visible before without a sqlite3 prompt.

        One pass in `(entity_id, ts)` order: the table is WITHOUT ROWID, so that
        primary key IS the storage order and the GROUP BY needs no sort.
        """
        return [{"entity_id": entity_id, "rows": rows,
                 "first": _iso(first), "last": _iso(last)}
                for entity_id, rows, first, last in self._db.execute(
                    "SELECT entity_id, COUNT(*), MIN(ts), MAX(ts) FROM states "
                    "GROUP BY entity_id ORDER BY entity_id")]

    def value_counts(self, entity_id: str, limit: int = 12) -> list[tuple[str, int]]:
        """The commonest values for one entity, most frequent first.

        A peek, never a histogram, and the cap is the reason: a presence entity
        has three distinct values and a distance sensor has one per row, so
        without a limit this is a 50,000-row answer to "what kind of thing is
        this". It is used to tell those two cases apart, which the first handful
        of rows settles.
        """
        return [(value, n) for value, n in self._db.execute(
            "SELECT value, COUNT(*) AS n FROM states WHERE entity_id = ? "
            "GROUP BY value ORDER BY n DESC, value LIMIT ?", (entity_id, limit))]

    def prune_forecasts(self, before: str | dt.datetime) -> int:
        """Drop forecasts about slots older than `before`. Returns rows removed.

        The archive is never pruned -- it is the training history and it is
        cheap. This table is neither: it is written every cycle for every
        horizon, so it grows about two hundred times faster per day, and its
        only consumer is a chart of the recent past.
        """
        cur = self._db.execute("DELETE FROM forecasts WHERE target_ts < ?",
                               (_ms(before),))
        self._db.commit()
        return cur.rowcount

    def close(self) -> None:
        self._db.close()
