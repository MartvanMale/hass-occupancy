"""Append-only SQLite history: the recorder purges, so the add-on keeps a copy.

LTS is no shortcut, covering only numeric entities with a `state_class`. One row
per state change, keyed (entity_id, ts), so re-importing a window is idempotent.
`forecasts` keeps what the add-on SAID, whatever the source; it alone is pruned.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
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

# Deliberately NO secondary index on `forecasts`: the primary key IS the
# storage order in a WITHOUT ROWID table, and every read here is a prefix of it.


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
    """The archive. One SQLite file, one connection PER THREAD.

    Shared, a `commit()` on one thread committed what another had in flight.
    """

    def __init__(self, path: Path | str = "/data/history.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        db = self._db
        db.executescript(SCHEMA)
        # WAL so a long feature build reading the store does not block the
        # collector appending to it. Persistent in the file, so once is enough.
        db.execute("PRAGMA journal_mode=WAL")
        db.commit()

    @property
    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            # `check_same_thread=False` only so `close()` can close every
            # connection from whichever thread runs the shutdown.
            db = sqlite3.connect(str(self.path), check_same_thread=False)
            # Wait rather than fail when the writer holds the lock: five
            # seconds is far longer than any single append takes.
            db.execute("PRAGMA busy_timeout=5000")
            self._local.db = db
            with self._connections_lock:
                self._connections.append(db)
        return db

    # -- writing ------------------------------------------------------------

    def append(self, rows: Iterable[tuple[str, int, str]]) -> int:
        """Insert (entity_id, epoch_ms, value). Duplicates are ignored, not errors."""
        rows = list(rows)
        if not rows:
            return 0
        # `rowcount` sums the rows executemany actually changed, and an ignored
        # duplicate is not a change.
        cursor = self._db.executemany(
            "INSERT OR IGNORE INTO states (entity_id, ts, value) VALUES (?, ?, ?)", rows)
        self._db.commit()
        return max(0, cursor.rowcount)

    def append_forecasts(self, rows: Iterable[tuple[str, int, int, float]]) -> int:
        """Insert (subject, target_ts_ms, horizon_h, p). Last write wins.

        Several cycles write each slot; the last is what the sensor was showing.
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

        Epoch ms, not ISO, so the join on the observed grid is integer equality.
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

    def user_version(self) -> int:
        """SQLite's own schema-version field, as a one-shot migration key.

        Zero on older files; set it only after the work it records is done.
        """
        return int(self._db.execute("PRAGMA user_version").fetchone()[0])

    def set_user_version(self, version: int) -> None:
        # No parameter binding: PRAGMA does not take one. The caller passes a
        # module constant, never anything from outside.
        self._db.execute(f"PRAGMA user_version = {int(version)}")
        self._db.commit()

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

        One pass in primary-key order, which IS the storage order: no sort.
        """
        return [{"entity_id": entity_id, "rows": rows,
                 "first": _iso(first), "last": _iso(last)}
                for entity_id, rows, first, last in self._db.execute(
                    "SELECT entity_id, COUNT(*), MIN(ts), MAX(ts) FROM states "
                    "GROUP BY entity_id ORDER BY entity_id")]

    def value_counts(self, entity_id: str, limit: int = 12) -> list[tuple[str, int]]:
        """The commonest values for one entity, most frequent first.

        A peek, never a histogram: a distance sensor has a value per row.
        """
        return [(value, n) for value, n in self._db.execute(
            "SELECT value, COUNT(*) AS n FROM states WHERE entity_id = ? "
            "GROUP BY value ORDER BY n DESC, value LIMIT ?", (entity_id, limit))]

    def prune_forecasts(self, before: str | dt.datetime) -> int:
        """Drop forecasts about slots older than `before`. Returns rows removed.

        Never the archive: this grows every cycle, for a chart of recent days.
        """
        cur = self._db.execute("DELETE FROM forecasts WHERE target_ts < ?",
                               (_ms(before),))
        self._db.commit()
        return cur.rowcount

    def close(self) -> None:
        """Close every thread's connection, at shutdown or on a source swap."""
        with self._connections_lock:
            connections, self._connections = self._connections, []
        for db in connections:
            try:
                db.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()
