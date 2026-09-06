"""The archive, and the record of what was forecast beside it.

The `forecasts` table is a migration onto a file that already exists on every
installation, so the first test here is the one that matters most: opening a
store against a database that predates the table must add it rather than fail.
CLAUDE.md's rule is "a migration, never a rewrite", and `CREATE TABLE IF NOT
EXISTS` under the existing `executescript` is what makes that true -- but only
if nothing else in the open path assumes a fresh file.
"""

import datetime as dt
import sqlite3

import pytest

from occupancy_forecast.sources.store import HistoryStore

STATES_ONLY = """
CREATE TABLE IF NOT EXISTS states (
    entity_id TEXT NOT NULL,
    ts        INTEGER NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (entity_id, ts)
) WITHOUT ROWID;
"""


def _ms(when: dt.datetime) -> int:
    return int(when.timestamp() * 1000)


@pytest.fixture
def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history.db")


def test_the_forecast_table_is_added_to_an_existing_archive(tmp_path, now):
    """The upgrade path every installed add-on takes: a database with months of
    `states` in it and no `forecasts`. The rows already there must survive."""
    path = tmp_path / "history.db"
    old = sqlite3.connect(str(path))
    old.executescript(STATES_ONLY)
    old.execute("INSERT INTO states VALUES (?, ?, ?)",
                ("person.alice", _ms(now - dt.timedelta(days=200)), "home"))
    old.commit()
    old.close()

    store = HistoryStore(path)
    assert store.count() == 1, "the archive survived the migration"
    assert store.forecast_count() == 0
    store.append_forecasts([("alice", _ms(now), 6, 0.4)])
    assert store.forecast_count() == 1


def test_the_last_forecast_for_a_slot_wins(store, now):
    """A 30-minute slot is covered by several five-minute cycles, each with a
    fresher feature row. The one the sensor was holding when the slot arrived is
    the last one written, and that is what the chart must score."""
    target = _ms(now)
    store.append_forecasts([("alice", target, 6, 0.10)])
    store.append_forecasts([("alice", target, 6, 0.90)])

    series = store.forecast_series("alice", 6, (now - dt.timedelta(hours=1)).isoformat())
    assert series == [(target, 0.90)]


def test_horizons_and_subjects_do_not_collide(store, now):
    """All three columns are in the key, so the same slot holds one row per
    horizon per subject -- 48 curves crossing every slot, not one."""
    target = _ms(now)
    store.append_forecasts([
        ("alice", target, 6, 0.1),
        ("alice", target, 24, 0.2),
        ("bob", target, 6, 0.3),
    ])
    assert store.forecast_count() == 3
    assert store.forecast_count("alice") == 2

    start = (now - dt.timedelta(hours=1)).isoformat()
    assert store.forecast_series("alice", 6, start) == [(target, 0.1)]
    assert store.forecast_series("bob", 6, start) == [(target, 0.3)]


def test_a_subject_with_no_forecasts_is_empty_rather_than_an_error(store, now):
    """A fresh install, and every install for its first hours. `explore` turns
    this into a readable "nothing has come due yet", which it can only do if the
    store answers rather than raises."""
    assert store.forecast_series("nobody", 6,
                                 (now - dt.timedelta(days=1)).isoformat()) == []
    assert store.forecast_count("nobody") == 0


def test_the_window_is_honoured_at_both_ends(store, now):
    """`stop` is what keeps forecasts about the FUTURE off a chart of the past:
    a +48 h forecast made now targets a slot two days out, and until that slot
    arrives there is nothing to compare it to."""
    store.append_forecasts([
        ("alice", _ms(now - dt.timedelta(days=3)), 6, 0.1),   # before start
        ("alice", _ms(now - dt.timedelta(hours=2)), 6, 0.2),  # inside
        ("alice", _ms(now + dt.timedelta(days=2)), 6, 0.3),   # not due yet
    ])
    series = store.forecast_series(
        "alice", 6,
        (now - dt.timedelta(days=1)).isoformat(),
        now.isoformat())
    assert [round(p, 1) for _, p in series] == [0.2]


def test_pruning_removes_only_what_is_past_the_window(store, now):
    """The archive is never pruned and this table always is -- so the test that
    earns its keep is the one proving the cut lands where it is asked to."""
    keep = _ms(now - dt.timedelta(days=5))
    drop = _ms(now - dt.timedelta(days=40))
    store.append_forecasts([("alice", drop, 6, 0.1), ("alice", keep, 6, 0.2)])

    removed = store.prune_forecasts(now - dt.timedelta(days=30))
    assert removed == 1
    assert store.forecast_count() == 1
    assert store.forecast_series("alice", 6,
                                 (now - dt.timedelta(days=90)).isoformat()) == [(keep, 0.2)]


def test_pruning_forecasts_leaves_the_archive_alone(store, now):
    """They live in one file and only one of them has a retention policy."""
    store.append([("person.alice", _ms(now - dt.timedelta(days=300)), "home")])
    store.append_forecasts([("alice", _ms(now - dt.timedelta(days=300)), 6, 0.1)])

    store.prune_forecasts(now - dt.timedelta(days=30))
    assert store.forecast_count() == 0
    assert store.count() == 1, "the training history is not the chart's to delete"
