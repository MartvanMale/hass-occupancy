"""The archive, and the record of what was forecast beside it.

The `forecasts` table is a migration onto a file that already exists on every
installation, so opening a store against a database that predates the table
must add it rather than fail.
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
    """A 30-minute slot is covered by several five-minute cycles; the one the
    sensor was holding when the slot arrived is the last written, and that is
    what the chart must score."""
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
    """A fresh install, and every install for its first hours: `explore` can only
    say "nothing has come due yet" if the store answers rather than raises."""
    assert store.forecast_series("nobody", 6,
                                 (now - dt.timedelta(days=1)).isoformat()) == []
    assert store.forecast_count("nobody") == 0


def test_the_window_is_honoured_at_both_ends(store, now):
    """`stop` is what keeps forecasts about the FUTURE off a chart of the past:
    a +48 h forecast targets a slot with nothing to compare it to yet."""
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


def test_append_reports_only_the_rows_it_actually_inserted(store, now):
    """The collector's "added" number, without a COUNT(*) scan on either side of
    the insert to compute it."""
    rows = [("person.alice", _ms(now), "home"), ("person.alice", _ms(now) + 1, "home")]
    assert store.append(rows) == 2
    assert store.append(rows) == 0, "duplicates are ignored, and not counted"
    assert store.append([*rows, ("person.bob", _ms(now), "home")]) == 1
    assert store.count() == 3


# ---------------------------------------------------------------------------
# One connection per thread
#
# The collector, the training thread and every uvicorn handler read this store
# at once, so none of them may share a connection.
# ---------------------------------------------------------------------------

def test_each_thread_gets_its_own_connection_and_close_shuts_them_all(store, now):
    import threading

    seen = {}
    errors = []

    def work(i: int):
        try:
            seen[i] = id(store._db)
            for k in range(50):
                store.append([(f"sensor.t{i}", _ms(now) + k, str(k))])
                store.count()
                store.states(f"sensor.t{i}", (now - dt.timedelta(hours=1)).isoformat())
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert not errors, errors
    assert len(set(seen.values())) == 4, "four threads, four connections"
    assert store.count() == 200, "every thread's writes landed"
    store.close()
    assert store._connections == []
    # And it is usable again afterwards: a fresh connection on demand.
    assert store.count() == 200


# ---------------------------------------------------------------------------
# What the collector asks Home Assistant for
#
# One request per watermark bucket, and an entity with no history is asked once
# an hour -- not for 400 days every five minutes, forever.
# ---------------------------------------------------------------------------

class _RecordingHA:
    def __init__(self):
        self.calls: list[tuple[list[str], str, str]] = []

    def history(self, entity_ids, start, stop=None):
        self.calls.append((list(entity_ids), start, stop))
        return []


def _iso_to_dt(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_the_collector_groups_entities_by_how_far_back_they_reach(store, now):
    from occupancy_forecast.sources import ha as ha_mod

    store.append([
        ("person.alice", _ms(now - dt.timedelta(hours=1)), "home"),
        ("zone.work", _ms(now - dt.timedelta(days=10)), "0"),
    ])
    ha = _RecordingHA()
    source = ha_mod.StoreSource(store, ha)

    result = source.collect(["person.alice", "zone.work", "sensor.never"])

    assert result["requests"] == 3
    by_entity = {tuple(ids): _iso_to_dt(start) for ids, start, _ in ha.calls}
    assert now - by_entity[("person.alice",)] < dt.timedelta(hours=3)
    assert dt.timedelta(days=9) < now - by_entity[("zone.work",)] < dt.timedelta(days=11)
    # The bootstrap walks backwards a chunk at a time, and an empty answer stops
    # the walk at the first window.
    assert now - by_entity[("sensor.never",)] < dt.timedelta(
        days=ha_mod.BOOTSTRAP_CHUNK_DAYS + 1)

    # Five minutes later: the empty entity is not asked again.
    ha.calls.clear()
    result = source.collect(["person.alice", "zone.work", "sensor.never"])
    assert result["requests"] == 2
    assert all("sensor.never" not in ids for ids, _, _ in ha.calls)

    # An hour later it is, but only for the stretch since the last ask.
    asked_mono, asked_at = source._asked_empty["sensor.never"]
    source._asked_empty["sensor.never"] = (asked_mono - ha_mod.EMPTY_RETRY_SECONDS - 1, asked_at)
    ha.calls.clear()
    source.collect(["sensor.never"])
    (_ids, start, _stop), = ha.calls
    assert asked_at - _iso_to_dt(start) < dt.timedelta(hours=2), \
        "the 400-day window is asked for once, not on every retry"


class _RecorderWithAFloor:
    """A recorder holding `days` of history and nothing before it. Like
    `minimal_response`, it carries one entry forward per entity even when
    nothing changed, so "no entries" means "past the purge horizon"."""

    def __init__(self, now, days, quiet=()):
        self.now, self.days, self.quiet = now, days, set(quiet)
        self.calls: list[tuple[list[str], str, str]] = []

    def history(self, entity_ids, start, stop=None):
        self.calls.append((list(entity_ids), start, stop))
        begin = _iso_to_dt(start)
        if self.now - begin > dt.timedelta(days=self.days):
            return []
        return [[{"entity_id": e, "state": "home", "last_changed": start}]
                if e in self.quiet else
                [{"entity_id": e, "state": "home", "last_changed": start},
                 {"entity_id": e, "state": "not_home",
                  "last_changed": _utc(begin + dt.timedelta(hours=1))}]
                for e in entity_ids]


def _utc(when: dt.datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_a_deep_archive_is_imported_in_windows_down_to_the_recorder_floor(store, now):
    """The bootstrap walks back a chunk at a time and stops where history does,
    rather than asking for the whole archive in one request."""
    from occupancy_forecast.sources import ha as ha_mod

    ha = _RecorderWithAFloor(now, days=95)
    result = ha_mod.StoreSource(store, ha).collect(["person.alice"])

    chunk = ha_mod.BOOTSTRAP_CHUNK_DAYS
    reached = now - min(_iso_to_dt(start) for _ids, start, _ in ha.calls)
    assert dt.timedelta(days=95) < reached < dt.timedelta(days=95 + chunk), \
        "walked past the floor to prove it, but not on to the full 400 days"
    assert result["requests"] < ha_mod.BOOTSTRAP_DAYS // chunk
    assert store.states("person.alice", _utc(now - dt.timedelta(days=400)))


def test_a_quiet_entity_does_not_stop_the_walk_early(store, now):
    """A zone nobody entered for a month yields no state CHANGES in that window,
    so the stop signal is an empty response, which only the purge horizon
    produces -- not "nothing was appended"."""
    from occupancy_forecast.sources import ha as ha_mod

    ha = _RecorderWithAFloor(now, days=95, quiet=["zone.work"])
    ha_mod.StoreSource(store, ha).collect(["zone.work"])

    reached = now - min(_iso_to_dt(start) for _ids, start, _ in ha.calls)
    assert reached > dt.timedelta(days=95)


def test_entities_that_reported_today_share_one_request(store, now):
    from occupancy_forecast.sources import ha as ha_mod

    store.append([
        ("person.alice", _ms(now - dt.timedelta(minutes=5)), "home"),
        ("person.bob", _ms(now - dt.timedelta(hours=2)), "not_home"),
        ("sensor.distance", _ms(now - dt.timedelta(minutes=1)), "12.5"),
    ])
    ha = _RecordingHA()
    result = ha_mod.StoreSource(store, ha).collect(
        ["person.alice", "person.bob", "sensor.distance"])
    assert result["requests"] == 1
    assert set(ha.calls[0][0]) == {"person.alice", "person.bob", "sensor.distance"}
