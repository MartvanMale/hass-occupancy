"""An unobserved tracker is not an empty house.

Everything here is one bug seen from four sides. A person entity that stops
reporting used to leave no trace at all: the collector dropped `unknown`, so
nothing ended the last known state, and `slot_fraction` carried it forward for
as long as the silence lasted. `observability` does not catch it either -- that
mask answers "was the recorder alive", built from the union of every tracked
entity, so one dead phone among nine live sensors passes straight through. The
result went into the training labels as a settled fact about where people were.
"""

import numpy as np
import pandas as pd

from occupancy_forecast import config, features
from occupancy_forecast.sources.ha import STORE_VERSION, StoreSource
from occupancy_forecast.sources.store import HistoryStore, _ms

from .conftest import settings


def _store(tmp_path, rows=()):
    store = HistoryStore(tmp_path / "history.db")
    if rows:
        store.append([(entity, _ms(when), state) for entity, when, state in rows])
    return store


# ---------------------------------------------------------------------------
# The slot
# ---------------------------------------------------------------------------

def test_unknown_interval_is_not_away_or_carried_home():
    """The middle slot has no coverage, rather than a fraction of zero."""
    slots = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    events = [(slots[0].isoformat(), "home"), (slots[1].isoformat(), "unknown"),
              (slots[2].isoformat(), "not_home")]

    result = features.slot_fraction(events, slots, config.HOME_STATE)

    assert result["frac"].iloc[0] == 1
    assert np.isnan(result["frac"].iloc[1])
    assert result["coverage"].iloc[1] == 0
    assert result["frac"].iloc[2] == 0


def test_a_silent_tracker_stops_reading_as_away():
    """The whole point, stated once: the gap ends the state it interrupts.

    Without the `unknown` row the last value holds for every later slot, which
    is how a phone that went quiet at breakfast reported an empty house until
    it came back.
    """
    slots = pd.date_range("2026-01-01", periods=6, freq="30min", tz="UTC")
    carried = features.slot_fraction(
        [(slots[0].isoformat(), "not_home")], slots, config.HOME_STATE)
    ended = features.slot_fraction(
        [(slots[0].isoformat(), "not_home"), (slots[1].isoformat(), "unknown")],
        slots, config.HOME_STATE)

    assert (carried["frac"] == 0).all()
    assert ended["frac"].iloc[0] == 0
    assert ended["frac"].iloc[1:].isna().all()


# ---------------------------------------------------------------------------
# The house
# ---------------------------------------------------------------------------

def test_the_house_is_unknown_when_the_only_known_person_is_out(tmp_path):
    """One un-observed housemate must not produce an "everyone out" interval."""
    # No group entity, so the house is the OR over the people and this
    # exercises the merge. With one configured, HA has already done the OR --
    # and its own `unknown` now reaches the store the same way.
    config.configure(settings(house_entity=None))
    store = _store(tmp_path, [
        ("person.alice", "2026-01-01T00:00:00Z", "not_home"),
        ("person.bob", "2026-01-01T00:00:00Z", "unknown"),
        ("person.alice", "2026-01-01T00:30:00Z", "home"),
    ])
    try:
        events = features.presence_events(
            StoreSource(store, None), config.SUBJECTS[-1],
            "2026-01-01T00:00:00Z", None)
        assert events[0][1] == "unknown"
        assert events[-1][1] == config.HOME_STATE
    finally:
        store.close()


def test_a_known_person_at_home_outranks_an_unknown_housemate(tmp_path):
    """Home survives an unknown; away does not.

    Somebody known to be in the house settles the question whatever anyone
    else's tracker is doing -- the house is occupied. The reverse does not
    hold, which is the asymmetry `presence_events` encodes by testing `anyone`
    before `all_known`, and the one a later reader is most likely to flatten.
    """
    # No group entity, so the house is the OR over the people and this
    # exercises the merge. With one configured, HA has already done the OR --
    # and its own `unknown` now reaches the store the same way.
    config.configure(settings(house_entity=None))
    store = _store(tmp_path, [
        ("person.alice", "2026-01-01T00:00:00Z", "home"),
        ("person.bob", "2026-01-01T00:00:00Z", "unknown"),
    ])
    try:
        source = StoreSource(store, None)
        assert [state for _, state in features.presence_events(
            source, config.SUBJECTS[-1], "2026-01-01T00:00:00Z", None)] \
            == [config.HOME_STATE]

        # And the people themselves stay independent: Alice is home, Bob is
        # not anywhere in particular.
        slots = pd.date_range("2026-01-01", periods=2, freq="30min", tz="UTC")
        alice, bob = (features.slot_fraction(
            features.presence_events(source, p, "2026-01-01T00:00:00Z", None),
            slots, config.HOME_STATE) for p in config.PEOPLE)
        assert alice["frac"].iloc[0] == 1
        assert np.isnan(bob["frac"].iloc[0])
    finally:
        store.close()


def test_a_person_added_today_does_not_blank_the_house(tmp_path):
    """Enrolling somebody must not delete the history that came before them.

    A person joins the OR at their first observation. Requiring every
    configured person instead would make the house unknown for every day the
    recorder cannot supply the newcomer -- which is all of them older than its
    ~10-day retention, so adding a housemate would throw away the training set.
    """
    # No group entity, so the house is the OR over the people and this
    # exercises the merge. With one configured, HA has already done the OR --
    # and its own `unknown` now reaches the store the same way.
    config.configure(settings(house_entity=None))
    store = _store(tmp_path, [
        ("person.alice", "2026-01-01T00:00:00Z", "home"),
        ("person.alice", "2026-01-01T01:00:00Z", "not_home"),
        ("person.bob", "2026-01-01T02:00:00Z", "home"),
    ])
    try:
        events = features.presence_events(
            StoreSource(store, None), config.SUBJECTS[-1],
            "2026-01-01T00:00:00Z", None)
        assert [state for _, state in events] == [
            config.HOME_STATE, "not_home", config.HOME_STATE]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------

class _HA:
    """Answers one history call, recording the windows it was asked for."""

    def __init__(self, series):
        self.series, self.asked = series, []

    def history(self, entity_ids, start, stop):
        self.asked.append((sorted(entity_ids), start))
        return self.series


def test_a_presence_gap_is_stored_rather_than_dropped(tmp_path):
    ha = _HA([[{"entity_id": "person.alice", "state": "home",
                "last_changed": "2026-01-01T09:00:00+00:00"},
               {"entity_id": "person.alice", "state": "unavailable",
                "last_changed": "2026-01-01T10:00:00+00:00"}]])
    store = _store(tmp_path)
    try:
        StoreSource(store, ha).collect(["person.alice"],
                                       gap_is_a_boundary=["person.alice"])
        # Normalised: HA writes both words for this and the reader should not
        # have to know which one arrived.
        assert [s for _, s in store.states("person.alice", "2026-01-01T00:00:00Z")] \
            == [config.HOME_STATE, "unknown"]
    finally:
        store.close()


def test_the_presence_refill_runs_once_and_not_on_every_restart(tmp_path):
    """The recorder is the only place the dropped transitions still exist.

    So they are re-pulled -- but keyed in the store, not on the instance. A
    per-process flag would re-request the whole bootstrap window every time the
    add-on restarted, which on a supervised box is often.
    """
    rows = [[{"entity_id": "person.alice", "state": "home",
              "last_changed": "2026-01-01T09:00:00+00:00"}]]
    store = _store(tmp_path)
    try:
        assert store.user_version() == 0

        first = _HA(rows)
        StoreSource(store, first).collect(["person.alice"],
                                          gap_is_a_boundary=["person.alice"])
        assert store.user_version() == STORE_VERSION

        # A second process, so nothing is remembered in memory. It asks from
        # the watermark, not from the bootstrap window. Compared on the OLDEST
        # window of each: the bootstrap is walked backwards in chunks, so its
        # FIRST request is the most recent one and says nothing about reach.
        second = _HA(rows)
        StoreSource(store, second).collect(["person.alice"],
                                           gap_is_a_boundary=["person.alice"])
        assert second.asked[-1][1] > first.asked[-1][1]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The training gate
# ---------------------------------------------------------------------------

def test_archive_age_does_not_count_untracked_person_days(tmp_path):
    """Twelve days of rows, one day of usable observation.

    `span()` measures the age of the oldest row, which is why a fresh install
    with one stale tracker looked ready to train days early.
    """
    config.configure(settings(zones=[], proximity={}))
    stamps = pd.date_range("2026-01-01", "2026-01-13", freq="30min", tz="UTC")
    store = _store(tmp_path, [
        (person, stamp.isoformat(),
         "unknown" if person == "person.bob" and stamp.day < 12 else config.HOME_STATE)
        for person in ("person.alice", "person.bob") for stamp in stamps])
    try:
        days = features.usable_history_days(StoreSource(store, None),
                                            "2026-01-13T00:00:00Z")
        assert store.span()["days"] == 12
        # The end-exclusive liveness window may withhold its final slot.
        assert 0.95 <= days <= 1
    finally:
        store.close()
