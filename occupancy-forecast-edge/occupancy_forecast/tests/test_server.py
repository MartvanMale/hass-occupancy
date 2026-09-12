"""The scheduling and the retrain guard, tested as plain functions because the
suite ships no httpx. `_start_background_train` has to refuse BEFORE it spawns:
an HTTPException raised inside that thread reaches nobody.
"""

import datetime as dt
import json
import re
import threading
import time

import pytest
from fastapi import HTTPException

from occupancy_forecast import runtime as runtime_mod, server


# ---------------------------------------------------------------------------
# When the next train is
# ---------------------------------------------------------------------------

def at(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


SHORT = server.MIN_DAYS_TO_TRAIN + 1      # still retraining daily
MATURE = server.FULL_HISTORY_DAYS + 1     # weekly now


def test_a_daily_schedule_takes_the_next_04_00():
    # Monday 03:00 -- today's run has not happened yet.
    assert server._next_train(at("2026-09-07T03:00"), SHORT).startswith("2026-09-07T04:00")
    # Monday 05:00 -- it has, so tomorrow.
    assert server._next_train(at("2026-09-07T05:00"), SHORT).startswith("2026-09-08T04:00")


def test_a_weekly_schedule_lands_on_the_training_weekday():
    """Monday, and the same Monday when there is still time to make it."""
    assert server._next_train(at("2026-09-07T03:00"), MATURE).startswith("2026-09-07T04:00")
    # Monday 05:00: this week's run is gone, so the next Monday.
    assert server._next_train(at("2026-09-07T05:00"), MATURE).startswith("2026-09-14T04:00")
    # Thursday: still the following Monday.
    assert server._next_train(at("2026-09-10T12:00"), MATURE).startswith("2026-09-14T04:00")


@pytest.mark.parametrize("days", [SHORT, MATURE])
@pytest.mark.parametrize("now", ["2026-09-07T03:59", "2026-09-07T04:00",
                                 "2026-09-07T04:01", "2026-09-13T23:59"])
def test_the_next_train_is_always_in_the_future(now, days):
    """04:00 exactly is the case that gets this wrong: the run is happening, so
    the next one is not today."""
    assert at(server._next_train(at(now), days)) > at(now)


def test_the_schedule_is_read_on_the_household_clock():
    """An aware `now` in the household's zone, which is what the worker and
    the status page pass since the naive `datetime.now()` went."""
    now = dt.datetime(2026, 9, 7, 3, 0, tzinfo=config_mod.tzinfo())
    answer = server._next_train(now, SHORT)
    assert answer.startswith("2026-09-07T04:00")
    assert dt.datetime.fromisoformat(answer).utcoffset() == now.utcoffset()


def test_a_horizon_off_the_grid_is_a_404_not_an_answer():
    with pytest.raises(HTTPException) as raised:
        server._known_horizon(999)
    assert raised.value.status_code == 404
    assert server._known_horizon(24) == 24


def test_there_is_no_next_train_before_there_is_enough_history():
    """Saying "tomorrow at 04:00" would be a lie -- the worker checks the same
    threshold and will skip."""
    assert server._next_train(at("2026-09-07T03:00"), 1.0) is None


# ---------------------------------------------------------------------------
# Starting one by hand
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def lock_released():
    yield
    if server._train_lock.locked():
        server._train_lock.release()


class _WithStore:
    """A source backed by the local archive, which is what gates on days."""
    store = object()


def test_the_status_page_never_recounts_usable_history(monkeypatch):
    """`usable_history_days` walks the whole archive, so it belongs to the worker
    and the ten-second status poll reads what the worker left."""
    def _refuse(*args, **kwargs):
        raise AssertionError("usable_history_days called from a request handler")

    monkeypatch.setattr(server.features, "usable_history_days", _refuse)
    monkeypatch.setitem(server._state, "source", _WithStore())
    monkeypatch.setitem(server._state, "usable_days", 12.5)
    assert server._history_days() == 12.5


def test_usable_days_is_zero_before_the_first_worker_cycle(monkeypatch):
    """Not None, and not the archive's age: nothing is trainable yet."""
    monkeypatch.setitem(server._state, "source", _WithStore())
    monkeypatch.setitem(server._state, "usable_days", None)
    assert server._history_days() == 0.0


def test_a_retrain_is_refused_when_the_history_is_too_short(monkeypatch):
    monkeypatch.setattr(server, "_history_days", lambda: 1.0)
    with pytest.raises(HTTPException) as raised:
        server._start_background_train()
    assert raised.value.status_code == 409
    assert "days of observed presence" in raised.value.detail
    # Refused before the lock was taken, or the next attempt would 409 forever.
    assert not server._train_lock.locked()


def test_a_retrain_is_refused_while_one_is_running(monkeypatch):
    monkeypatch.setattr(server, "_history_days", lambda: 999.0)
    server._train_lock.acquire()
    with pytest.raises(HTTPException) as raised:
        server._start_background_train()
    assert raised.value.status_code == 409
    assert "already running" in raised.value.detail


def _wait_for_unlock(timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not server._train_lock.locked():
            return True
        time.sleep(0.02)
    return False


def test_a_retrain_runs_in_the_background_and_frees_the_lock(monkeypatch):
    """The train is held open on an event, because a lock held for the DURATION
    of the run is unobservable against a fake that has already finished."""
    started = threading.Event()
    finish = threading.Event()
    predicted = threading.Event()

    def slow_train():
        started.set()
        finish.wait(5)

    monkeypatch.setattr(server, "_history_days", lambda: 999.0)
    monkeypatch.setattr(server, "do_train", slow_train)
    monkeypatch.setattr(server, "do_predict", predicted.set)

    server._start_background_train()
    # Held in the CALLING thread, before the worker was even created, so that
    # `training_in_progress` is true from the moment the caller is answered.
    assert server._train_lock.locked()
    assert started.wait(5)
    assert server._train_lock.locked()

    finish.set()
    assert predicted.wait(5)
    assert _wait_for_unlock()


def test_a_failing_retrain_reports_itself_and_still_frees_the_lock(monkeypatch):
    """A train that dies holding the lock would refuse every later attempt with
    "already running" until the add-on was restarted."""
    def boom():
        raise RuntimeError("no feature table")

    monkeypatch.setattr(server, "_history_days", lambda: 999.0)
    monkeypatch.setattr(server, "do_train", boom)
    monkeypatch.setitem(server._state, "last_error", None)

    server._start_background_train()
    assert _wait_for_unlock()
    assert "no feature table" in (server._state["last_error"] or "")


# ---------------------------------------------------------------------------
# The crossing cuts a save is allowed to carry
# ---------------------------------------------------------------------------
# `crossing_patch` runs BEFORE `api_save_config` assigns anything, because the
# settings object it would assign to is the live one the predict cycle reads.

from occupancy_forecast import config as config_mod                      # noqa: E402
from occupancy_forecast import predict as predict_mod                    # noqa: E402
from occupancy_forecast.tests.conftest import settings as make_settings  # noqa: E402


@pytest.mark.parametrize("value", [-0.1, 0, 0.0, 1, 1.0, 1.5, "0.5", True, None])
def test_a_cut_outside_zero_to_one_is_refused(value):
    """Open at both ends: 0 and 1 can never be met by a rounded curve. `True` is
    in the list because JSON booleans arrive as Python ints."""
    with pytest.raises(HTTPException) as err:
        server.crossing_patch({"departure_threshold": value}, make_settings())
    assert err.value.status_code == 400


def test_the_away_cut_may_not_sit_above_the_home_cut():
    """Otherwise one forecast counts as both leaving and arriving."""
    with pytest.raises(HTTPException):
        server.crossing_patch(
            {"departure_threshold": 0.7, "arrival_threshold": 0.4}, make_settings())

    # The one-key case: raising only the away cut is exactly how the band gets
    # inverted without either number looking wrong on its own.
    current = make_settings(departure_threshold=0.5, arrival_threshold=0.5)
    with pytest.raises(HTTPException):
        server.crossing_patch({"departure_threshold": 0.7}, current)
    assert server.crossing_patch({"arrival_threshold": 0.7}, current) == {
        "arrival_threshold": 0.7}


@pytest.mark.parametrize("value", [0, 49, 1.5, True, "2", -1])
def test_a_minimum_run_must_be_whole_hours_inside_the_curve(value):
    with pytest.raises(HTTPException):
        server.crossing_patch({"crossing_min_hours": value}, make_settings())


def test_a_whole_number_run_is_coerced_to_int():
    patch = server.crossing_patch({"crossing_min_hours": 2.0}, make_settings())
    assert patch == {"crossing_min_hours": 2}
    assert isinstance(patch["crossing_min_hours"], int)


def test_a_run_may_reach_the_end_of_the_curve():
    limit = max(config_mod.HORIZONS_H)
    assert server.crossing_patch({"crossing_min_hours": limit},
                                 make_settings()) == {"crossing_min_hours": limit}


def test_a_patch_that_names_none_of_them_leaves_them_alone():
    assert server.crossing_patch({"people": ["person.alice"]}, make_settings()) == {}


# ---------------------------------------------------------------------------
# The identity fields a save is allowed to carry, and what a rejected save
# may NOT do
# ---------------------------------------------------------------------------
# The endpoint's schema is `payload: dict`, so every field is checked here
# before anything touches the LIVE settings object.

@pytest.mark.parametrize("payload", [
    {"people": "person.alice"},
    {"people": ["person.alice", 42]},
    {"people": ["device_tracker.phone"]},
    {"zones": "zone.office"},
    {"zones": ["person.alice"]},
    {"house_entity": ["group.household"]},
    {"source": "csv"},
    {"proximity": ["sensor.distance"]},
    {"next_alarm": "sensor.alarm"},
])
def test_a_field_of_the_wrong_shape_is_refused(payload):
    with pytest.raises(HTTPException) as raised:
        server.typed_patch(payload)
    assert raised.value.status_code == 400


def test_well_formed_identity_fields_pass_through():
    patch = server.typed_patch({
        "people": ["person.alice"], "zones": [], "house_entity": None,
        "source": "store", "day_schedule": "", "next_alarm": None,
    })
    assert patch == {"people": ["person.alice"], "zones": [], "house_entity": None,
                     "source": "store", "day_schedule": None, "next_alarm": None}
    assert server.typed_patch({"departure_threshold": 0.4}) == {}, \
        "the crossing cuts are crossing_patch's, not this one's"


@pytest.mark.parametrize("value", [1, 30.5, True, "30", -1])
def test_retention_must_be_whole_days_the_chart_can_reach(value):
    """One day is the case worth pinning: legal-looking, and it deletes a +48 h
    forecast before it can ever be scored."""
    with pytest.raises(HTTPException) as raised:
        server.typed_patch({"forecast_retention_days": value})
    assert raised.value.status_code == 400


def test_zero_is_keep_everything_and_has_no_upper_bound():
    floor = -(-max(config_mod.HORIZONS_H) // 24)
    assert server.typed_patch({"forecast_retention_days": 0}) == {
        "forecast_retention_days": 0}
    assert server.typed_patch({"forecast_retention_days": floor}) == {
        "forecast_retention_days": floor}
    patch = server.typed_patch({"forecast_retention_days": 3650.0})
    assert patch == {"forecast_retention_days": 3650}
    assert isinstance(patch["forecast_retention_days"], int)


class _FakeHA:
    """Just enough Home Assistant for a save: the live entity ids, and the
    core config `refresh_environment` re-reads."""

    def __init__(self, *entity_ids: str):
        self._ids = entity_ids

    def states(self):
        return [{"entity_id": e, "state": "home", "attributes": {}} for e in self._ids]

    def config(self):
        return {"time_zone": "Europe/Amsterdam", "country": "NL",
                "latitude": 52.0, "longitude": 4.5}


@pytest.mark.parametrize("payload", [
    {"people": []},                          # `configure` refuses an empty house
    {"people": ["person.nobody"]},           # not an entity HA has
    {"zones": ["zone.typo"]},
    {"people": "person.alice"},              # the wrong shape
    {"day_schedule": "schedule.none"},
])
def test_a_rejected_save_leaves_the_live_settings_alone(monkeypatch, payload):
    import copy

    live = make_settings()
    before = copy.deepcopy(live)
    monkeypatch.setitem(server._state, "settings", live)
    monkeypatch.setitem(server._state, "ha",
                        _FakeHA("person.alice", "person.bob", "zone.alice_office"))
    people_before = [s.entity_id for s in config_mod.PEOPLE]

    with pytest.raises(HTTPException) as raised:
        server.api_save_config(payload)
    assert raised.value.status_code == 400
    assert live == before, "the live object must not carry a rejected value"
    assert server._state["settings"] is live
    assert [s.entity_id for s in config_mod.PEOPLE] == people_before


class _RecordingClient:
    def __init__(self):
        self.published: list[tuple[str, str, bool]] = []

    def publish(self, topic, payload, retain=False, qos=0):
        self.published.append((topic, payload, retain))


def _accepted_save_setup(monkeypatch):
    saved = []
    monkeypatch.setattr(config_mod.Settings, "save",
                        lambda self, path=None: saved.append(self.people))
    monkeypatch.setattr(server.runtime, "build_source",
                        lambda settings, ha, store=None: object())
    live = make_settings()
    monkeypatch.setitem(server._state, "settings", live)
    monkeypatch.setitem(server._state, "source", object())
    monkeypatch.setitem(server._state, "forecast_log", object())
    monkeypatch.setitem(server._state, "ha",
                        _FakeHA("person.alice", "person.bob", "zone.alice_office"))
    monkeypatch.setattr(server, "_listener", None)
    monkeypatch.setattr(server, "_threads_started", True)     # never start real threads
    retrains = []
    monkeypatch.setattr(server, "_start_background_train", lambda: retrains.append(1))
    client = _RecordingClient()
    monkeypatch.setattr(server._broker, "client", lambda: client)
    return live, saved, retrains, client


def test_an_accepted_save_swaps_the_live_settings_and_writes_them(monkeypatch):
    live, saved, _retrains, _client = _accepted_save_setup(monkeypatch)

    answer = server.api_save_config({"people": ["person.bob"], "zones": []})

    assert answer == {"saved": True, "people": ["bob"]}
    assert server._state["settings"] is not live, "a copy was validated and swapped in"
    assert server._state["settings"].people == ["person.bob"]
    assert live.people == ["person.alice", "person.bob"], "the old object is untouched"
    assert saved == [["person.bob"]]


def test_removing_a_person_clears_their_retained_entities_and_retrains(monkeypatch):
    """Every payload is retained and nothing ever unpublished one, so a removed
    person kept their sensors forever. The retrain happens now because the models
    are about a different house."""
    _live, _saved, retrains, client = _accepted_save_setup(monkeypatch)

    server.api_save_config({"people": ["person.bob"]})

    cleared = {topic for topic, payload, retain in client.published
               if payload == "" and retain}
    assert cleared, "nothing was retracted"
    assert all("/alice" in t or "_alice_" in t for t in cleared), cleared
    assert any(t.endswith("/alice/state") for t in cleared)
    assert any(t.startswith("homeassistant/sensor/") and t.endswith("/config")
               for t in cleared)
    assert retrains == [1]


def test_a_save_that_changes_nobody_neither_retracts_nor_retrains(monkeypatch):
    _live, _saved, retrains, client = _accepted_save_setup(monkeypatch)
    server.api_save_config({"departure_threshold": 0.4})
    assert client.published == []
    assert retrains == []


def test_the_forecast_log_survives_a_save_that_changes_the_source(monkeypatch):
    """A save is how a refused start-up gets going and how `source` changes; the
    log is neither -- open it if start-up never got that far, leave it alone
    otherwise."""
    _accepted_save_setup(monkeypatch)
    held = server._state["forecast_log"]

    server.api_save_config({"departure_threshold": 0.4})
    assert server._state["forecast_log"] is held

    opened = object()
    monkeypatch.setattr(server.runtime, "forecast_log", lambda: opened)
    monkeypatch.setitem(server._state, "forecast_log", None)
    server.api_save_config({"departure_threshold": 0.5})
    assert server._state["forecast_log"] is opened


# ---------------------------------------------------------------------------
# Start-up that refuses, and the health check Supervisor can act on
# ---------------------------------------------------------------------------

def test_a_refused_bootstrap_still_hands_the_panel_something_to_edit(monkeypatch):
    """No people, a corrupt config.json, HA down at boot: none may exit the
    process, which would take down the panel that fixes the first two."""
    monkeypatch.setitem(server._state, "last_error", None)

    monkeypatch.setattr(server.runtime, "home_assistant",
                        lambda: (_ for _ in ()).throw(RuntimeError("no Home Assistant")))
    monkeypatch.setattr(config_mod.Settings, "load",
                        classmethod(lambda cls, path=None: (_ for _ in ()).throw(
                            ValueError("Expecting value: line 1 column 1"))))
    settings, ha, source, log = server._degraded_bootstrap(
        RuntimeError("no people configured"))

    assert ha is None and source is None and log is None
    assert isinstance(settings, config_mod.Settings) and settings.people == []
    assert "no people configured" in server._state["last_error"]


def test_health_is_503_while_stalled_and_200_otherwise(monkeypatch):
    """`watchdog:` in config.yaml points Supervisor here; a non-2xx is what
    makes it restart the add-on."""
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._state, "ha", object())
    monkeypatch.setitem(server._stall, "since", None)
    assert server.health().status_code == 200

    monkeypatch.setitem(server._stall, "since", "2026-09-01T18:34:52+00:00")
    assert server.health().status_code == 503

    monkeypatch.setitem(server._stall, "since", None)
    monkeypatch.setitem(server._state, "ha", None)        # unreachable at start-up
    assert server.health().status_code == 503


# ---------------------------------------------------------------------------
# The progress notification is sent on a transition, not every five minutes
# ---------------------------------------------------------------------------

class _Notifier:
    def __init__(self):
        self.notified: list[str] = []
        self.dismissed = 0

    def notify(self, title, message, notification_id):
        self.notified.append(message)

    def dismiss(self, notification_id):
        self.dismissed += 1


def test_the_still_learning_notification_is_not_re_raised_every_cycle(monkeypatch):
    """Re-creating it each cycle replaced it each cycle, so a dismissed
    notification was back within five minutes."""
    ha = _Notifier()
    monkeypatch.setitem(server._state, "ha", ha)
    monkeypatch.setattr(server, "_notified", None)
    days = {"n": 3.0}
    monkeypatch.setattr(server, "_history_days", lambda: days["n"])
    monkeypatch.setattr(server, "_shipping_horizons", lambda: 0)

    for _ in range(5):
        server._notify_progress()
    assert len(ha.notified) == 1, "five cycles on the same day, one notification"

    days["n"] = 4.2                                         # a new day
    server._notify_progress()
    server._notify_progress()
    assert len(ha.notified) == 2

    days["n"] = server.MIN_DAYS_TO_TRAIN + 1                # training, nothing ships
    server._notify_progress()
    server._notify_progress()
    assert len(ha.notified) == 3
    assert "No horizon beats" in ha.notified[-1]

    monkeypatch.setattr(server, "_shipping_horizons", lambda: 5)
    server._notify_progress()
    server._notify_progress()
    assert ha.dismissed == 1, "dismissed once when something ships, not every cycle"


# ---------------------------------------------------------------------------
# The worker watchdog
# ---------------------------------------------------------------------------
# A blocked thread leaves every health signal green: `last_error` is None
# because nothing raised, and both connections are up.

def test_a_moving_worker_is_never_called_stalled(monkeypatch):
    monkeypatch.setitem(server._heartbeat, "at", 1000.0)
    monkeypatch.setitem(server._stall, "since", None)
    assert not server.check_stall(now=1000.0 + server.STALL_SECONDS - 1)


def test_a_blocked_worker_is_reported_once_with_its_phase(monkeypatch, caplog):
    monkeypatch.setitem(server._heartbeat, "at", 1000.0)
    monkeypatch.setitem(server._heartbeat, "phase", "predict")
    monkeypatch.setitem(server._stall, "since", None)
    monkeypatch.setitem(server._stall, "acted", False)
    monkeypatch.setitem(server._stall, "count", 0)
    monkeypatch.setattr(server._broker, "close", lambda: None)
    dumped = []
    late = 1000.0 + server.STALL_SECONDS + 1

    assert server.check_stall(now=late, dump=lambda: dumped.append(1))
    assert server._stall["count"] == 1
    assert server._stall["phase"] == "predict"
    assert dumped == [1], "the stacks are the whole point; without them there is nothing to debug"
    assert any("STALLED in predict" in r.getMessage() for r in caplog.records)

    # Still stalled a minute later: no second report, no second dump.
    assert server.check_stall(now=late + 60, dump=lambda: dumped.append(1))
    assert server._stall["count"] == 1
    assert dumped == [1]


def test_recovery_clears_the_stall_and_re_arms(monkeypatch, caplog):
    monkeypatch.setitem(server._heartbeat, "at", 1000.0)
    monkeypatch.setitem(server._stall, "since", "2026-09-01T18:34:52+00:00")
    monkeypatch.setitem(server._stall, "phase", "predict")
    monkeypatch.setitem(server._stall, "acted", True)

    assert not server.check_stall(now=1000.0 + 1)
    assert server._stall["since"] is None
    assert not server._stall["acted"], "a second stall has to be reportable"
    assert any("recovered" in r.getMessage() for r in caplog.records)


def test_a_retrain_is_not_a_stall(monkeypatch):
    """A retrain legitimately holds the worker for minutes, so it is measured
    against TRAIN_STALL_SECONDS from when the lock was taken. The signal is the
    TRAIN LOCK, and this test says so by taking it."""
    monkeypatch.setitem(server._heartbeat, "at", 1000.0)
    monkeypatch.setitem(server._stall, "since", None)
    monkeypatch.setitem(server._stall, "acted", False)
    monkeypatch.setitem(server._stall, "count", 0)
    monkeypatch.setattr(server._broker, "close", lambda: None)
    late = 1000.0 + server.STALL_SECONDS * 10
    # A train that started well inside its own deadline, however stale the
    # worker's last beat.
    monkeypatch.setitem(server._train_started, "at", late - 60)

    with server._train_lock:
        assert not server.check_stall(now=late)
    assert server._stall["count"] == 0

    # And the other half: the same clock with nobody training IS a stall.
    # Without it this test passes against a `check_stall` that never fires.
    assert server.check_stall(now=late, dump=lambda: None)
    assert server._stall["count"] == 1


def test_a_train_that_overruns_its_own_deadline_is_a_stall(monkeypatch):
    """The exemption this replaces made a hung train the one failure the
    watchdog could not see: a pool that never returns holds the lock forever,
    and forever was exempt."""
    monkeypatch.setitem(server._heartbeat, "at", 1000.0)
    monkeypatch.setitem(server._stall, "since", None)
    monkeypatch.setitem(server._stall, "acted", False)
    monkeypatch.setitem(server._stall, "count", 0)
    monkeypatch.setattr(server._broker, "close", lambda: None)
    late = 1000.0 + server.TRAIN_STALL_SECONDS + 1
    monkeypatch.setitem(server._train_started, "at", 1000.0)
    dumped = []

    with server._train_lock:
        assert server.check_stall(now=late, dump=lambda: dumped.append(1))
    assert server._stall["count"] == 1
    assert server._stall["phase"] == "train"
    assert dumped == [1]


def test_taking_the_train_lock_stamps_when(monkeypatch):
    """Every acquisition goes through `_take_train_lock`, or the watchdog
    measures a train against whenever the previous one started."""
    monkeypatch.setitem(server._train_started, "at", 0.0)
    assert server._take_train_lock()
    assert server._train_started["at"] > 0.0
    assert not server._take_train_lock(), "held, so refused"
    server._train_lock.release()


def test_a_good_cycle_clears_the_error_a_bad_cycle_left_and_only_that(monkeypatch):
    """A sticky `last_error` makes "failing now" and "failed once" read the
    same. A train's error is deliberately NOT cleared by a later cycle."""
    monkeypatch.setitem(server._state, "last_error", None)
    monkeypatch.setattr(server, "_cycle_failed", False)

    server._record_error(RuntimeError("no row with a usable state_now"), from_cycle=True)
    assert "state_now" in server._state["last_error"]
    server._clear_cycle_error()
    assert server._state["last_error"] is None

    server._record_error(RuntimeError("every horizon failed to train"), from_cycle=False)
    server._clear_cycle_error()
    assert "failed to train" in server._state["last_error"]


def test_the_status_page_keeps_an_errors_stamp_and_drops_its_message(monkeypatch):
    """`/api/status` needs no login, so an exception's own text stays in `_state`
    and in the log. The ISO stamp survives because `panel/src/format.ts` splits
    on it."""
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._state, "last_error", None)
    monkeypatch.setitem(server._state, "last_error_public", None)
    monkeypatch.setattr(server, "_cycle_failed", False)

    server._record_error(OSError("no such file: /data/features.parquet"),
                         from_cycle=True)

    assert "/data/features.parquet" in server._state["last_error"], "kept for the log"
    served = server._status()
    assert "/data/features.parquet" not in json.dumps(served)
    assert server.log.SEE_THE_LOG in served["last_error"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T[\d:.+\-Z]+: ", served["last_error"]), \
        "the shape splitError() matches, or the Training card loses its clock"


def test_the_status_page_shows_the_worker_ageing(monkeypatch):
    """`seconds_since_phase` is the number that moves when nothing else does."""
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._heartbeat, "at", server.time.monotonic() - 42)
    monkeypatch.setitem(server._heartbeat, "phase", "collect")
    worker = server._status()["worker"]
    assert worker["phase"] == "collect"
    assert worker["seconds_since_phase"] >= 42
    assert worker["stalled"] is False


def test_the_heartbeat_is_hourly_and_unconditional(monkeypatch, caplog):
    """One line an hour when nothing has changed, so that SILENCE means something.
    Unconditional on health: a heartbeat that only appears when things are fine
    cannot be told apart from a stopped process."""
    caplog.set_level("INFO")
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._heartbeat, "said", 0.0)
    monkeypatch.setitem(server._heartbeat, "cycles", 12)

    assert server._say_alive(now=server.HEARTBEAT_SECONDS + 1)
    assert any("alive: 12 cycles" in r.getMessage() for r in caplog.records)

    # Not again until the hour is up.
    assert not server._say_alive(now=server.HEARTBEAT_SECONDS + 2)
    assert server._say_alive(now=server.HEARTBEAT_SECONDS * 2 + 3)


def test_the_heartbeat_says_so_when_mqtt_is_down(monkeypatch, caplog):
    """The numbers it carries are the ones a slow failure moves."""
    caplog.set_level("INFO")
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._heartbeat, "said", 0.0)
    monkeypatch.setattr(type(server._broker), "connected",
                        property(lambda self: False))
    server._say_alive(now=server.HEARTBEAT_SECONDS + 1)
    assert any("mqtt DOWN" in r.getMessage() for r in caplog.records)


# --- recording what was forecast ------------------------------------------

class _Recorder:
    """A store that only knows how to be written to."""

    def __init__(self, fail: bool = False):
        self.fail, self.rows, self.pruned = fail, [], []

    def append_forecasts(self, rows):
        if self.fail:
            raise OSError("attempt to write a readonly database")
        self.rows.extend(rows)
        return len(rows)

    def prune_forecasts(self, before):
        self.pruned.append(before)
        return 0


def _result(curve, observed_at="2026-09-02T20:30:00+00:00"):
    return {"subject": "alice", "observed_at": observed_at, "curve": curve}


def test_a_forecast_is_recorded_on_the_slot_it_was_about(monkeypatch):
    """+6 h from a row observed at 20:30 is about 02:30, not six hours after the
    cycle ran. The read side joins on equality, so half a slot out matches
    nothing."""
    store = _Recorder()
    monkeypatch.setitem(server._state, "forecast_log", store)

    server._record_forecasts([_result({6: 0.8})])

    (subject, target_ms, horizon, p), = store.rows
    assert (subject, horizon, p) == ("alice", 6, 0.8)
    assert dt.datetime.fromtimestamp(target_ms / 1000, dt.timezone.utc) == \
        dt.datetime(2026, 9, 3, 2, 30, tzinfo=dt.timezone.utc)


def test_an_unserved_horizon_writes_no_row(monkeypatch):
    """The absence IS the record. It is what makes the gap appear on the chart,
    and it is why this must not be helpfully backfilled with a null row."""
    store = _Recorder()
    monkeypatch.setitem(server._state, "forecast_log", store)

    server._record_forecasts([_result({1: 0.9, 2: 0.8})])

    assert sorted(row[2] for row in store.rows) == [1, 2]
    assert len(store.rows) == 2, "no row for the 46 horizons that were not served"


def test_the_retention_window_is_pruned_every_cycle(monkeypatch):
    store = _Recorder()
    monkeypatch.setitem(server._state, "forecast_log", store)
    monkeypatch.setitem(server._state, "settings", make_settings())

    server._record_forecasts([_result({6: 0.8})])

    assert len(store.pruned) == 1
    age = dt.datetime.now(dt.timezone.utc) - store.pruned[0]
    assert abs(age.days - config_mod.FORECAST_RETENTION_DAYS) <= 1


def test_the_window_pruned_is_the_one_the_setting_names(monkeypatch):
    """The whole point of making it settable: the number on the Setup tab has
    to be the number the pruner uses, not a constant that happens to match."""
    store = _Recorder()
    settings = make_settings()
    settings.forecast_retention_days = 7
    monkeypatch.setitem(server._state, "forecast_log", store)
    monkeypatch.setitem(server._state, "settings", settings)

    server._record_forecasts([_result({6: 0.8})])

    age = dt.datetime.now(dt.timezone.utc) - store.pruned[0]
    assert abs(age.days - 7) <= 1


def test_zero_days_prunes_nothing_at_all(monkeypatch):
    """Not "delete everything older than now", which is what a cutoff computed
    from 0 would mean and would erase the table on the first cycle."""
    store = _Recorder()
    settings = make_settings()
    settings.forecast_retention_days = 0
    monkeypatch.setitem(server._state, "forecast_log", store)
    monkeypatch.setitem(server._state, "settings", settings)

    server._record_forecasts([_result({6: 0.8})])

    assert len(store.rows) == 1, "still recorded"
    assert store.pruned == [], "and nothing was deleted"


def test_a_store_that_cannot_be_written_does_not_fail_the_serve_cycle(monkeypatch):
    """The house getting a forecast outranks the chart getting a data point: a
    full disk costs a gap on a panel card, not the prediction HA waits for."""
    monkeypatch.setitem(server._state, "forecast_log", _Recorder(fail=True))

    server._record_forecasts([_result({6: 0.8})])  # must not raise


def test_an_influx_installation_records_what_it_published(monkeypatch):
    """The record is the add-on's own output and has nothing to do with where
    history is read from. Reaching it through `source.store` meant an Influx
    install recorded nothing."""
    class Influx:
        pass

    store = _Recorder()
    monkeypatch.setitem(server._state, "source", Influx())
    monkeypatch.setitem(server._state, "forecast_log", store)

    server._record_forecasts([_result({6: 0.8})])

    assert len(store.rows) == 1


def test_a_start_up_that_has_no_log_yet_records_nothing_and_says_nothing(monkeypatch):
    """`_degraded_bootstrap` opens no files at all, so the worker can reach a
    cycle before there is anywhere to write."""
    monkeypatch.setitem(server._state, "forecast_log", None)
    server._record_forecasts([_result({6: 0.8})])  # must not raise


# ---------------------------------------------------------------------------
# Who may change something
# ---------------------------------------------------------------------------
# Tested against `require_admin` directly (no httpx). What the route decorators
# do with it is asserted separately by reading the app's route table.

def _request(headers: dict | None = None, path: str = "/train"):
    from starlette.requests import Request
    return Request({
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "server": ("testserver", 80),
        "root_path": "",
        "path": path,
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode())
                    for k, v in (headers or {}).items()],
    })


def test_an_empty_allowlist_lets_everyone_through(monkeypatch):
    """The default, and what every install had before the option existed: an
    upgrade must not lock the owner out of their own panel."""
    monkeypatch.delenv("OCCUPANCY_ADMIN_USERS", raising=False)
    assert server.require_admin(_request()) is None
    assert server.require_admin(_request({"X-Remote-User-Id": "anyone"})) is None


def test_a_listed_user_is_allowed(monkeypatch):
    monkeypatch.setenv("OCCUPANCY_ADMIN_USERS", "abc123,def456")
    request = _request({"X-Remote-User-Id": "def456"})
    assert server.require_admin(request) == "def456"


def test_an_unlisted_user_is_refused(monkeypatch):
    monkeypatch.setenv("OCCUPANCY_ADMIN_USERS", "abc123")
    with pytest.raises(HTTPException) as raised:
        server.require_admin(_request({"X-Remote-User-Id": "somebody_else"}))
    assert raised.value.status_code == 403


def test_no_header_is_refused_once_the_allowlist_is_set(monkeypatch):
    """No header means Ingress was bypassed or a proxy never set it; "cannot be
    named" must not mean "allowed"."""
    monkeypatch.setenv("OCCUPANCY_ADMIN_USERS", "abc123")
    with pytest.raises(HTTPException) as raised:
        server.require_admin(_request())
    assert raised.value.status_code == 403


def test_the_allowlist_is_read_per_request(monkeypatch):
    """The option can change while the add-on runs. A gate that cached the value
    it saw at import time would stop matching the Configuration tab."""
    monkeypatch.setenv("OCCUPANCY_ADMIN_USERS", "abc123")
    with pytest.raises(HTTPException):
        server.require_admin(_request({"X-Remote-User-Id": "later"}))
    monkeypatch.setenv("OCCUPANCY_ADMIN_USERS", "abc123, later ")
    assert server.require_admin(_request({"X-Remote-User-Id": "later"})) == "later"


def test_every_mutating_route_is_gated():
    """A new POST added without the dependency is invisible in review."""
    gated = {
        (path, method)
        for route in server.app.routes
        for path in [getattr(route, "path", None)]
        for method in getattr(route, "methods", None) or ()
        if path and any(d.dependency is server.require_admin
                        for d in getattr(route, "dependencies", ()))
    }
    posts = {
        (route.path, method)
        for route in server.app.routes
        for method in getattr(route, "methods", None) or ()
        if method == "POST"
    }
    # Named, not just counted: `posts == gated` is satisfied by two empty sets.
    assert {path for path, _ in posts} == {
        "/api/config", "/api/config/check", "/api/config/check-broker",
        "/collect", "/predict", "/train", "/reload"}
    assert posts == gated, f"ungated POST routes: {sorted(posts - gated)}"


# ---------------------------------------------------------------------------
# Settings that used to be add-on options. The panel owns them since 0.4.0, and
# two of them are secrets on an endpoint that is deliberately open.

def test_a_blank_secret_box_keeps_the_stored_one(monkeypatch):
    """The form is never given the stored value, so an empty box is "unchanged"
    -- treating it as "" would blank a token the user cannot see to retype."""
    live, _saved, _retrains, _client = _accepted_save_setup(monkeypatch)
    live.influx_token = "stored"
    live.mqtt_password = "secret"

    server.api_save_config({"people": ["person.alice"], "influx_token": "",
                            "mqtt_password": ""})

    assert server._state["settings"].influx_token == "stored"
    assert server._state["settings"].mqtt_password == "secret"


def test_a_typed_secret_replaces_it_and_null_forgets_it(monkeypatch):
    live, _saved, _retrains, _client = _accepted_save_setup(monkeypatch)
    live.influx_token = "stored"

    server.api_save_config({"people": ["person.alice"], "influx_token": "fresh"})
    assert server._state["settings"].influx_token == "fresh"

    server._state["settings"].influx_token = "fresh"
    monkeypatch.setitem(server._state, "settings", server._state["settings"])
    server.api_save_config({"people": ["person.alice"], "influx_token": None})
    assert server._state["settings"].influx_token == ""


def test_influx_without_credentials_is_a_400_not_a_quiet_fall_back(monkeypatch):
    """Falling back to the local archive would start a fresh empty one and look
    perfectly healthy for the ten days it takes to notice."""
    _accepted_save_setup(monkeypatch)

    with pytest.raises(HTTPException) as raised:
        server.api_save_config({"people": ["person.alice"], "source": "influx",
                                "influx_url": "", "influx_token": ""})
    assert raised.value.status_code == 400
    assert "not all set" in raised.value.detail


@pytest.mark.parametrize("payload", [
    {"influx_url": "192.0.2.10:8086"},        # no scheme
    {"mqtt_port": 0},
    {"mqtt_port": 70000},
    {"mqtt_port": 1883.5},
    {"mqtt_ssl": "true"},                      # the string, not the boolean
    {"influx_url": 42},
])
def test_a_connection_field_of_the_wrong_shape_is_refused(payload):
    with pytest.raises(HTTPException) as raised:
        server.typed_patch(payload)
    assert raised.value.status_code == 400


def test_the_source_option_no_longer_overrides_a_panel_choice(monkeypatch):
    """The bug this move exists to fix: `refresh_environment` runs on every save
    as well as every boot, so the old override reverted the edit being saved."""
    monkeypatch.setenv("OCCUPANCY_SOURCE", "influx")
    settings = make_settings(source="store")

    refreshed = runtime_mod.refresh_environment(settings, _FakeHA("person.alice"))

    assert refreshed.source == "store"


def test_the_add_on_options_are_imported_once_and_then_left_alone(monkeypatch):
    """Guarded on the marker, not on each field being empty: "every boot" would
    re-apply the option over a panel edit, and "only the first version" would
    miss anyone who updates straight past it."""
    monkeypatch.setenv("OCCUPANCY_SOURCE", "influx")
    monkeypatch.setenv("INFLUX_URL", "http://influx:8086")
    monkeypatch.setenv("INFLUX_TOKEN", "from-options")
    monkeypatch.setenv("OCCUPANCY_LEGACY_MQTT_HOST", "broker.example")
    monkeypatch.setenv("OCCUPANCY_LEGACY_MQTT_PORT", "8883")
    monkeypatch.setenv("OCCUPANCY_LEGACY_MQTT_SSL", "true")
    settings = make_settings()

    assert runtime_mod.import_legacy_options(settings) is True
    assert settings.source == "influx"
    assert settings.influx_token == "from-options"
    assert (settings.mqtt_host, settings.mqtt_port, settings.mqtt_ssl) == (
        "broker.example", 8883, True)
    assert settings.migrated_from_options

    settings.source = "store"
    assert runtime_mod.import_legacy_options(settings) is False
    assert settings.source == "store", "a panel edit must survive the next boot"


def test_supervisors_discovered_broker_is_never_frozen_into_the_settings(monkeypatch):
    """`MQTT_HOST` also carries Supervisor's own service. Importing that would
    pin today's address into config.json, where it would then win over the
    discovery that is meant to track it."""
    monkeypatch.setenv("MQTT_HOST", "core-mosquitto")
    monkeypatch.delenv("OCCUPANCY_LEGACY_MQTT_HOST", raising=False)
    settings = make_settings()

    runtime_mod.import_legacy_options(settings)

    assert settings.mqtt_host == ""
    assert config_mod.mqtt_settings(settings)["host"] == "core-mosquitto"


# --- the broker check ------------------------------------------------------

class _FakeMqttClient:
    """Records what `predict.check_connection` does to a client."""

    made: list = []
    # On the CLASS, so a test can flip it before the client is constructed.
    refuse = False

    def __init__(self, _api, client_id=None):
        self.client_id = client_id
        self.connected_to = None
        self.will = None
        self.tls = False
        self.auth = None
        _FakeMqttClient.made.append(self)

    def username_pw_set(self, user, password): self.auth = (user, password)
    def tls_set(self, *a, **k): self.tls = True
    def will_set(self, *a, **k): self.will = a
    def disconnect(self): pass

    def connect(self, host, port, keepalive=60):
        if self.refuse:
            raise ConnectionRefusedError("[Errno 111] Connection refused")
        self.connected_to = (host, port)


@pytest.fixture
def fake_mqtt(monkeypatch):
    _FakeMqttClient.made = []
    monkeypatch.setattr(predict_mod.mqtt, "Client", _FakeMqttClient)
    return _FakeMqttClient


def test_the_broker_check_never_reuses_the_live_client_id(fake_mqtt):
    """MQTT kicks the existing session on an id collision, silently and
    permanently -- so a check on the live id would disconnect the add-on's own
    publisher on every press. It must not set a will either: that would retract
    the entities on the way out."""
    settings = make_settings(mqtt_host="broker.example", mqtt_port=1883)

    result = predict_mod.check_connection(settings)

    probe = fake_mqtt.made[-1]
    assert result["ok"]
    assert probe.client_id != predict_mod.client_id()
    assert probe.client_id.startswith(predict_mod.client_id())
    assert probe.will is None, "a will would retract the live entities"
    assert probe.connected_to == ("broker.example", 1883)


def test_a_refused_broker_names_the_address_but_not_the_exception(fake_mqtt, monkeypatch):
    """The address is fine HERE -- this endpoint is admin-gated. What must not
    travel is the library's own text, which is what `/api/status` redacts."""
    monkeypatch.setattr(_FakeMqttClient, "refuse", True)
    settings = make_settings(mqtt_host="broker.example", mqtt_port=1883)

    result = predict_mod.check_connection(settings)

    assert result["ok"] is False
    assert "broker.example:1883" in result["detail"]
    assert "Errno 111" not in result["detail"]


def test_the_broker_check_falls_back_to_supervisors_service(fake_mqtt, monkeypatch):
    """An empty broker card means Supervisor's own broker, which is the usual
    install -- the check has to test that rather than refuse."""
    monkeypatch.setenv("MQTT_HOST", "core-mosquitto")
    monkeypatch.setenv("MQTT_PORT", "1883")

    result = predict_mod.check_connection(make_settings(mqtt_host=""))

    assert result["ok"] and result["host"] == "core-mosquitto:1883"
