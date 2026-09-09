"""The scheduling and the retrain guard.

Both are here rather than behind an HTTP client because the suite deliberately
ships no httpx: the endpoints are kept thin so that the parts worth testing are
plain functions. `_next_train` answers "when", which is what goes on the panel;
`_start_background_train` is the guard that has to refuse *before* it spawns,
because an HTTPException raised inside that thread reaches nobody.
"""

import datetime as dt
import threading
import time

import pytest
from fastapi import HTTPException

from occupancy_forecast import server


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


def test_a_retrain_is_refused_when_the_history_is_too_short(monkeypatch):
    monkeypatch.setattr(server, "_history_days", lambda: 1.0)
    with pytest.raises(HTTPException) as raised:
        server._start_background_train()
    assert raised.value.status_code == 409
    assert "days of history" in raised.value.detail
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
    """The caller is told it started; the work happens elsewhere.

    The train is held open on an event rather than returning at once, because
    the thing worth asserting -- that the lock is held for the DURATION of the
    run -- is unobservable against a fake that has already finished by the time
    the assertion executes.
    """
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
#
# `crossing_patch` runs BEFORE `api_save_config` assigns anything, because the
# settings object it would assign to is the live one the five-minute predict
# cycle reads. A rejected save must not leave the process publishing off a value
# that never reached disk.

from occupancy_forecast import config as config_mod                      # noqa: E402
from occupancy_forecast.tests.conftest import settings as make_settings  # noqa: E402


@pytest.mark.parametrize("value", [-0.1, 0, 0.0, 1, 1.0, 1.5, "0.5", True, None])
def test_a_cut_outside_zero_to_one_is_refused(value):
    """Open at both ends: 0 and 1 can never be met by a rounded curve, so they
    would leave the sensor permanently unknown rather than doing the obvious
    thing. `True` is in the list because JSON booleans arrive as Python ints."""
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
#
# The endpoint's whole schema is `payload: dict`. `{"people": "person.alice"}`
# used to reach `config.configure` as a string, which iterated it and minted
# twelve one-letter subjects. And every field was assigned onto the LIVE
# settings object before any of the checks ran, so a rejected save left the
# process running on values that never reached disk -- an empty people list
# stopped the collector while the forecasts carried on.

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
    """Every payload the add-on publishes is retained, and nothing ever
    unpublished one: a removed person kept their sensors in Home Assistant
    forever, holding the last forecast. And the models on disk are now about
    a different house, so the retrain happens now rather than at the next
    scheduled 04:00, which on a mature install is a week away."""
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


# ---------------------------------------------------------------------------
# Start-up that refuses, and the health check Supervisor can act on
# ---------------------------------------------------------------------------

def test_a_refused_bootstrap_still_hands_the_panel_something_to_edit(monkeypatch):
    """No people yet, a corrupt config.json, Home Assistant down at boot: each
    used to exit the process, taking down the panel that is the tool for
    fixing the first two."""
    monkeypatch.setitem(server._state, "last_error", None)

    monkeypatch.setattr(server.runtime, "home_assistant",
                        lambda: (_ for _ in ()).throw(RuntimeError("no Home Assistant")))
    monkeypatch.setattr(config_mod.Settings, "load",
                        classmethod(lambda cls, path=None: (_ for _ in ()).throw(
                            ValueError("Expecting value: line 1 column 1"))))
    settings, ha, source = server._degraded_bootstrap(RuntimeError("no people configured"))

    assert ha is None and source is None
    assert isinstance(settings, config_mod.Settings) and settings.people == []
    assert "no people configured" in server._state["last_error"]


def test_health_is_503_while_stalled_and_200_otherwise(monkeypatch):
    """`watchdog:` in config.yaml points Supervisor here; a non-2xx is what
    makes it restart the add-on. It always said 200 before."""
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
    """Re-creating it each cycle replaced it each cycle, so a user who
    dismissed it had it back within five minutes, for up to seven weeks."""
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
#
# On 2026-09-01 the add-on published nothing for 11.5 hours against a 5-minute
# cycle, and every health signal stayed green: `last_error` was None because
# the thread was blocked rather than raising, and both connections were up. The
# outage was found by looking at a chart. These pin the thing that would have
# said so.
# ---------------------------------------------------------------------------

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

    # Still stalled a minute later: no second report, no second dump. A
    # watchdog that logs every minute for eleven hours is one nobody reads.
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
    """It legitimately holds the worker for minutes. A threshold wide enough to
    span one would be too wide to catch anything else, so it is measured
    against its own, longer deadline instead -- TRAIN_STALL_SECONDS from the
    moment the lock was taken.

    The signal is the TRAIN LOCK, and this test says so by taking it. The
    version before it set `_state["training_in_progress"]` -- a key nothing in
    the add-on ever writes -- so the test was the only thing that had ever made
    the exemption fire, and it passed against a `check_stall` that would have
    dumped every thread's stack in the middle of a healthy retrain."""
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
    watchdog could not see: a worker pool that never returns holds the lock
    forever, and forever was exempt. Now it is a stall in phase `train`, with
    the stacks dumped -- and the occupancy-train thread is among them."""
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
    """`last_error` used to be sticky: one transient failure stayed on the
    status page until a restart, so "failing now" and "failed once" read the
    same. A train's error is deliberately NOT cleared by a later cycle -- a
    failed 04:00 train is worth seeing at breakfast."""
    monkeypatch.setitem(server._state, "last_error", None)
    monkeypatch.setattr(server, "_cycle_failed", False)

    server._record_error(RuntimeError("no row with a usable state_now"), from_cycle=True)
    assert "state_now" in server._state["last_error"]
    server._clear_cycle_error()
    assert server._state["last_error"] is None

    server._record_error(RuntimeError("every horizon failed to train"), from_cycle=False)
    server._clear_cycle_error()
    assert "failed to train" in server._state["last_error"]


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
    """One line an hour when nothing has changed, so that SILENCE means
    something. Before it, a working add-on and a hung one wrote identical
    logs -- two lines per start and nothing else -- which is how 11.5 hours of
    nothing went unnoticed.

    Unconditional on health on purpose: a heartbeat that only appears when
    things are fine cannot be told apart from a stopped process."""
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


class _Source:
    def __init__(self, store):
        self.store = store


def _result(curve, observed_at="2026-09-02T20:30:00+00:00"):
    return {"subject": "alice", "observed_at": observed_at, "curve": curve}


def test_a_forecast_is_recorded_on_the_slot_it_was_about(monkeypatch):
    """+6 h from a row observed at 20:30 is about 02:30, not about 'six hours
    after whenever this cycle happened to run'. The join on the read side is an
    equality, so an anchor half a slot out would line nothing up ever."""
    store = _Recorder()
    monkeypatch.setitem(server._state, "source", _Source(store))

    server._record_forecasts([_result({6: 0.8})])

    (subject, target_ms, horizon, p), = store.rows
    assert (subject, horizon, p) == ("alice", 6, 0.8)
    assert dt.datetime.fromtimestamp(target_ms / 1000, dt.timezone.utc) == \
        dt.datetime(2026, 9, 3, 2, 30, tzinfo=dt.timezone.utc)


def test_an_unserved_horizon_writes_no_row(monkeypatch):
    """The absence IS the record. It is what makes the gap appear on the chart,
    and it is why this must not be helpfully backfilled with a null row."""
    store = _Recorder()
    monkeypatch.setitem(server._state, "source", _Source(store))

    server._record_forecasts([_result({1: 0.9, 2: 0.8})])

    assert sorted(row[2] for row in store.rows) == [1, 2]
    assert len(store.rows) == 2, "no row for the 46 horizons that were not served"


def test_the_retention_window_is_pruned_every_cycle(monkeypatch):
    store = _Recorder()
    monkeypatch.setitem(server._state, "source", _Source(store))

    server._record_forecasts([_result({6: 0.8})])

    assert len(store.pruned) == 1
    age = dt.datetime.now(dt.timezone.utc) - store.pruned[0]
    assert abs(age.days - config_mod.FORECAST_RETENTION_DAYS) <= 1


def test_a_store_that_cannot_be_written_does_not_fail_the_serve_cycle(monkeypatch):
    """The house getting a forecast outranks the chart getting a data point. A
    full disk or a read-only database must cost a gap on a panel card, not the
    prediction Home Assistant is waiting for."""
    monkeypatch.setitem(server._state, "source", _Source(_Recorder(fail=True)))

    server._record_forecasts([_result({6: 0.8})])  # must not raise


def test_an_influx_installation_records_nothing_and_says_nothing(monkeypatch):
    """No store to write to, and that is not an error -- it is a configuration
    in which this card is honestly unavailable."""
    class Storeless:
        pass

    monkeypatch.setitem(server._state, "source", Storeless())
    server._record_forecasts([_result({6: 0.8})])  # must not raise


# ---------------------------------------------------------------------------
# Who may change something
# ---------------------------------------------------------------------------
#
# Tested against `require_admin` directly rather than over HTTP, for the reason
# in this module's docstring: the suite ships no httpx. What the route
# decorators do with it -- `dependencies=admin_only` on the five POSTs -- is
# asserted separately by reading the app's own route table, which is the part a
# new endpoint can silently get wrong.

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
    """The default, and what every install had before the option existed. An
    upgrade that locked the owner out of their own panel would be a worse bug
    than the one this fixes."""
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
    """The only ways to arrive without the header are to bypass Ingress or to
    sit behind a proxy that never set it. Neither is a user we can name, and
    "cannot be named" must not mean "allowed"."""
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
    """The list is the point. A new POST added without the dependency is exactly
    the regression this file exists to catch, and it is invisible in review."""
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
    # Named, not just counted: `posts == gated` is satisfied by two empty sets,
    # so a refactor that renamed every endpoint would pass a test that only
    # compared them to each other.
    assert {path for path, _ in posts} == {
        "/api/config", "/collect", "/predict", "/train", "/reload"}
    assert posts == gated, f"ungated POST routes: {sorted(posts - gated)}"
