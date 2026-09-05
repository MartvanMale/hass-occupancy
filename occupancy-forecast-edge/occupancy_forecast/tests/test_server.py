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
    span one would be too wide to catch anything else, so it is exempt.

    The exemption is the TRAIN LOCK, and this test says so by taking it. The
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

    with server._train_lock:
        assert not server.check_stall(now=late)
    assert server._stall["count"] == 0

    # And the other half: the same clock with nobody training IS a stall.
    # Without it this test passes against a `check_stall` that never fires.
    assert server.check_stall(now=late, dump=lambda: None)
    assert server._stall["count"] == 1


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
