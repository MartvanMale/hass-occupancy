"""The add-on: collector, scheduler, HTTP API and the configuration UI.

Handlers are sync `def`, not `async def`. FastAPI runs sync handlers in a
threadpool; an `async def` doing blocking history reads and a scikit-learn fit
would occupy the event loop and hang `/health` at exactly the moment you need it
to answer.

Three things run on their own:

  collector   every COLLECT_MINUTES, pull new states from Home Assistant into
              the store and sample synthesised distances. This is what makes the
              add-on work on an install with ten days of recorder -- it
              accumulates its own history from the moment it is switched on.
  predictor   after each collection, republish the forecast.
  trainer     weekly, and on demand.

ADVISORY ONLY. Nothing here calls a Home Assistant service that changes
anything; the only writes are MQTT sensor states and persistent notifications.
"""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import json
import faulthandler
import logging
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from . import (config, departure, discover, eta as eta_mod, evaluate, explore,
               features, listen, log, night)
from . import outing as outing_mod, predict as predict_mod, runtime
from . import train as train_mod
from . import web

_log = log.get(__name__)

PORT = 8099

COLLECT_MINUTES = 5

# Floor between two cycles, however many events arrive in between. Coming home
# fires the person entity, the zone and the device_tracker within a second or
# two of each other, and EACH cycle is a `predict.LOOKBACK_DAYS` feature
# rebuild -- 32 days, which on the `influx` source means a 32-day Flux query.
# This is load protection, not politeness: without it one arrival runs three
# rebuilds back to back.
MIN_CYCLE_SECONDS = 60
TRAIN_WEEKDAY = 0          # Monday
TRAIN_HOUR = 4

# How long the worker may go without reaching its next phase before the
# watchdog calls it stalled.
#
# MEASURED, the hard way: on 2026-09-01 the add-on restarted at 18:34:50 UTC,
# published two forecasts, and then published NOTHING for 11.5 hours -- against
# a COLLECT_MINUTES of 5, so roughly 140 missed cycles. Nothing showed it. The
# worker wraps every cycle in `except Exception` and records `last_error`, and
# `last_error` was None, because the thread was not raising: it was BLOCKED.
# `/health` said `mqtt.connected: true`, `listener.connected: true`, no error --
# a hung add-on and a healthy one were the same page. The only tell was
# `last_predict` quietly ageing, and the outage was found days later by looking
# at a chart and wondering why the line was flat.
#
# Three cycles, so a slow-but-moving box is never called stalled. A retrain
# gets its OWN deadline rather than an exemption: it legitimately takes
# minutes, so the cycle threshold cannot span it, but an exemption meant a
# train that hung -- a worker pool that never returns -- was the one failure
# the watchdog could not see, and it is the in-cycle train that holds the
# worker thread. An hour is twenty times the measured ~190 s and still a
# bound. Measured from the moment the lock was taken, whichever thread took it.
STALL_SECONDS = COLLECT_MINUTES * 60 * 3
TRAIN_STALL_SECONDS = 60 * 60
WATCHDOG_SECONDS = 60

# How often the add-on says it is alive when nothing has changed.
#
# The point is to make SILENCE mean something. Before this the log carried two
# lines per start and nothing else, so eleven hours of nothing looked exactly
# like eleven hours of working -- see log.py. Hourly, because that is quiet
# enough to leave the Log tab readable and short enough that a stall is never
# more than an hour from being obvious. Per-cycle detail is a DEBUG line, which
# the add-on's own log_level option now turns on.
HEARTBEAT_SECONDS = 3600

# Below this there is not enough history for even a tapered fold geometry, and
# `calendar_folds` would return nothing. Above it the add-on trains, but early
# models are weak and mostly do not clear the ship gate -- which is the point:
# a horizon the model has not earned publishes nothing, so training early
# cannot make any published number worse and lets people watch horizons come
# online one at a time.
MIN_DAYS_TO_TRAIN = evaluate.MIN_TRAINABLE_DAYS

# Retrain weekly once the history is mature, but daily while it is still short:
# a fresh install changes noticeably from day to day, and a week between
# retrains is a week of looking at a model that is already out of date.
FULL_HISTORY_DAYS = evaluate.FULL_GEOMETRY_DAYS

def notify_collecting_id() -> str:
    """The persistent_notification id for the "still learning" notice.

    A FUNCTION, not a module constant, and derived from the slug like every
    other identity string here. As a constant it was the one thing that did not
    separate stable from edge: both add-ons raised and dismissed the same
    notification id with the same title, so edge -- which starts from an empty
    archive -- would re-raise "still learning" over stable's dismissal, and
    later dismiss stable's. Nothing errors; the notification just moves around.

    It cannot be a module constant even now: `topic_prefix()` calls Supervisor,
    and evaluating that at import time would put a network round trip in the
    import graph and make the tests need a Supervisor.
    """
    return f"{config.topic_prefix()}_collecting"


def code_fingerprint(package_dir: Path | None = None) -> str:
    """A short hash over the package sources; answers "is this the code on disk"."""
    package_dir = package_dir or Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


_CODE_FINGERPRINT = code_fingerprint()
_IMPORTED_AT = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

_state: dict = {
    "settings": None, "ha": None, "source": None,
    "models": {}, "eta_models": {}, "out_routine": {},
    "loaded_at": None, "last_collect": None, "last_predict": None,
    "last_train": None, "last_train_seconds": None, "training_started_at": None,
    "last_error": None, "forecast": [],
    # (computed_at, {state: count}). The scan is a full-history read of every
    # person, and the status page polls every few seconds -- see _unmatched.
    "unmatched_zones": (None, {}),
}
_broker = predict_mod.Broker()
_train_lock = threading.Lock()
# When `_train_lock` was last taken, so the watchdog can measure a train
# against TRAIN_STALL_SECONDS. Stamped by `_take_train_lock`, which is the only
# way the lock is meant to be acquired.
_train_started = {"at": 0.0}
# Whether `last_error` was written by the five-minute cycle, as opposed to a
# train. A later good cycle clears its own error and leaves a train's alone:
# a failed 04:00 train is worth seeing at breakfast, a hiccup at 04:05 is not.
_cycle_failed = False
# The worker's pulse: when it last STARTED a phase, and which one. Written by
# the worker, read by the watchdog thread -- which has to be a separate thread,
# because the failure being watched for is the worker being unable to run.
_heartbeat = {"at": time.monotonic(), "phase": "starting", "cycles": 0,
              "said": 0.0}
# What the watchdog has seen. `since` is None whenever the worker is moving.
_stall = {"count": 0, "since": None, "phase": None, "acted": False}
# Set by the Home Assistant trigger subscription to wake the worker early. See
# `_wait_for_work` for why the periodic poll stays regardless.
_nudge = threading.Event()
_listener: listen.Listener | None = None
_stop = threading.Event()


# ---------------------------------------------------------------------------
# Work
# ---------------------------------------------------------------------------

def _load_models() -> None:
    _state["models"] = predict_mod.load_models(config.MODELS_DIR)
    _state["eta_models"] = eta_mod.load_models(config.MODELS_DIR)
    _state["out_routine"] = outing_mod.load_routine(config.MODELS_DIR)
    _state["loaded_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _history_days() -> float:
    source = _state["source"]
    store = getattr(source, "store", None)
    return _span(store)["days"] if store else float("inf")


def do_collect() -> dict:
    """Pull new history, and sample a distance for anyone without Proximity."""
    settings, ha, source = _state["settings"], _state["ha"], _state["source"]
    store = getattr(source, "store", None)
    if store is None:
        return {"skipped": "influx source keeps its own history"}

    result = source.collect(runtime.tracked_entities(settings),
                            absence_is_a_reading=runtime.absence_entities(settings))
    synthetic = discover.sample_distances(ha, settings)
    result["synthetic"] = store.append(synthetic)
    _state["last_collect"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return result


def _record_forecasts(results: list[dict]) -> None:
    """Keep what was published, so it can be scored against what happened.

    A HORIZON WITH NO FORECAST WRITES NO ROW. The absence is the record: it is
    what makes the gap appear on the verification chart, which is the serving
    rule made visible over time rather than only in this cycle's strip.

    The target slot is measured from `observed_at` -- the feature row the model
    actually predicted from -- and not from the wall clock. That row is already
    on the 30-minute grid, so a whole number of hours lands exactly on a later
    slot and the join against observed truth is an equality rather than a
    tolerance. Anchoring on `now` would scatter target times across the grid by
    up to half a slot and nothing would ever line up.

    Wrapped whole: this is bookkeeping for a chart, and a store that has gone
    read-only or filled its disk must not stop the house getting a forecast.
    """
    store = getattr(_state["source"], "store", None)
    if store is None:
        return
    try:
        rows = []
        for result in results:
            observed_at = result.get("observed_at")
            if not observed_at:
                continue
            anchor = dt.datetime.fromisoformat(observed_at)
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=dt.timezone.utc)
            for horizon, value in (result.get("curve") or {}).items():
                target = anchor + dt.timedelta(hours=int(horizon))
                rows.append((result["subject"], int(target.timestamp() * 1000),
                             int(horizon), float(value)))
        store.append_forecasts(rows)
        store.prune_forecasts(
            dt.datetime.now(dt.timezone.utc)
            - dt.timedelta(days=config.FORECAST_RETENTION_DAYS))
    except Exception:
        _log.warning("could not record this cycle's forecasts; the verification "
                     "chart will show a gap here", exc_info=True)


def do_predict() -> list[dict]:
    results = predict_mod.run_cycle(
        _state["models"], _broker.client(), _state["source"], _state["eta_models"],
        _state["out_routine"])
    _state["forecast"] = results
    _state["last_predict"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _record_forecasts(results)
    return results


def _too_little_history() -> HTTPException | None:
    """The 409 that says a train cannot be attempted yet, or None.

    Split out of `do_train` so that the background variant can refuse *before*
    it spawns a thread. An HTTPException raised inside that thread reaches
    nobody -- the response has long since been sent.
    """
    days = _history_days()
    if days >= MIN_DAYS_TO_TRAIN:
        return None
    return HTTPException(
        status_code=409,
        detail=f"only {days:.1f} days of history; {MIN_DAYS_TO_TRAIN} are needed "
               f"before there is enough to hold out a test window. No forecast "
               f"is published until then.")


def do_train() -> dict:
    refusal = _too_little_history()
    if refusal is not None:
        raise refusal

    # Timed end to end rather than around `train_all` alone: the feature table
    # and the ETA models are built either side of it, and what a person waiting
    # on the button experiences is the total.
    started = time.monotonic()
    _state["training_started_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds")
    # Announced, because it is the one thing the add-on does that takes minutes
    # and pins the box. Without it a slow retrain and a hung worker look the
    # same from outside -- and the watchdog exempts training, so the log is the
    # only place that distinction is recorded.
    _log.info("training started (%.0f days of history)", _history_days())
    # Timed per stretch as well as end to end. `train_all` keeps its own, finer
    # breakdown; these four are what says whether the answer is in the fits at
    # all -- the feature rebuild is 1,900 pandas joins and looks like the
    # expensive one until it is measured.
    phases = train_mod.Phases()
    source = _state["source"]
    with phases("features"):
        table = features.build(source)
        features.write(table, config.FEATURES_PATH)
    with phases("models"):
        summary = train_mod.train_all(config.FEATURES_PATH, config.MODELS_DIR)
    try:
        with phases("eta"):
            eta_summary = eta_mod.train_all(source, config.MODELS_DIR)
    except Exception as err:  # noqa: BLE001
        _log.error("eta training failed: %s", err)
        eta_summary = {}
    # The out routine is arithmetic over the table already in hand, so it
    # costs a second and needs no source read. Guarded like the ETA models
    # beside it: a household with no zones configured has nothing to answer,
    # and that must not fail a training run.
    try:
        with phases("out routine"):
            labelled = outing_mod.label_out_days(table, departure.label_days(table))
            routine = outing_mod.fit_routine(labelled)
            outing_mod.save_routine(routine, config.MODELS_DIR)
        _log.info("out routine fitted for %d person(s)", len(routine))
    except Exception as err:  # noqa: BLE001
        _log.error("out routine failed: %s", err)
    _load_models()
    elapsed = time.monotonic() - started
    train_mod.stamp_duration(elapsed, config.MODELS_DIR)
    _state["last_train"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _state["last_train_seconds"] = round(elapsed, 1)
    _log.info("training finished in %.0fs (%s): %d/%d horizons ship a model "
              "(%d dedicated, %d pooled)", elapsed, phases.line(),
              _shipping_horizons(), len(config.HORIZONS_H),
              sum(1 for a in _state["models"].values()
                  if a.get("metrics", {}).get("kind") == "dedicated"),
              sum(1 for a in _state["models"].values()
                  if a.get("metrics", {}).get("kind") == "pooled"))
    return {"horizons": summary, "eta": eta_summary,
            "duration_s": round(elapsed, 1),
            "feature_rows": len(table),
            "labelled_rows": int(table["home_frac"].notna().sum())}


def _next_train(now: dt.datetime, days: float) -> str | None:
    """When the worker will next retrain, or None if it cannot yet.

    A separate function from the `due` test in `_worker` and deliberately so:
    that one answers "is it now", this one answers "when", and only the second
    is something to put on a page. They share the constants, which is what keeps
    them honest -- change TRAIN_HOUR and both move.

    Local time, because the schedule is: the worker compares against
    `datetime.now(config.tzinfo())`, the household's zone. The offset travels
    with the string so the browser renders it in the same clock the user's
    house runs on.
    """
    if days < MIN_DAYS_TO_TRAIN:
        return None
    candidate = now.replace(hour=TRAIN_HOUR, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += dt.timedelta(days=1)
    if days >= FULL_HISTORY_DAYS:
        # Mature: weekly rather than daily, so walk on to the training weekday.
        candidate += dt.timedelta(days=(TRAIN_WEEKDAY - candidate.weekday()) % 7)
    return candidate.isoformat(timespec="minutes")


def _start_background_train() -> None:
    """Kick off a train and return immediately.

    The synchronous endpoint holds the request open for the whole run -- fine for
    a script, useless for a button, because it is minutes of work and an Ingress
    proxy will give up long before it finishes. Both refusals therefore have to
    happen HERE, in the request thread: an HTTPException raised inside the worker
    below reaches nobody, the response having gone long ago.

    The lock is acquired here and released there. That is legal for a plain Lock
    and it is the point: `training_in_progress` has to be true from the moment
    the caller is told the train started, not from whenever the thread gets
    scheduled.
    """
    refusal = _too_little_history()
    if refusal is not None:
        raise refusal
    if not _take_train_lock():
        raise HTTPException(status_code=409, detail="a train is already running")

    def run() -> None:
        try:
            do_train()
            # Publish with the new models at once rather than waiting up to five
            # minutes for the next cycle, exactly as the worker does.
            do_predict()
        except Exception as err:  # noqa: BLE001
            _record_error(err, from_cycle=False)
            _log.error("train failed: %s", err)
        finally:
            _train_lock.release()

    try:
        threading.Thread(target=run, name="occupancy-train", daemon=True).start()
    except Exception:
        # A thread that never started never reaches the `finally` above, and a
        # lock held by nobody is a 409 forever plus a watchdog measuring a
        # train that does not exist.
        _train_lock.release()
        raise


def _take_train_lock() -> bool:
    """Acquire `_train_lock` without blocking, stamping when. False if held."""
    if not _train_lock.acquire(blocking=False):
        return False
    _train_started["at"] = time.monotonic()
    return True


def _record_error(err: Exception, from_cycle: bool) -> None:
    global _cycle_failed
    _state["last_error"] = f"{dt.datetime.now(dt.timezone.utc).isoformat()}: {err}"
    _cycle_failed = from_cycle


def _clear_cycle_error() -> None:
    """A good cycle clears the error a bad cycle left, and only that one."""
    global _cycle_failed
    if _cycle_failed:
        _state["last_error"] = None
        _cycle_failed = False


def _shipping_horizons() -> int:
    """How many horizons the model has actually earned. The number that moves."""
    return sum(1 for a in _state["models"].values()
               if a.get("metrics", {}).get("ships"))


_notify_error: str | None = None
# What the notification last said, so it is sent on a TRANSITION and not every
# five minutes. Re-creating it each cycle replaced it each cycle, so a user
# who dismissed it had it back within five minutes, for up to seven weeks.
_notified: tuple | None = None


def _notify_progress() -> None:
    """Tell the user why nothing is published yet, once, and clear it later."""
    global _notify_error, _notified
    # Both the id and the title carry the add-on's own name, so that stable and
    # edge raise two separate notifications and the reader can tell them apart
    # -- which means neither may be raised under a GUESSED name. See
    # config.resolve_topic_prefix.
    if not config.topic_prefix_resolved():
        return
    ha, days = _state["ha"], _history_days()
    notify_id, name = notify_collecting_id(), config.display_name()
    if days < MIN_DAYS_TO_TRAIN:
        # The day count is in the text, so a new day is a new message; that
        # is once a day, which is the cadence a progress note deserves.
        state: tuple = ("collecting", int(days))
    elif not _shipping_horizons():
        state = ("training",)
    else:
        state = ("published",)
    if state == _notified:
        return
    try:
        if state[0] == "collecting":
            ha.notify(
                f"{name} is still learning",
                f"Collected **{days:.0f} of {MIN_DAYS_TO_TRAIN} days** of history. "
                f"No forecast is published yet -- the sensors exist and read "
                f"unknown until a model has earned a horizon.",
                notify_id)
        elif state[0] == "training":
            # Training now, but nothing has beaten its baseline yet. Say so
            # rather than going quiet -- silence here reads as "broken".
            ha.notify(
                f"{name} is still learning",
                f"Training on **{days:.0f} days** of history. No horizon beats "
                f"its baseline yet, so nothing is published. This improves as "
                f"history accumulates.",
                notify_id)
        else:
            ha.dismiss(notify_id)
        _notified = state
        _notify_error = None
    except Exception as err:  # noqa: BLE001
        # Never worth failing a cycle for, but worth one line per distinct
        # failure: a token or proxy that has stopped working is otherwise
        # invisible here, and this is the same token every other HA call uses.
        if str(err) != _notify_error:
            _log.warning("could not update the progress notification: %s", err)
        _notify_error = str(err)


def _wait_for_work(since: float) -> None:
    """Block until the poll is due, or Home Assistant says something changed.

    `since` is the `time.monotonic()` at which the cycle that just finished
    started, which is what MIN_CYCLE_SECONDS is measured from.

    The periodic poll stays even with a healthy listener, and is not a
    fallback: it covers everything no state change announces -- the 30-minute
    slot turning over, the daily train check, and an install where the
    subscription never connected at all.

    Waited in one-second slices because there is no wait-on-either-event
    primitive: the version of this that blocked on `_nudge` for the full five
    minutes ignored `_stop` for just as long, which is a shutdown that hangs.
    """
    deadline = since + COLLECT_MINUTES * 60
    while not _stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if not _nudge.wait(timeout=min(remaining, 1.0)):
            continue                                 # slice expired; re-check
        _nudge.clear()
        # Debounce. Anything that arrives during this sleep sets the event
        # again and is answered by the single cycle that follows, which is the
        # point: one arrival, one rebuild.
        elapsed = time.monotonic() - since
        if elapsed < MIN_CYCLE_SECONDS and _stop.wait(MIN_CYCLE_SECONDS - elapsed):
            return
        _nudge.clear()
        return


def beat(phase: str) -> None:
    """Mark the worker as having reached a new phase.

    Called before each step rather than once per cycle, so a stall report names
    the step that hung instead of only the cycle that did not finish.
    """
    _heartbeat["at"] = time.monotonic()
    _heartbeat["phase"] = phase


def stall_seconds(now: float | None = None) -> float:
    """How long the worker has been in one phase."""
    return (time.monotonic() if now is None else now) - _heartbeat["at"]


def check_stall(now: float | None = None, dump=None) -> bool:
    """Is the worker stuck, and if it has just got stuck, say so loudly.

    Split out from the thread that calls it so the decision is testable without
    waiting on a real clock; `now` is a `time.monotonic()` reading and `dump`
    is the stack dumper, injected for the same reason.

    A retrain is measured against its own deadline, TRAIN_STALL_SECONDS, from
    the moment `_train_lock` was taken -- by the worker at 04:00 or by the
    Train button's thread, it makes no difference. It used to be EXEMPT while
    the lock was held (and before that the exemption read
    `_state["training_in_progress"]`, a key nothing ever wrote, so it was
    dead). An exemption meant the one failure this watchdog exists for -- a
    thread that never comes back -- was invisible for exactly as long as a
    train held the lock, which for a wedged worker pool is forever.

    Reports the transition, never the state: one report per episode, and one
    when it recovers. A watchdog that logs every minute for eleven hours is a
    watchdog nobody reads.
    """
    now = time.monotonic() if now is None else now
    if _train_lock.locked():
        late, limit, phase = now - _train_started["at"], TRAIN_STALL_SECONDS, "train"
    else:
        late, limit, phase = stall_seconds(now), STALL_SECONDS, _heartbeat["phase"]
    if late < limit:
        if _stall["since"] is not None:
            _log.warning("worker recovered; it was stuck in %s", _stall["phase"])
            _stall.update(since=None, phase=None, acted=False)
        return False

    if not _stall["acted"]:
        _stall["count"] += 1
        _stall["since"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        _stall["phase"] = phase
        _stall["acted"] = True
        _log.critical(
            "worker STALLED in %s for %.0fs after %d cycles. Thread stacks "
            "follow -- the frame at the top of occupancy-worker (or "
            "occupancy-train) is where it is stuck.",
            phase, late, _heartbeat["cycles"])
        try:
            (dump or _dump_stacks)()
        except Exception as err:  # noqa: BLE001
            _log.error("could not dump stacks: %s", err)
        # Best-effort nudge, not a diagnosis. The MQTT client is the only piece
        # the worker touches that can be freed from another thread, and closing
        # its socket raises inside a blocked publish rather than leaving it
        # parked. `Broker.client()` reconnects on the next cycle by design.
        try:
            _broker.close()
            _log.warning("dropped the MQTT client; it reconnects next cycle")
        except Exception as err:  # noqa: BLE001
            _log.error("could not drop the MQTT client: %s", err)
    return True


def _dump_stacks() -> None:
    """Every thread's stack, to the add-on log.

    `faulthandler` rather than `traceback`: it prints the C-level frame too, so
    a thread parked in a socket read is distinguishable from one spinning in
    Python, which is exactly the distinction needed here.
    """
    faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
    sys.stdout.flush()


def _watchdog() -> None:
    """Watch the worker from outside it. See STALL_SECONDS for what happened."""
    while not _stop.wait(WATCHDOG_SECONDS):
        try:
            check_stall()
        except Exception as err:  # noqa: BLE001
            _log.error("watchdog: %s", err)


def _say_alive(now: float | None = None) -> bool:
    """One INFO line an hour when nothing else has been worth saying.

    Deliberately unconditional on health: a heartbeat that only appears when
    things are good is a heartbeat you cannot distinguish from a stopped
    process. It carries the numbers that would show a slow failure -- the cycle
    count moving, how many horizons still ship, whether the two connections are
    up -- so a glance at the last line answers "is it working" without the API.
    """
    now = time.monotonic() if now is None else now
    if now - _heartbeat["said"] < HEARTBEAT_SECONDS:
        return False
    _heartbeat["said"] = now
    listener = (_listener.status if _listener else {}) or {}
    _log.info("alive: %d cycles, %d/%d horizons shipping, mqtt %s, "
              "listener %s, last predict %s",
              _heartbeat["cycles"], _shipping_horizons(), len(config.HORIZONS_H),
              "up" if _broker.connected else "DOWN",
              "up" if listener.get("connected") else "down",
              _state["last_predict"] or "never")
    return True


def _worker() -> None:
    """Collect, predict, and retrain on schedule. One thread, no scheduler library."""
    last_train_day = None
    while not _stop.is_set():
        started = time.monotonic()
        try:
            # Keep asking until Supervisor answers. Nothing publishes and no
            # notification is raised until it does; see config.resolve_topic_prefix.
            if not config.topic_prefix_resolved():
                config.resolve_topic_prefix()
            beat("collect")
            do_collect()
            # No `if models` guard: with none trained, predict still publishes
            # a record with an empty curve, so the entities exist and read
            # `unknown` rather than never appearing. Guarding here is what left
            # a fresh install with no entities at all for its first seven
            # weeks -- see predict.predict_rows.
            beat("predict")
            do_predict()
            beat("notify")
            _notify_progress()

            # The household's clock, not the container's: TRAIN_HOUR is a
            # local hour. Supervisor happens to inject TZ, which is what made
            # a naive `now()` work; this stops depending on that.
            now = dt.datetime.now(config.tzinfo())
            days = _history_days()
            # Daily while the history is still growing fast, weekly once it is
            # mature and a retrain has little left to change.
            due = (now.hour == TRAIN_HOUR
                   and (days < FULL_HISTORY_DAYS or now.weekday() == TRAIN_WEEKDAY))
            if due and last_train_day != now.date() and days >= MIN_DAYS_TO_TRAIN:
                last_train_day = now.date()
                if _take_train_lock():
                    beat("train")
                    try:
                        do_train()
                        do_predict()
                    finally:
                        _train_lock.release()
            _clear_cycle_error()
        except Exception as err:  # noqa: BLE001
            _record_error(err, from_cycle=True)
            _log.error("cycle failed: %s", err)
        _heartbeat["cycles"] += 1
        _log.debug("cycle %d done in %.1fs; %d horizon(s) shipping, "
                   "mqtt=%s, listener=%s", _heartbeat["cycles"],
                   time.monotonic() - started, _shipping_horizons(),
                   _broker.connected,
                   bool((_listener.status if _listener else {}).get("connected")))
        _say_alive()
        beat("waiting")
        _wait_for_work(started)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global _listener
    # First, so that bootstrap's own warnings land in the configured format
    # rather than being the one thing that still prints raw.
    log.configure()
    # Who this add-on is on MQTT, asked of Supervisor BEFORE anything is named.
    # A few tries with a pause, because the usual reason for a miss is a host
    # that is still booting; if it still fails, the worker keeps asking every
    # cycle and nothing is published in the meantime. It used to be asked once,
    # at import, and a miss was remembered for the life of the process.
    config.resolve_topic_prefix(attempts=5, delay=3.0)
    try:
        settings, ha, source = runtime.bootstrap()
        configured = True
    except Exception as err:  # noqa: BLE001
        settings, ha, source = _degraded_bootstrap(err)
        configured = False
    _state.update({"settings": settings, "ha": ha, "source": source})
    try:
        _load_models()
    except Exception as err:  # noqa: BLE001
        _record_error(err, from_cycle=False)
        _log.error("could not load the models: %s -- serving nothing until a retrain", err)

    # When the models on disk were trained. In memory only, this reset on every
    # restart and the panel said the add-on had never trained while sitting on a
    # full set of models with the timestamp written beside them.
    trained = train_mod.last_summary(config.MODELS_DIR)
    if trained:
        _state["last_train"] = trained["trained_at"]
        _state["last_train_seconds"] = trained["duration_s"]
    if configured:
        _start_worker_threads(settings)

    _log.info("ready: %d model(s), %d person(s), source=%s, log level %s%s",
              len(_state["models"]), len(settings.people), settings.source,
              logging.getLevelName(logging.getLogger().level).lower(),
              "" if configured else " -- NOT RUNNING: fix the configuration on the panel")
    yield
    _stop.set()
    if _listener is not None:
        _listener.stop()
    _broker.close()
    store = getattr(_state.get("source"), "store", None)
    if store is not None:
        store.close()


def _degraded_bootstrap(err: Exception) -> tuple:
    """What to run with when `runtime.bootstrap` refuses.

    It refuses for three reasons -- no people configured (a fresh install
    before its first `person.*` exists), a `config.json` that cannot be read,
    Home Assistant unreachable -- and every one of them used to exit the
    process, taking down the panel that is the tool for fixing the first two.
    Now the app starts with the worker idle, the reason in `last_error`, and
    the Setup tab serving; a successful save starts the worker.
    """
    _record_error(err, from_cycle=False)
    _log.error("start-up failed: %s. The panel is up so this can be fixed "
               "there; nothing is collected or published until a configuration "
               "is saved.", err)
    try:
        ha = runtime.home_assistant()
    except Exception:  # noqa: BLE001
        ha = None
    settings = None
    try:
        settings = (runtime.load_settings(ha) if ha is not None
                    else config.Settings.load())
    except Exception as load_err:  # noqa: BLE001
        _log.error("could not read the saved configuration (%s); starting "
                   "from a blank one", load_err)
    return settings or config.Settings(), ha, None


_threads_started = False


def _start_worker_threads(settings) -> None:
    """Start the worker, the watchdog and the listener. Once, ever."""
    global _threads_started, _listener
    if _threads_started:
        return
    _threads_started = True
    threading.Thread(target=_worker, name="occupancy-worker", daemon=True).start()
    threading.Thread(target=_watchdog, name="occupancy-watchdog",
                     daemon=True).start()
    # Event-driven wake-ups, if Home Assistant will have us. `start` never
    # raises: a listener that cannot connect is a slower forecast, not a
    # broken add-on, and the reason lands on the status page.
    if _listener is None:
        _listener = listen.Listener(runtime.trigger_entities(settings), _nudge.set)
        _listener.start()


# A literal, on purpose. The title is only ever read by the generated OpenAPI
# page, and `config.display_name()` here put a Supervisor round trip in the
# import graph -- before `log.configure()`, before `lifespan` had a chance to
# retry it, and with the failure remembered for the life of the process.
# Everything a user sees derives from the slug in `lifespan` and later.
app = FastAPI(title="Occupancy Forecast", version=train_mod.MODEL_VERSION,
              lifespan=lifespan)
web.mount(app)


# The header Supervisor's Ingress proxy sets, and strips from the incoming
# request first so a browser cannot supply its own. See config.admin_users().
REMOTE_USER_HEADER = "X-Remote-User-Id"


def require_admin(request: Request) -> str | None:
    """Guard the endpoints that change something. Returns the caller's user id.

    Read the allowlist per request rather than caching it: the option can change
    while the add-on is running, and a gate that only reflects the value it saw
    at import time is a gate that quietly stops matching what the Configuration
    tab says.

    A request with no header and an empty allowlist is the ordinary case
    (unrestricted, and outside Ingress there is no header to have). A request
    with no header and a NON-empty allowlist is refused: the only ways to arrive
    without one are to bypass Ingress or to be Supervisor talking to a proxy
    that never set it, and neither is a user we can name.
    """
    allowed = config.admin_users()
    if not allowed:
        return None
    user = request.headers.get(REMOTE_USER_HEADER)
    if user is None or user not in allowed:
        _log.warning("refused %s %s from user %r: not in admin_users",
                     request.method, request.url.path, user)
        raise HTTPException(
            status_code=403,
            detail="not permitted: add this Home Assistant user id to the "
                   "add-on's admin_users option")
    return user


# Applied to the POSTs and to nothing else. The GETs stay open because the panel
# needs them on load and none of them change state; /health stays open because a
# watchdog is not a logged-in user.
admin_only = [Depends(require_admin)]


def _status() -> dict:
    settings = _state["settings"]
    source = _state["source"]
    store = getattr(source, "store", None)

    # Keyed over the HORIZON GRID, not over the loaded artifacts. Two reasons,
    # both visible on the panel: a fresh install has no artifacts at all and
    # would otherwise send an empty map, leaving the strip with nothing to draw
    # on the one day it most needs to explain itself; and a horizon whose
    # pickle failed to load would silently vanish from the denominator, so the
    # card would say "42 of 46" with no hint that two went missing.
    def _metrics(horizon: int) -> dict:
        return (_state["models"].get(horizon) or {}).get("metrics") or {}

    served = {str(h): ("model" if _metrics(h).get("ships") else "none")
              for h in config.HORIZONS_H}
    days = _history_days()
    return {
        "status": "ok" if _state["models"] else "collecting",
        # The panel's header and title. It has to come from here rather than be
        # baked into the bundle: both add-ons build from one tree, so a literal
        # would put stable's name on edge's panel.
        "display_name": config.display_name(),
        "model_version": train_mod.MODEL_VERSION,
        "source": settings.source if settings else None,
        "history": _span(store) if store else {"note": "influx"},
        "days_until_training": max(0, round(MIN_DAYS_TO_TRAIN - days, 1)),
        "horizons_shipping": _shipping_horizons(),
        "people": [s.slug for s in config.PEOPLE] if settings else [],
        "feature_groups": _feature_groups(),
        "horizons": sorted(_state["models"]),
        "served_by": served,
        # WHICH family served, additive beside `served_by` rather than folded
        # into it: the panel counts `served_by == "model"` and the API contract
        # test pins those two values, so the family travels separately.
        "model_kind": {
            str(h): _metrics(h).get("kind")
            for h in config.HORIZONS_H if _metrics(h).get("ships")
        },
        # For the horizons nothing is published for, the baseline that beat the
        # model -- absent where no model was ever trained, which is how the
        # panel tells "the model lost" from "there is no model yet" and colours
        # the same grey cell with two different tooltips. Absence is the
        # signal, the same convention `model_kind` uses one field up.
        #
        # A separate field rather than a richer `served_by` value on purpose:
        # folding the reason into the string is what the old
        # `baseline:persistence` did, and string-splitting a status value is
        # exactly the thing this change removes.
        "best_baseline": {
            str(h): _metrics(h)["best_baseline"]
            for h in config.HORIZONS_H
            if not _metrics(h).get("ships") and _metrics(h).get("best_baseline")
        },
        "eta_models": {s: a.get("metrics", {}).get("ships")
                       for s, a in _state["eta_models"].items()},
        "mqtt": {"connected": _broker.connected, "error": _broker.last_error},
        "listener": _listener.status if _listener else {"connected": False,
                                                        "last_error": "not started"},
        # The worker's own health, because everything else on this page can look
        # perfect while it is hung: `last_error` stays None when the thread is
        # blocked rather than raising, and both connections stay up.
        # `seconds_since_phase` is the number that ages when nothing else does.
        "worker": {
            "phase": _heartbeat["phase"],
            "cycles": _heartbeat["cycles"],
            "seconds_since_phase": round(stall_seconds(), 1),
            "stalled": _stall["since"] is not None,
            "stalled_since": _stall["since"],
            "stalled_in": _stall["phase"],
            "stalls": _stall["count"],
        },
        "loaded_at": _state["loaded_at"], "last_collect": _state["last_collect"],
        "last_predict": _state["last_predict"], "last_train": _state["last_train"],
        "last_train_seconds": _state["last_train_seconds"],
        "next_train": _next_train(dt.datetime.now(config.tzinfo()), days),
        # Which schedule that came off. The panel should not have to infer the
        # policy from the gap between two timestamps.
        "train_cadence": "weekly" if days >= FULL_HISTORY_DAYS else "daily",
        "training_started_at": _state["training_started_at"],
        "last_error": _state["last_error"],
        "training_in_progress": _train_lock.locked(),
        "code": {"fingerprint": _CODE_FINGERPRINT, "imported_at": _IMPORTED_AT},
    }


def _feature_groups() -> dict:
    """Which optional signals this installation actually has.

    The whole point of the status page: a missing group is not an error, it is
    a quieter model, and the user should be able to see which ones they could
    turn on.
    """
    settings = _state["settings"]
    if not settings:
        return {}
    real_proximity = [p for p, pair in settings.proximity.items() if pair and pair[0]]
    return {
        "presence": {"active": bool(settings.people), "detail": settings.people},
        "house_group": {"active": bool(settings.house_entity),
                        "detail": settings.house_entity or "derived from the people"},
        "zones": _zone_signal(settings),
        # Collected, not served. Saying so here is the point: the row explains
        # why a configured sensor is not moving any number yet.
        "next_alarm": {"active": bool(settings.next_alarm),
                       "detail": (f"{len(settings.next_alarm)} found — collecting "
                                  f"history, not yet used by any model"
                                  if settings.next_alarm else
                                  "no companion-app next-alarm sensor found")},
        "proximity": {"active": True,
                      "detail": (f"{len(real_proximity)} from the Proximity integration, "
                                 f"{len(settings.people) - len(real_proximity)} synthesised "
                                 f"from GPS")},
        "holidays": _holiday_signal(settings),
    }


def _zone_signal(settings) -> dict:
    """The zones row, which has to be able to report a rename.

    A zone's history is keyed on its friendly name (features._resolve_zone_events),
    so renaming one silently strands every earlier row in `zone_other`. That was
    the old implementation's failure mode and it went unnoticed for five months.
    Counting the away-states that match nothing turns it into a number on this
    page -- which is the whole reason the count is computed at all.
    """
    if not settings.zones:
        return {"active": False,
                "detail": "none ticked — nothing is known about where they go"}

    names = [z.slug for z in config.ZONES]
    detail = f"{len(names)} ticked — {', '.join(names)}"
    unmatched = _unmatched_zone_states()
    if unmatched:
        listed = ", ".join(f"{name!r} x{n}" for name, n in list(unmatched.items())[:3])
        detail += (f"; {len(unmatched)} away-state(s) in history match no ticked "
                   f"zone: {listed} — a renamed zone looks exactly like this")
    return {"active": True, "detail": detail}


# A rename does not happen twice an hour, and the scan reads every person's
# whole history. Recomputed on that cadence rather than on every status poll.
UNMATCHED_TTL_S = 900


def _unmatched_zone_states() -> dict[str, int]:
    """Cached `features.unmatched_away_states`. Never raises: it is a diagnostic.

    A source that cannot answer must not take the status page down with it --
    the page is where the user goes precisely when something is wrong.
    """
    computed_at, cached = _state["unmatched_zones"]
    now = time.time()
    if computed_at is not None and now - computed_at < UNMATCHED_TTL_S:
        return cached
    # One scan at a time. The status page polls every few seconds and the scan
    # reads every person's whole history; without this, every poll that landed
    # during a scan started another one on its own threadpool thread.
    if not _unmatched_lock.acquire(blocking=False):
        return cached
    try:
        source = _state["source"]
        found = features.unmatched_away_states(
            source, features.history_start(source), None)
        _state["unmatched_zones"] = (now, found)
    except Exception as err:  # noqa: BLE001
        # Keep the old answer, but retry in a minute rather than a quarter
        # hour: a transient read error is not worth a stale diagnostic for
        # that long.
        _log.debug("unmatched-zone scan failed: %s", err)
        found = cached
        _state["unmatched_zones"] = (now - UNMATCHED_TTL_S + 60, cached)
    finally:
        _unmatched_lock.release()
    return found


_unmatched_lock = threading.Lock()

# `store.span()` is a MIN/MAX/COUNT over the whole archive, and `_status` ran
# it twice per status poll -- once directly and once through `_history_days`.
# The collector appends every five minutes, so thirty seconds of staleness on
# the status page is invisible and saves a full scan every ten seconds.
SPAN_TTL_S = 30
_span_cache: dict = {"store": None, "at": 0.0, "span": None}


def _span(store) -> dict:
    now = time.monotonic()
    if (_span_cache["store"] is store and _span_cache["span"] is not None
            and now - _span_cache["at"] < SPAN_TTL_S):
        return _span_cache["span"]
    span = store.span()
    _span_cache.update(store=store, at=now, span=span)
    return span


def _holiday_signal(settings) -> dict:
    """The holidays row, which has to say where the calendar came from.

    "NL" on its own is what prompted this: it reads like something the add-on
    decided, and there was no way to tell it was Home Assistant's country nor
    that it could be changed.
    """
    chosen = config.HOLIDAY_COUNTRY
    if not chosen:
        return {"active": False,
                "detail": "no calendar — is_holiday is always 0"}

    names = {c["code"]: c["name"] for c in discover.holiday_countries()}
    where = ("matches Home Assistant's country" if chosen == settings.country
             else f"your choice; Home Assistant says {settings.country or 'nothing'}")
    return {"active": True,
            "detail": f"{chosen} — {names.get(chosen, chosen)} public holidays, {where}"}


@app.get("/health")
def health() -> JSONResponse:
    """The same page as /api/status, with a status code Supervisor can act on.

    503 while the worker is stalled, or while Home Assistant could not be
    reached at start-up and the worker never started. `watchdog:` in
    config.yaml points Supervisor here, and a non-2xx is what makes it
    restart the add-on -- which is the standard answer to a hung thread that
    this add-on had never used, because this always said 200.
    """
    body = _status()
    unhealthy = _stall["since"] is not None or _state.get("ha") is None
    return JSONResponse(content=body, status_code=503 if unhealthy else 200)


@app.get("/api/status")
def api_status() -> dict:
    return _status()


# The shading changes at most on the hour and costs an HA history call, while
# the panel polls the forecast every minute.
_night_cache: dict = {"at": 0.0, "hours": 0, "bands": []}


def _night_bands(hours: int) -> list[dict]:
    now = time.monotonic()
    if _night_cache["bands"] and _night_cache["hours"] == hours \
            and now - _night_cache["at"] < 900:
        return _night_cache["bands"]
    bands = night.night_bands(_state.get("ha"),
                             dt.datetime.now(dt.timezone.utc), hours)
    _night_cache.update(at=now, hours=hours, bands=bands)
    return bands


@app.get("/api/forecast")
def api_forecast() -> dict:
    """The forecast as last published, for the panel's Overview tab.

    Read out of `_state` rather than recomputed: this is what went to MQTT, so
    the panel and the Home Assistant entities cannot disagree, and opening a tab
    never costs a feature rebuild. It is empty until the first cycle finishes,
    which on a fresh install is a few seconds and is worth saying rather than
    rendering an empty chart.
    """
    rows = _state["forecast"] or []
    hours = max(config.HORIZONS_H) if config.HORIZONS_H else 48
    return {
        "available": bool(rows),
        "predicted_at": _state["last_predict"],
        "house": config.HOUSE_SLUG,
        "horizons": list(config.HORIZONS_H),
        # Decoration, and empty unless a schedule is configured. Offsets from
        # now in hours, because that is the chart's axis -- see night.bands.
        "night": _night_bands(hours),
        "subjects": [{
            "subject": r["subject"],
            # A FRACTION OF THE LAST FIVE MINUTES spent at home, not a
            # forecast -- `nowcast.presence_fraction`. The panel shows it as
            # "actually", beside the prediction, which is the comparison the
            # card exists to make.
            "current": r["current"],
            # The slot the horizons are measured FROM, which is not
            # `predicted_at`: a horizon is h hours after the feature row's slot,
            # and that slot is the in-progress one, so it can be up to half an
            # hour older. The chart's clock labels have to use this or every
            # time on the axis is wrong by up to 30 minutes.
            # `.get`, because a missing anchor must cost the chart its clock
            # labels and not the whole forecast endpoint -- the panel's `at`
            # prop is optional for exactly this reason.
            "observed_at": r.get("observed_at"),
            "curve": r["curve"],
            "next_departure_h": r["next_departure_h"],
            "next_arrival_h": r["next_arrival_h"],
            "eta_minutes": r["eta_minutes"],
            # The routine for today, or null. Deliberately NOT part of
            # `curve` and not a forecast: it is this person's own history for
            # this weekday, calibrated, and the panel says so rather than
            # letting it sit among the model's numbers unlabelled.
            "out": r.get("out"),
            # The combined answer the card actually renders. `.get` so a
            # forecast produced before this field existed still serves.
            "next_change": r.get("next_change"),
        } for r in rows],
    }


@app.get("/api/candidates")
def api_candidates() -> dict:
    return discover.candidates(_state["ha"].states())


@app.get("/api/config")
def api_config() -> dict:
    from dataclasses import asdict
    return asdict(_state["settings"])


def _number(payload: dict, key: str, what: str) -> float:
    """One numeric field out of a config patch, or a 400.

    `bool` is rejected explicitly: JSON `true` arrives as a Python `int`, so
    without this a checkbox sent by mistake would sail through as 1 and be
    clamped to a legal-looking cut.
    """
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(
            status_code=400, detail=f"{key} must be {what}, not {value!r}.")
    return float(value)


def crossing_patch(payload: dict, current: config.Settings) -> dict:
    """The validated crossing cuts in `payload`, as a dict to apply. Or a 400.

    A separate function rather than more lines in the endpoint so that the part
    worth testing needs no HTTP client, and so that it can run BEFORE anything
    is assigned -- see the note at its call site.
    """
    out: dict = {}
    for key in ("departure_threshold", "arrival_threshold"):
        if key not in payload:
            continue
        value = _number(payload, key, "a probability")
        # Open at both ends: a cut of exactly 0 or 1 can never be met by a
        # rounded curve, so it would leave the sensor permanently unknown rather
        # than doing the obvious thing the number looks like it should do.
        if not 0.0 < value < 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must sit strictly between 0 and 1. {value} would "
                       f"make the sensor never fire.")
        out[key] = value

    if "crossing_min_hours" in payload:
        value = _number(payload, "crossing_min_hours", "a whole number of hours")
        limit = max(config.HORIZONS_H)
        if value != int(value) or not 1 <= value <= limit:
            raise HTTPException(
                status_code=400,
                detail=f"crossing_min_hours must be a whole number of hours "
                       f"between 1 and {limit} -- the curve is only {limit} "
                       f"hours long.")
        out["crossing_min_hours"] = int(value)

    # Checked against the merge, not the patch, so that a one-key save is caught
    # too: raising only `departure_threshold` is exactly how the band gets
    # inverted without either number looking wrong on its own.
    departure = out.get("departure_threshold", current.departure_threshold)
    arrival = out.get("arrival_threshold", current.arrival_threshold)
    if departure > arrival:
        raise HTTPException(
            status_code=400,
            detail=f"the away cut ({departure}) is above the home cut "
                   f"({arrival}), so the same forecast would count as both "
                   f"leaving and arriving.")
    return out


def _entity_list(value, key: str, domain: str) -> list[str]:
    """A list of `<domain>.*` entity ids out of a config patch, or a 400.

    The endpoint takes `payload: dict`, which is the whole of its schema, so
    `{"people": "person.alice"}` used to reach `config.configure` as a string
    -- which iterated it and minted twelve one-letter subjects, one with an
    empty slug. Every list field goes through here.
    """
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be a list of entity ids, not {value!r}.")
    wrong = [v for v in value if not v.startswith(f"{domain}.")]
    if wrong:
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be {domain}.* entities: {', '.join(wrong)}")
    return list(value)


def _optional_str(value, key: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise HTTPException(
            status_code=400, detail=f"{key} must be text or null, not {value!r}.")
    return value or None


def typed_patch(payload: dict) -> dict:
    """The identity fields of a config patch, type-checked. Or a 400.

    Shapes only; whether an entity exists is the endpoint's question, because
    that needs Home Assistant. Separate so it is testable without one.
    """
    out: dict = {}
    if "people" in payload:
        out["people"] = _entity_list(payload["people"], "people", "person")
    if "zones" in payload:
        out["zones"] = _entity_list(payload["zones"], "zones", "zone")
    for key in ("house_entity", "holiday_country", "day_schedule"):
        if key in payload:
            out[key] = _optional_str(payload[key], key)
    if "source" in payload:
        if payload["source"] not in ("store", "influx"):
            raise HTTPException(
                status_code=400,
                detail=f"source must be 'store' or 'influx', not {payload['source']!r}.")
        out["source"] = payload["source"]
    for key in ("proximity", "next_alarm"):
        if key in payload:
            value = payload[key]
            if value is not None and not (
                    isinstance(value, dict)
                    and all(isinstance(k, str) for k in value)):
                raise HTTPException(
                    status_code=400,
                    detail=f"{key} must be a mapping keyed by person entity, not {value!r}.")
            out[key] = value
    return out


@app.post("/api/config", dependencies=admin_only)
def api_save_config(payload: dict) -> dict:
    """Replace the configuration and rebuild from it.

    Changing who is tracked changes what the collector fetches and what the
    feature table contains, so the models are now about a different house. They
    are left in place rather than deleted -- a wrong model that says so beats no
    forecast at all -- but the next scheduled train replaces them.

    EVERYTHING is validated on a COPY, and the live settings are swapped only
    once `config.configure` has accepted the copy. The version before this
    assigned onto the live object first and validated second, so a rejected
    save left the process running on values that never reached disk. For a
    typo'd zone that was latent; for an empty people list it was not: the 400
    from `configure` arrived after `settings.people` was already `[]`, so from
    the next cycle the collector fetched nobody's history while the forecasts
    carried on from the old `config.PEOPLE` -- an archive quietly going hollow
    behind a working-looking add-on, until a restart re-read the file.
    """
    live_settings = _state["settings"]
    if _state.get("ha") is None:
        raise HTTPException(
            status_code=503,
            detail="Home Assistant was unreachable when the add-on started; "
                   "restart the add-on once it is up.")
    slugs_before = {s.slug for s in config.PEOPLE}

    # Both validators run before anything is touched, and neither needs HA.
    crossing = crossing_patch(payload, live_settings)
    typed = typed_patch(payload)

    candidate = copy.deepcopy(live_settings)
    for key, value in {**typed, **crossing}.items():
        setattr(candidate, key, value)

    # Rejected rather than absorbed, like the holiday country below. A typo here
    # used to be accepted silently and then spend a week producing an all-zero
    # column, because nothing downstream distinguishes "zone nobody visited"
    # from "zone that does not exist". People get the same rule now: the
    # collector would spend forever asking for a person's history that Home
    # Assistant has no entity for.
    live = {s["entity_id"] for s in _state["ha"].states()}
    missing = [p for p in candidate.people if p not in live]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"not people Home Assistant knows about: {', '.join(missing)}")
    # Same rule as the zones: rejected rather than absorbed. A schedule that
    # does not exist would leave the chart unshaded with no explanation, and
    # "the setting is saved but does nothing" is the state this add-on keeps
    # having to design its way out of.
    schedule = candidate.day_schedule
    if schedule and (not schedule.startswith("schedule.") or schedule not in live):
        raise HTTPException(
            status_code=400,
            detail=f"{schedule!r} is not a schedule entity that exists here.")
    unknown = [z for z in candidate.zones if z not in live]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"not zones Home Assistant knows about: {', '.join(unknown)}")

    # Rejected loudly rather than absorbed. `_holiday_flags` degrades an unknown
    # country to an all-zero column, which is right for a feature build that
    # must not abort but wrong here: it would leave the user looking at a
    # calendar setting that says "IND" and does nothing.
    chosen = candidate.holiday_country
    if chosen and not discover.is_supported_country(chosen):
        raise HTTPException(
            status_code=400,
            detail=f"no holiday calendar for {chosen!r}. Pick one of the "
                   f"countries offered, or none at all.")

    settings = runtime.refresh_environment(candidate, _state["ha"])
    try:
        # Raises before it assigns anything, so a refusal here leaves the
        # module globals on the previous, accepted configuration.
        config.configure(settings)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    settings.save()
    _state["settings"] = settings
    old_store = getattr(_state["source"], "store", None)
    _state["source"] = runtime.build_source(settings, _state["ha"], old_store)
    if old_store is not None and getattr(_state["source"], "store", None) is not old_store:
        old_store.close()
    # Who to listen to changed with who to track. Without this, a person added
    # here is not subscribed to until the next restart -- and nothing says so,
    # because the add-on carries on publishing perfectly good five-minute
    # forecasts for them.
    if _listener is not None:
        _listener.update_entities(runtime.trigger_entities(settings))
    # A save is also how a start-up that refused (no people yet, say) gets
    # going: the worker was never started, and this is the first good
    # configuration. No-op on an add-on that is already running.
    _start_worker_threads(settings)

    # Somebody removed: clear their retained entities, or Home Assistant keeps
    # them forever with the last forecast under a `predicted_at` that never
    # moves. Somebody added or removed at all: the models on disk are about a
    # different house, so retrain now rather than at the next scheduled 04:00,
    # which on a mature install can be a week away.
    slugs_after = {s.slug for s in config.PEOPLE}
    removed = slugs_before - slugs_after
    if removed:
        client = _broker.client()
        if client is not None:
            for slug in sorted(removed):
                cleared = predict_mod.retract(slug, client)
                _log.info("cleared %d retained topic(s) for removed person %s", cleared, slug)
        else:
            _log.warning("no MQTT client; the retained entities for %s stay until "
                         "the broker is back", ", ".join(sorted(removed)))
    if (candidate.people != live_settings.people
            or candidate.zones != live_settings.zones):
        try:
            _start_background_train()
            _log.info("retraining now: the people or zones changed")
        except HTTPException as err:
            _log.info("not retraining yet after the configuration change: %s", err.detail)
    return {"saved": True, "people": sorted(slugs_after)}


# --- the Data tab ---------------------------------------------------------
#
# Thin, like the rest: the reading lives in `explore.py` so that it can be
# tested without an HTTP client, which the shipped image has no room for.
#
# None of these is polled. The status endpoint above is, every ten seconds, so
# that MQTT reconnecting is visible; there is nothing here worth a timer, and
# the panel fetches on mount and when a control changes. That matters because
# these are sync handlers running in the threadpool -- cheap individually, and
# not something to put on a repeat.

@app.get("/api/explore/archive")
def api_explore_archive() -> dict:
    return explore.archive_inventory(_state["source"], _state["settings"])


@app.get("/api/explore/entity")
def api_explore_entity(entity_id: str, days: int = explore.DEFAULT_DAYS) -> dict:
    return explore.entity_series(_state["source"], _state["settings"],
                                 entity_id, days)


# Keyed on (path, mtime_ns), so a retrain -- which rewrites the file -- drops the
# entry without anything having to remember to invalidate it. Two files, so a
# plain dict rather than an LRU.
#
# The archive is deliberately NOT cached: the collector rewrites it every five
# minutes, and a stale row count is precisely the thing that card exists to
# report. Neither is the feature series, whose key space is unbounded.
_explore_cache: dict[str, tuple[int, dict]] = {}
_explore_lock = threading.Lock()


def _cached(key: str, path: Path, build) -> dict:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return build()
    with _explore_lock:
        hit = _explore_cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    answer = build()
    with _explore_lock:
        _explore_cache[key] = (stamp, answer)
    return answer


@app.get("/api/explore/features")
def api_explore_features() -> dict:
    return _cached("features", config.FEATURES_PATH,
                   lambda: explore.feature_inventory(config.FEATURES_PATH))


def _known_horizon(horizon: int) -> int:
    """A horizon on the grid, or a 404. `/horizon/999` used to be answered."""
    if horizon not in config.HORIZONS_H:
        raise HTTPException(
            status_code=404,
            detail=f"+{horizon} h is not a forecast horizon; the grid is "
                   f"+{min(config.HORIZONS_H)} h to +{max(config.HORIZONS_H)} h.")
    return horizon


@app.get("/api/explore/feature-series")
def api_explore_feature_series(subject: str, column: str, days: int = 30) -> dict:
    # Clamped like every other `days` here; this one arrived unclamped.
    return explore.feature_series(config.FEATURES_PATH, subject, column,
                                  explore._clamp_days(days))


@app.get("/api/explore/horizon/{horizon}")
def api_explore_horizon(horizon: int) -> dict:
    return explore.horizon_recipe(_known_horizon(horizon), _state["models"])


# Uncached, for the same reason `entity_series` is: it reads the archive and
# the forecast table, both of which the serve cycle rewrites every five
# minutes, and a stale answer here is precisely the thing the card exists to
# catch.
@app.get("/api/explore/verification")
def api_explore_verification(subject: str, horizon: int,
                             days: int = explore.DEFAULT_DAYS) -> dict:
    return explore.verification(_state["source"], _state["settings"],
                                subject, horizon, days)


@app.get("/api/explore/metrics")
def api_explore_metrics() -> dict:
    return _cached("metrics", train_mod.summary_path(config.MODELS_DIR),
                   lambda: explore.metrics_summary(config.MODELS_DIR,
                                                   _state["models"]))


@app.get("/api/explore/metrics/{horizon}")
def api_explore_metrics_detail(horizon: int) -> dict:
    return explore.metrics_detail(config.MODELS_DIR, _known_horizon(horizon),
                                  _state["models"])


@app.post("/collect", dependencies=admin_only)
def run_collect() -> dict:
    return do_collect()


@app.post("/predict", dependencies=admin_only)
def run_predict() -> dict:
    try:
        results = do_predict()
    except Exception as err:  # noqa: BLE001
        _record_error(err, from_cycle=True)
        raise HTTPException(status_code=500, detail=str(err)) from err
    return {"predicted_at": _state["last_predict"],
            "subjects": [{k: r[k] for k in ("subject", "current", "curve",
                                            "next_departure_h", "next_arrival_h",
                                            "eta_minutes")} for r in results]}


@app.post("/train", dependencies=admin_only)
def run_train(response: Response, background: bool = False) -> dict:
    """Train now. `?background=1` starts it and returns; the panel uses that.

    The default stays synchronous so that anything already scripted against this
    endpoint -- which waits for the summary it returns -- is unaffected.
    """
    if background:
        _start_background_train()
        response.status_code = 202
        return {"started": True}

    if not _take_train_lock():
        raise HTTPException(status_code=409, detail="a train is already running")
    try:
        summary = do_train()
    except HTTPException:
        raise
    except Exception as err:  # noqa: BLE001
        _record_error(err, from_cycle=False)
        raise HTTPException(status_code=500, detail=str(err)) from err
    finally:
        _train_lock.release()
    return {"finished_at": _state["last_train"], **summary}


@app.post("/reload", dependencies=admin_only)
def reload_models() -> dict:
    _load_models()
    return {"loaded_at": _state["loaded_at"], "horizons": sorted(_state["models"])}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return web.index_html()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
