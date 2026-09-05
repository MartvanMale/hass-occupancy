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
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

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
# Three cycles, so a slow-but-moving box is never called stalled. A retrain is
# exempt outright (`training_in_progress`) rather than covered by a larger
# number -- it legitimately takes minutes, and picking a threshold that spans it
# would mean picking one that cannot see a stall during it either.
STALL_SECONDS = COLLECT_MINUTES * 60 * 3
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
    return store.span()["days"] if store else float("inf")


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

    Local time, because the schedule is: the worker compares against a naive
    `datetime.now()`. The offset travels with the string so the browser renders
    it in the same clock the user's house runs on.
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
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a train is already running")

    def run() -> None:
        try:
            do_train()
            # Publish with the new models at once rather than waiting up to five
            # minutes for the next cycle, exactly as the worker does.
            do_predict()
        except Exception as err:  # noqa: BLE001
            _state["last_error"] = f"{dt.datetime.now(dt.timezone.utc).isoformat()}: {err}"
            _log.error("train failed: %s", err)
        finally:
            _train_lock.release()

    threading.Thread(target=run, name="occupancy-train", daemon=True).start()


def _shipping_horizons() -> int:
    """How many horizons the model has actually earned. The number that moves."""
    return sum(1 for a in _state["models"].values()
               if a.get("metrics", {}).get("ships"))


def _notify_progress() -> None:
    """Tell the user why nothing is published yet, once, and clear it later."""
    ha, days = _state["ha"], _history_days()
    # Both the id and the title carry the add-on's own name, so that stable and
    # edge raise two separate notifications and the reader can tell them apart.
    notify_id, name = notify_collecting_id(), config.display_name()
    try:
        if days < MIN_DAYS_TO_TRAIN:
            ha.notify(
                f"{name} is still learning",
                f"Collected **{days:.0f} of {MIN_DAYS_TO_TRAIN} days** of history. "
                f"No forecast is published yet -- the sensors exist and read "
                f"unknown until a model has earned a horizon.",
                notify_id)
        elif not _shipping_horizons():
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
    except Exception:  # noqa: BLE001
        pass  # a notification is never worth failing a cycle for


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

    A retrain is exempt. It legitimately runs for minutes and holds the worker
    the whole time, and a threshold wide enough to span one would be too wide
    to catch anything else. The exemption reads `_train_lock`, which is where
    that fact actually lives: it used to read `_state["training_in_progress"]`,
    a key **nothing ever writes** -- `/api/status` derives the same name from
    the lock -- so the exemption had quietly been dead. It has never fired only
    because a train takes about three minutes against `STALL_SECONDS`; a slower
    box or a longer history would have dumped every thread's stack and dropped
    the MQTT client in the middle of a perfectly healthy retrain.

    Reports the transition, never the state: one report per episode, and one
    when it recovers. A watchdog that logs every minute for eleven hours is a
    watchdog nobody reads.
    """
    if _train_lock.locked():
        return False
    late = stall_seconds(now)
    if late < STALL_SECONDS:
        if _stall["since"] is not None:
            _log.warning("worker recovered; it was stuck in %s", _stall["phase"])
            _stall.update(since=None, phase=None, acted=False)
        return False

    if not _stall["acted"]:
        _stall["count"] += 1
        _stall["since"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        _stall["phase"] = _heartbeat["phase"]
        _stall["acted"] = True
        _log.critical(
            "worker STALLED in %s for %.0fs after %d cycles. Thread stacks "
            "follow -- the frame at the top of occupancy-worker is where it "
            "is stuck.", _heartbeat["phase"], late, _heartbeat["cycles"])
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

            now = dt.datetime.now()
            days = _history_days()
            # Daily while the history is still growing fast, weekly once it is
            # mature and a retrain has little left to change.
            due = (now.hour == TRAIN_HOUR
                   and (days < FULL_HISTORY_DAYS or now.weekday() == TRAIN_WEEKDAY))
            if due and last_train_day != now.date() and days >= MIN_DAYS_TO_TRAIN:
                last_train_day = now.date()
                if _train_lock.acquire(blocking=False):
                    try:
                        do_train()
                        do_predict()
                    finally:
                        _train_lock.release()
        except Exception as err:  # noqa: BLE001
            _state["last_error"] = f"{dt.datetime.now(dt.timezone.utc).isoformat()}: {err}"
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
    settings, ha, source = runtime.bootstrap()
    _state.update({"settings": settings, "ha": ha, "source": source})
    _load_models()

    # When the models on disk were trained. In memory only, this reset on every
    # restart and the panel said the add-on had never trained while sitting on a
    # full set of models with the timestamp written beside them.
    trained = train_mod.last_summary(config.MODELS_DIR)
    if trained:
        _state["last_train"] = trained["trained_at"]
        _state["last_train_seconds"] = trained["duration_s"]
    worker = threading.Thread(target=_worker, name="occupancy-worker", daemon=True)
    worker.start()
    threading.Thread(target=_watchdog, name="occupancy-watchdog",
                     daemon=True).start()

    # Event-driven wake-ups, if Home Assistant will have us. `start` never
    # raises: a listener that cannot connect is a slower forecast, not a
    # broken add-on, and the reason lands on the status page.
    _listener = listen.Listener(runtime.trigger_entities(settings), _nudge.set)
    _listener.start()

    _log.info("ready: %d model(s), %d person(s), source=%s, log level %s",
              len(_state["models"]), len(settings.people), settings.source,
              logging.getLevelName(logging.getLogger().level).lower())
    yield
    _stop.set()
    if _listener is not None:
        _listener.stop()
    _broker.close()


app = FastAPI(title=config.display_name(), version=train_mod.MODEL_VERSION,
              lifespan=lifespan)
web.mount(app)


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
        "history": store.span() if store else {"note": "influx"},
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
        "next_train": _next_train(dt.datetime.now().astimezone(), days),
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
    try:
        source = _state["source"]
        found = features.unmatched_away_states(
            source, features.history_start(source), None)
    except Exception:
        found = cached
    _state["unmatched_zones"] = (now, found)
    return found


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
def health() -> dict:
    return _status()


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


@app.post("/api/config")
def api_save_config(payload: dict) -> dict:
    """Replace the configuration and rebuild from it.

    Changing who is tracked changes what the collector fetches and what the
    feature table contains, so the models are now about a different house. They
    are left in place rather than deleted -- a wrong model that says so beats no
    forecast at all -- but the next scheduled train replaces them.
    """
    settings = _state["settings"]

    # Validated BEFORE anything is assigned, unlike the zone and holiday checks
    # below. `settings` is the live object the five-minute predict cycle reads,
    # so assigning first and rejecting second leaves the process serving values
    # that never reached disk. Latent for zones -- nothing reads them until the
    # next feature build -- but immediately visible for a cut, which the very
    # next cycle publishes from.
    crossing = crossing_patch(payload, settings)

    for key in ("people", "zones", "house_entity", "proximity", "source",
                "holiday_country", "next_alarm", "day_schedule"):
        if key in payload:
            setattr(settings, key, payload[key])
    # Not in the tuple above: these are validated and coerced by
    # `crossing_patch`, so the tuple is not the complete list of writable keys.
    for key, value in crossing.items():
        setattr(settings, key, value)

    # Rejected rather than absorbed, like the holiday country below. A typo here
    # used to be accepted silently and then spend a week producing an all-zero
    # column, because nothing downstream distinguishes "zone nobody visited"
    # from "zone that does not exist".
    live = {s["entity_id"] for s in _state["ha"].states()}
    # Same rule as the zones: rejected rather than absorbed. A schedule that
    # does not exist would leave the chart unshaded with no explanation, and
    # "the setting is saved but does nothing" is the state this add-on keeps
    # having to design its way out of.
    schedule = settings.day_schedule
    if schedule and (not schedule.startswith("schedule.") or schedule not in live):
        raise HTTPException(
            status_code=400,
            detail=f"{schedule!r} is not a schedule entity that exists here.")
    unknown = [z for z in settings.zones
               if not z.startswith("zone.") or z not in live]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"not zones Home Assistant knows about: {', '.join(unknown)}")

    # Rejected loudly rather than absorbed. `_holiday_flags` degrades an unknown
    # country to an all-zero column, which is right for a feature build that
    # must not abort but wrong here: it would leave the user looking at a
    # calendar setting that says "IND" and does nothing.
    chosen = settings.holiday_country
    if chosen and not discover.is_supported_country(chosen):
        raise HTTPException(
            status_code=400,
            detail=f"no holiday calendar for {chosen!r}. Pick one of the "
                   f"countries offered, or none at all.")

    settings = runtime.refresh_environment(settings, _state["ha"])
    try:
        config.configure(settings)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    settings.save()
    _state["settings"] = settings
    _state["source"] = runtime.build_source(settings, _state["ha"],
                                            getattr(_state["source"], "store", None))
    # Who to listen to changed with who to track. Without this, a person added
    # here is not subscribed to until the next restart -- and nothing says so,
    # because the add-on carries on publishing perfectly good five-minute
    # forecasts for them.
    if _listener is not None:
        _listener.update_entities(runtime.trigger_entities(settings))
    return {"saved": True, "people": [s.slug for s in config.PEOPLE]}


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


@app.get("/api/explore/feature-series")
def api_explore_feature_series(subject: str, column: str, days: int = 30) -> dict:
    return explore.feature_series(config.FEATURES_PATH, subject, column, days)


@app.get("/api/explore/horizon/{horizon}")
def api_explore_horizon(horizon: int) -> dict:
    return explore.horizon_recipe(horizon, _state["models"])


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
    return explore.metrics_detail(config.MODELS_DIR, horizon, _state["models"])


@app.post("/collect")
def run_collect() -> dict:
    return do_collect()


@app.post("/predict")
def run_predict() -> dict:
    try:
        results = do_predict()
    except Exception as err:  # noqa: BLE001
        _state["last_error"] = f"{dt.datetime.now(dt.timezone.utc).isoformat()}: {err}"
        raise HTTPException(status_code=500, detail=str(err)) from err
    return {"predicted_at": _state["last_predict"],
            "subjects": [{k: r[k] for k in ("subject", "current", "curve",
                                            "next_departure_h", "next_arrival_h",
                                            "eta_minutes")} for r in results]}


@app.post("/train")
def run_train(response: Response, background: bool = False) -> dict:
    """Train now. `?background=1` starts it and returns; the panel uses that.

    The default stays synchronous so that anything already scripted against this
    endpoint -- which waits for the summary it returns -- is unaffected.
    """
    if background:
        _start_background_train()
        response.status_code = 202
        return {"started": True}

    if not _train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a train is already running")
    try:
        summary = do_train()
    except HTTPException:
        raise
    except Exception as err:  # noqa: BLE001
        _state["last_error"] = f"{dt.datetime.now(dt.timezone.utc).isoformat()}: {err}"
        raise HTTPException(status_code=500, detail=str(err)) from err
    finally:
        _train_lock.release()
    return {"finished_at": _state["last_train"], **summary}


@app.post("/reload")
def reload_models() -> dict:
    _load_models()
    return {"loaded_at": _state["loaded_at"], "horizons": sorted(_state["models"])}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return web.index_html()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
