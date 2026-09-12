"""The add-on: collector, scheduler, HTTP API and the configuration UI.

Handlers are sync `def`: an `async def` doing blocking history reads or a fit
would occupy the event loop and hang `/health` exactly when it is needed. Three
loops run on their own -- collector every COLLECT_MINUTES, predictor after each
collect, trainer weekly and on demand. ADVISORY ONLY: the only writes are MQTT
sensor states and persistent notifications.
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

# Floor between two cycles: coming home fires three entities within seconds and
# each cycle is a full `predict.LOOKBACK_DAYS` rebuild. Load protection, not
# politeness.
MIN_CYCLE_SECONDS = 60
TRAIN_WEEKDAY = 0          # Monday
TRAIN_HOUR = 4

# How long the worker may go without reaching its next phase. The failure this
# exists for is silent: a blocked thread leaves `last_error` None and `/health`
# green while nothing is published. A retrain gets its OWN deadline rather than
# an exemption -- an exempt train that hangs is invisible.
STALL_SECONDS = COLLECT_MINUTES * 60 * 3
TRAIN_STALL_SECONDS = 60 * 60
WATCHDOG_SECONDS = 60

# How often the add-on says it is alive when nothing changed -- the point is to
# make silence mean something. Hourly keeps the Log tab readable; per-cycle
# detail is DEBUG.
HEARTBEAT_SECONDS = 3600

# Below this `calendar_folds` returns nothing. Above it early models are weak
# and mostly fail the ship gate, which is the point: training early cannot make
# any published number worse.
MIN_DAYS_TO_TRAIN = evaluate.MIN_TRAINABLE_DAYS

# Weekly once the history is mature, daily while it is short: a fresh install
# changes from day to day.
FULL_HISTORY_DAYS = evaluate.FULL_GEOMETRY_DAYS

def notify_collecting_id() -> str:
    """The persistent_notification id for the "still learning" notice.

    Derived from the slug so stable and edge do not raise and dismiss each
    other's. Cannot be a module constant: `topic_prefix()` calls Supervisor,
    and that must not be in the import graph.
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
    # What was published, held here rather than on the source so an Influx
    # install has one too.
    "forecast_log": None,
    "models": {}, "eta_models": {}, "out_routine": {}, "departure_routine": {},
    "loaded_at": None, "last_collect": None, "last_predict": None,
    "last_train": None, "last_train_seconds": None, "training_started_at": None,
    # The full text, and what `/api/status` may show of it -- see log.SEE_THE_LOG.
    "last_error": None, "last_error_public": None, "forecast": [],
    # Days of usable history, recounted once a worker cycle. None until the
    # first one finishes, which reads as 0.0 -- see `_history_days`.
    "usable_days": None,
    # (computed_at, {state: count}). The scan is a full-history read of every
    # person, and the status page polls every few seconds -- see _unmatched.
    "unmatched_zones": (None, {}),
}
_broker = predict_mod.Broker()
_train_lock = threading.Lock()
# When `_train_lock` was last taken, so the watchdog can time a train. Stamped
# by `_take_train_lock`, the only way the lock is meant to be acquired.
_train_started = {"at": 0.0}
# Whether `last_error` came from the cycle: a later good cycle clears its own
# error and leaves a train's alone.
_cycle_failed = False
# The worker's pulse, written by the worker and read by the watchdog -- a
# separate thread, because the failure watched for is the worker not running.
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
    _state["departure_routine"] = departure.load_routine(config.MODELS_DIR)
    _state["loaded_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _history_days() -> float:
    """Days of history a model could be fitted on, as of the last worker cycle.

    READ and never computed: `usable_history_days` walks the whole archive and
    this is reached from `/api/status` and `/health`. `inf` without a store.
    """
    if getattr(_state["source"], "store", None) is None:
        return float("inf")
    days = _state.get("usable_days")
    return 0.0 if days is None else days


def _refresh_history_days() -> None:
    """Recount usable history. The worker's job, and nobody else's."""
    source = _state["source"]
    if getattr(source, "store", None) is None:
        return
    _state["usable_days"] = features.usable_history_days(source)


def do_collect() -> dict:
    """Pull new history, and sample a distance for anyone without Proximity."""
    settings, ha, source = _state["settings"], _state["ha"], _state["source"]
    store = getattr(source, "store", None)
    if store is None:
        return {"skipped": "influx source keeps its own history"}

    result = source.collect(runtime.tracked_entities(settings),
                            absence_is_a_reading=runtime.absence_entities(settings),
                            gap_is_a_boundary=runtime.presence_entities(settings))
    synthetic = discover.sample_distances(ha, settings)
    result["synthetic"] = store.append(synthetic)
    _state["last_collect"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return result


def _record_forecasts(results: list[dict]) -> None:
    """Keep what was published, so it can be scored later.

    A HORIZON WITH NO FORECAST WRITES NO ROW -- the absence is the record. The
    target slot is measured from `observed_at`, already on the grid, so the
    join against truth is an equality. Wrapped whole: bookkeeping for a chart
    must not stop the house getting a forecast.
    """
    store = _state["forecast_log"]
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
        settings = _state["settings"]
        keep = (settings.forecast_retention_days if settings
                else config.FORECAST_RETENTION_DAYS)
        if keep:                                  # 0 means keep everything
            store.prune_forecasts(
                dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep))
    except Exception:
        _log.warning("could not record this cycle's forecasts; the verification "
                     "chart will show a gap here", exc_info=True)


def do_predict() -> list[dict]:
    results = predict_mod.run_cycle(
        _state["models"], _broker.client(), _state["source"], _state["eta_models"],
        _state["out_routine"], _state["departure_routine"])
    _state["forecast"] = results
    _state["last_predict"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _record_forecasts(results)
    return results


def _too_little_history() -> HTTPException | None:
    """The 409 that says a train cannot be attempted yet, or None.

    Split out so the background variant can refuse before spawning a thread:
    an HTTPException raised in there reaches nobody.
    """
    days = _history_days()
    if days >= MIN_DAYS_TO_TRAIN:
        return None
    return HTTPException(
        status_code=409,
        detail=f"only {days:.1f} days of observed presence; "
               f"{MIN_DAYS_TO_TRAIN} are needed "
               f"before there is enough to hold out a test window. No forecast "
               f"is published until then.")


def do_train() -> dict:
    refusal = _too_little_history()
    if refusal is not None:
        raise refusal

    # Timed end to end, not around `train_all` alone: the feature table and the
    # ETA models are built either side of it.
    started = time.monotonic()
    _state["training_started_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds")
    # Announced, because it is the one thing here that takes minutes and pins
    # the box.
    _log.info("training started (%.0f days of history)", _history_days())
    # Timed per stretch as well as end to end: these four say whether the
    # answer is in the fits at all.
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
    # Arithmetic over the table already in hand. Guarded like the ETA models: a
    # household with no zones has nothing to answer, and that must not fail a
    # train.
    try:
        with phases("out routine"):
            labelled = outing_mod.label_out_days(table, departure.label_days(table))
            routine = outing_mod.fit_routine(labelled)
            outing_mod.save_routine(routine, config.MODELS_DIR)
        _log.info("out routine fitted for %d person(s)", len(routine))
    except Exception as err:  # noqa: BLE001
        _log.error("out routine failed: %s", err)
    # Guarded SEPARATELY from the one above, which reads the configured zones:
    # this one reads none, and it is what times the next-change row.
    try:
        with phases("departure routine"):
            leaving = departure.fit_routine(departure.label_days(table))
            departure.save_routine(leaving, config.MODELS_DIR)
        _log.info("departure routine fitted for %d subject(s)", len(leaving))
    except Exception as err:  # noqa: BLE001
        _log.error("departure routine failed: %s", err)
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

    Separate from the `due` test in `_worker` -- that answers "is it now" --
    and they share the constants. Local time, with the offset in the string,
    because the schedule is.
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
    """Kick off a train and return: a synchronous run is minutes and Ingress
    gives up first.

    Both refusals happen HERE, in the request thread. The lock is acquired here
    and released in the thread, so `training_in_progress` is true from the
    moment the caller is told.
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
        # lock held by nobody is a 409 forever.
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
    # Same ISO stamp on both: the panel splits on it to say how long ago, and a
    # timestamp leaks nothing. Only the message is held back.
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    _state["last_error"] = f"{stamp}: {err}"
    _state["last_error_public"] = f"{stamp}: {log.SEE_THE_LOG}"
    _cycle_failed = from_cycle


def _clear_cycle_error() -> None:
    """A good cycle clears the error a bad cycle left, and only that one."""
    global _cycle_failed
    if _cycle_failed:
        _state["last_error"] = None
        _state["last_error_public"] = None
        _cycle_failed = False


def _shipping_horizons() -> int:
    """How many horizons the model has actually earned. The number that moves."""
    return sum(1 for a in _state["models"].values()
               if a.get("metrics", {}).get("ships"))


_notify_error: str | None = None
# What the notification last said, so it is sent on a transition: re-creating
# it each cycle brought it back for anyone who dismissed it.
_notified: tuple | None = None


def _notify_progress() -> None:
    """Tell the user why nothing is published yet, once, and clear it later."""
    global _notify_error, _notified
    # Id and title carry the add-on's own name, so neither may be raised under
    # a GUESSED name -- see config.resolve_topic_prefix.
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
                f"Observed **{days:.0f} of {MIN_DAYS_TO_TRAIN} days** of "
                f"presence. No forecast is published yet -- the sensors exist "
                f"and read unknown until a model has earned a horizon.",
                notify_id)
        elif state[0] == "training":
            # Training now, but nothing has beaten its baseline yet. Say so
            # rather than going quiet -- silence here reads as "broken".
            ha.notify(
                f"{name} is still learning",
                f"Training on **{days:.0f} days** of presence. No horizon beats "
                f"its baseline yet, so nothing is published. This improves as "
                f"history accumulates.",
                notify_id)
        else:
            ha.dismiss(notify_id)
        _notified = state
        _notify_error = None
    except Exception as err:  # noqa: BLE001
        # Never worth failing a cycle for, but worth one line per distinct
        # failure: this is the same token every other HA call uses.
        if str(err) != _notify_error:
            _log.warning("could not update the progress notification: %s", err)
        _notify_error = str(err)


def _wait_for_work(since: float) -> None:
    """Block until the poll is due, or Home Assistant says something changed.

    `since` is when the finished cycle started; MIN_CYCLE_SECONDS is measured
    from it. The periodic poll is not a fallback: it covers the slot turning
    over, the train check, and an install where the subscription never
    connected. Waited in one-second slices so `_stop` is not ignored for five
    minutes.
    """
    deadline = since + COLLECT_MINUTES * 60
    while not _stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if not _nudge.wait(timeout=min(remaining, 1.0)):
            continue                                 # slice expired; re-check
        _nudge.clear()
        # Debounce: anything arriving during this sleep is answered by the
        # single cycle that follows. One arrival, one rebuild.
        elapsed = time.monotonic() - since
        if elapsed < MIN_CYCLE_SECONDS and _stop.wait(MIN_CYCLE_SECONDS - elapsed):
            return
        _nudge.clear()
        return


def beat(phase: str) -> None:
    """Mark the worker as having reached a new phase, so a stall report names
    the step that hung."""
    _heartbeat["at"] = time.monotonic()
    _heartbeat["phase"] = phase


def stall_seconds(now: float | None = None) -> float:
    """How long the worker has been in one phase."""
    return (time.monotonic() if now is None else now) - _heartbeat["at"]


def check_stall(now: float | None = None, dump=None) -> bool:
    """Is the worker stuck, and if it has just got stuck, say so once.

    `now` and `dump` are injected so the decision is testable. A retrain is
    measured against TRAIN_STALL_SECONDS from the moment the lock was taken,
    by whichever thread took it: an exempt train hid the one failure this
    exists for. Reports transitions, never the state.
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
        # Best-effort nudge, not a diagnosis: the MQTT client is the only piece
        # the worker touches that another thread can free, and closing its
        # socket raises inside a blocked publish.
        try:
            _broker.close()
            _log.warning("dropped the MQTT client; it reconnects next cycle")
        except Exception as err:  # noqa: BLE001
            _log.error("could not drop the MQTT client: %s", err)
    return True


def _dump_stacks() -> None:
    """Every thread's stack, to the add-on log.

    `faulthandler`, not `traceback`: it prints the C frame, so a thread parked
    in a socket read is distinguishable from one spinning.
    """
    faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
    sys.stdout.flush()


def _watchdog() -> None:
    """Watch the worker from outside it; see STALL_SECONDS."""
    while not _stop.wait(WATCHDOG_SECONDS):
        try:
            check_stall()
        except Exception as err:  # noqa: BLE001
            _log.error("watchdog: %s", err)


def _say_alive(now: float | None = None) -> bool:
    """One INFO line an hour, unconditional on health: a heartbeat that only
    appears when things are good cannot be told from a stopped process."""
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
    stale_retrained = False
    while not _stop.is_set():
        started = time.monotonic()
        try:
            # Keep asking until Supervisor answers. Nothing publishes and no
            # notification is raised until it does; see config.resolve_topic_prefix.
            if not config.topic_prefix_resolved():
                config.resolve_topic_prefix()
            beat("collect")
            do_collect()
            # Its own phase: a full-archive recount is slow enough that the
            # watchdog would otherwise be timing it as part of the collect.
            beat("history")
            _refresh_history_days()
            # No `if models` guard: with none trained, predict still publishes
            # an empty curve, so the entities exist and read `unknown` rather
            # than never appearing -- see predict.predict_rows.
            beat("predict")
            do_predict()
            beat("notify")
            _notify_progress()

            # The household's clock, not the container's: TRAIN_HOUR is a
            # local hour.
            now = dt.datetime.now(config.tzinfo())
            days = _history_days()
            # Daily while the history is still growing fast, weekly once it is
            # mature and a retrain has little left to change.
            due = (now.hour == TRAIN_HOUR
                   and (days < FULL_HISTORY_DAYS or now.weekday() == TRAIN_WEEKDAY)
                   and last_train_day != now.date())
            # A MODEL_VERSION bump refuses every artifact at once, and the next
            # scheduled train may be a week away -- so retrain off-schedule, once.
            forced = (not _state["models"] and not stale_retrained
                      and bool(predict_mod.stale_artifacts()))
            if (due or forced) and days >= MIN_DAYS_TO_TRAIN:
                if forced:
                    stale_retrained = True
                    _log.info("retraining now: every model on disk was built by "
                              "an older version, and nothing is published until "
                              "they are replaced")
                if due:
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
    # Who this add-on is on MQTT, asked of Supervisor BEFORE anything is named;
    # a few retries because the usual cause of a miss is a host still booting.
    config.resolve_topic_prefix(attempts=5, delay=3.0)
    try:
        settings, ha, source, forecast_log = runtime.bootstrap()
        configured = True
    except Exception as err:  # noqa: BLE001
        settings, ha, source, forecast_log = _degraded_bootstrap(err)
        configured = False
    _state.update({"settings": settings, "ha": ha, "source": source,
                   "forecast_log": forecast_log})
    try:
        _load_models()
    except Exception as err:  # noqa: BLE001
        _record_error(err, from_cycle=False)
        _log.error("could not load the models: %s -- serving nothing until a retrain", err)

    # When the models on disk were trained; in memory only, this reset on every
    # restart.
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
    # One object on `store` (the source reads through it) and on `influx` (only
    # the forecast table is written), so closing the log closes everything.
    store = _state.get("forecast_log")
    if store is not None:
        store.close()


def _degraded_bootstrap(err: Exception) -> tuple:
    """What to run with when `runtime.bootstrap` refuses: the worker idles and
    the panel stays up, because it is the tool for fixing the refusal.
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
    return settings or config.Settings(), ha, None, None


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
    # `start` never raises: a listener that cannot connect is a slower
    # forecast, not a broken add-on.
    if _listener is None:
        _listener = listen.Listener(runtime.trigger_entities(settings), _nudge.set)
        _listener.start()


# A literal on purpose: the title is only read by the OpenAPI page, and
# `display_name()` here would put a Supervisor round trip in the import graph
# before `log.configure()`.
app = FastAPI(title="Occupancy Forecast", version=train_mod.MODEL_VERSION,
              lifespan=lifespan)
web.mount(app)


# The header Supervisor's Ingress proxy sets, and strips from the incoming
# request first so a browser cannot supply its own. See config.admin_users().
REMOTE_USER_HEADER = "X-Remote-User-Id"


def require_admin(request: Request) -> str | None:
    """Guard the endpoints that change something. Returns the caller's user id.

    Read the allowlist per request, because the option changes while the
    add-on runs. No header plus an empty allowlist is the ordinary unrestricted
    case; no header plus a NON-empty one is refused.
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


# POSTs only: the GETs stay open because the panel needs them on load, and
# /health because a watchdog is not a user.
admin_only = [Depends(require_admin)]


def _status() -> dict:
    settings = _state["settings"]
    source = _state["source"]
    store = getattr(source, "store", None)

    # Keyed over the HORIZON GRID, not the loaded artifacts: a fresh install
    # would otherwise send an empty map, and a failed pickle would vanish from
    # the denominator.
    def _metrics(horizon: int) -> dict:
        return (_state["models"].get(horizon) or {}).get("metrics") or {}

    served = {str(h): ("model" if _metrics(h).get("ships") else "none")
              for h in config.HORIZONS_H}
    days = _history_days()
    return {
        "status": "ok" if _state["models"] else "collecting",
        # The panel's title, from here rather than baked into the bundle: both
        # add-ons build from one tree.
        "display_name": config.display_name(),
        "model_version": train_mod.MODEL_VERSION,
        "source": settings.source if settings else None,
        "history": _span(store) if store else {"note": "influx"},
        "days_until_training": max(0, round(MIN_DAYS_TO_TRAIN - days, 1)),
        # Not the same as `history.days`, which is the age of the oldest row.
        "usable_presence_days": round(days, 3) if store else None,
        "horizons_shipping": _shipping_horizons(),
        "people": [s.slug for s in config.PEOPLE] if settings else [],
        "feature_groups": _feature_groups(),
        "horizons": sorted(_state["models"]),
        "served_by": served,
        # Additive beside `served_by`, which the contract test pins to two
        # values.
        "model_kind": {
            str(h): _metrics(h).get("kind")
            for h in config.HORIZONS_H if _metrics(h).get("ships")
        },
        # For horizons nothing is published for, the baseline that beat the
        # model; absent where no model was trained, so the panel can tell the
        # two greys apart.
        "best_baseline": {
            str(h): _metrics(h)["best_baseline"]
            for h in config.HORIZONS_H
            if not _metrics(h).get("ships") and _metrics(h).get("best_baseline")
        },
        "eta_models": {s: a.get("metrics", {}).get("ships")
                       for s, a in _state["eta_models"].items()},
        "mqtt": {"connected": _broker.connected,
                 "error": _broker.last_error_public},
        "listener": _listener.status if _listener else {"connected": False,
                                                        "last_error": "not started"},
        # The worker's own health: everything else here can look perfect while
        # it is hung, and `seconds_since_phase` is the number that ages.
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
        "last_error": _state["last_error_public"],
        "training_in_progress": _train_lock.locked(),
        "code": {"fingerprint": _CODE_FINGERPRINT, "imported_at": _IMPORTED_AT},
    }


def _feature_groups() -> dict:
    """Which optional signals this installation has. A missing group is not an
    error, it is a quieter model."""
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
    """The zones row, which has to be able to report a rename: zone history is
    keyed on the friendly name (features._resolve_zone_events), so renaming one
    silently strands every earlier row in `zone_other`."""
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
    """Cached `features.unmatched_away_states`. Never raises: a source that
    cannot answer must not take the status page down."""
    computed_at, cached = _state["unmatched_zones"]
    now = time.time()
    if computed_at is not None and now - computed_at < UNMATCHED_TTL_S:
        return cached
    # One scan at a time: the page polls every few seconds and the scan reads
    # every person's whole history.
    if not _unmatched_lock.acquire(blocking=False):
        return cached
    try:
        source = _state["source"]
        found = features.unmatched_away_states(
            source, features.history_start(source), None)
        _state["unmatched_zones"] = (now, found)
    except Exception as err:  # noqa: BLE001
        # Keep the old answer, but retry in a minute rather than a quarter hour.
        _log.debug("unmatched-zone scan failed: %s", err)
        found = cached
        _state["unmatched_zones"] = (now - UNMATCHED_TTL_S + 60, cached)
    finally:
        _unmatched_lock.release()
    return found


_unmatched_lock = threading.Lock()

# `store.span()` is a full-archive aggregate and `_status` ran it twice per
# poll; thirty seconds of staleness is invisible.
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
    """The holidays row, which has to say where the calendar came from: a bare
    country code reads like something the add-on decided."""
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

    503 while the worker is stalled or HA was unreachable at start-up.
    `watchdog:` in config.yaml points here, and a non-2xx is what restarts the
    add-on.
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

    Read out of `_state`, not recomputed, so the panel and the HA entities
    cannot disagree and opening a tab never costs a rebuild.
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
            # A fraction of the last five minutes spent at home, not a forecast
            # -- the panel shows it as "actually", beside the prediction.
            "current": r["current"],
            # The slot the horizons are measured FROM, which is not
            # `predicted_at` and can be half an hour older; the chart's clock
            # labels must use it. `.get`, so a missing anchor costs the labels
            # and not the endpoint.
            "observed_at": r.get("observed_at"),
            "curve": r["curve"],
            "next_departure_h": r["next_departure_h"],
            "next_arrival_h": r["next_arrival_h"],
            "eta_minutes": r["eta_minutes"],
            # The routine for today, or null: this person's own history for
            # this weekday, deliberately NOT part of `curve`.
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

    `bool` is rejected explicitly: JSON `true` arrives as an int and would be
    clamped into a legal-looking cut.
    """
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(
            status_code=400, detail=f"{key} must be {what}, not {value!r}.")
    return float(value)


def crossing_patch(payload: dict, current: config.Settings) -> dict:
    """The validated crossing cuts in `payload`, as a dict to apply. Or a 400.

    Separate so it is testable without HTTP and runs BEFORE anything is
    assigned.
    """
    out: dict = {}
    for key in ("departure_threshold", "arrival_threshold"):
        if key not in payload:
            continue
        value = _number(payload, key, "a probability")
        # Open at both ends: a cut of exactly 0 or 1 can never be met by a
        # rounded curve.
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

    # Checked against the merge, not the patch, so a one-key save cannot invert
    # the band.
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

    The endpoint's schema is `dict`, so without this a bare string iterates
    into one-letter subjects. Every list field goes through here.
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
    """The self-contained fields of a config patch, type-checked. Or a 400.

    Shapes only: whether an entity exists needs HA and belongs to the endpoint.
    Fields read against the CURRENT settings are in `crossing_patch`.
    """
    out: dict = {}
    if "forecast_retention_days" in payload:
        value = _number(payload, "forecast_retention_days", "a whole number of days")
        # The floor is derived: under the longest horizon a forecast is pruned
        # before it can come due. 0 prunes nothing.
        floor = -(-max(config.HORIZONS_H) // 24)
        if value != int(value) or value < 0 or 0 < value < floor:
            raise HTTPException(
                status_code=400,
                detail=f"forecast_retention_days must be 0 (keep everything) "
                       f"or at least {floor} whole days -- shorter than that "
                       f"and a +{max(config.HORIZONS_H)} h forecast is deleted "
                       f"before it can be scored.")
        out["forecast_retention_days"] = int(value)
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
    """Replace the configuration and rebuild from it; the models are left in
    place and the next train replaces them.

    EVERYTHING is validated on a COPY, and the live settings are swapped only
    once `config.configure` accepts it -- validating second once left the
    process running on values that never reached disk.
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

    # Rejected rather than absorbed: nothing downstream distinguishes "zone
    # nobody visited" from "zone that does not exist", and the collector would
    # ask forever for a person HA has no entity for.
    live = {s["entity_id"] for s in _state["ha"].states()}
    missing = [p for p in candidate.people if p not in live]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"not people Home Assistant knows about: {', '.join(missing)}")
    # Same rule as the zones: a schedule that does not exist leaves the chart
    # unshaded with no explanation.
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

    # Rejected loudly rather than absorbed: the feature build degrades an
    # unknown country to zeros, which is wrong here.
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
    # The log outlives a save, including one that switches `source`.
    if _state["forecast_log"] is None:
        _state["forecast_log"] = runtime.forecast_log()
    _state["source"] = runtime.build_source(settings, _state["ha"],
                                            _state["forecast_log"])
    # Who to listen to changed with who to track; without this a person added
    # here is not subscribed until a restart.
    if _listener is not None:
        _listener.update_entities(runtime.trigger_entities(settings))
    # A save is also how a refused start-up gets going: the worker was never
    # started. No-op on an add-on that is already running.
    _start_worker_threads(settings)

    # Somebody removed: clear their retained entities, or HA keeps them forever
    # under a `predicted_at` that never moves. Either way the models are about
    # a different house, so retrain now.
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
# Thin, like the rest: the reading lives in `explore.py` so it is testable
# without an HTTP client. None of these is polled -- they are sync handlers in
# the threadpool.

@app.get("/api/explore/archive")
def api_explore_archive() -> dict:
    return explore.archive_inventory(_state["source"], _state["settings"])


@app.get("/api/explore/entity")
def api_explore_entity(entity_id: str, days: int = explore.DEFAULT_DAYS) -> dict:
    return explore.entity_series(_state["source"], _state["settings"],
                                 entity_id, days)


# Keyed on (path, mtime_ns), so a retrain drops the entry with nothing to
# invalidate. The archive is deliberately NOT cached: a stale row count is what
# that card exists to report.
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
    """A horizon on the grid, or a 404."""
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


# Uncached: it reads the archive and the forecast table, both rewritten every
# five minutes.
@app.get("/api/explore/verification")
def api_explore_verification(subject: str, horizon: int,
                             days: int = explore.DEFAULT_DAYS) -> dict:
    return explore.verification(_state["source"], _state["forecast_log"],
                                _state["settings"], subject, horizon, days)


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

    The default stays synchronous for anything already scripted against it.
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
