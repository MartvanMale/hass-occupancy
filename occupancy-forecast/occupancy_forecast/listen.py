"""Event-driven wake-ups from Home Assistant.

An add-on has no trigger system of its own -- it is a container, not an
integration -- so the only way to hear that somebody came home, rather than
poll for it, is Home Assistant's WebSocket API. Inside an add-on that is
reachable through the Supervisor proxy at `ws://supervisor/core/websocket`
with `SUPERVISOR_TOKEN`, gated by the same `homeassistant_api: true` in
config.yaml that already grants `/core/api`. Outside one (tests, a laptop) it
falls back to HA_URL + HA_TOKEN, exactly like `sources.ha.HomeAssistant`.

**`subscribe_trigger`, not `subscribe_events`.** It takes the same trigger
schema an automation does and Home Assistant evaluates it server-side, so we
receive only the entities we asked about. Subscribing to `state_changed`
instead would deliver every state change in the house -- hundreds a minute on
a real installation -- to be JSON-decoded and thrown away.

This is the add-on's own answer to what pyscript's `@state_trigger` used to do
from the outside. The difference that matters: it ships *with* the add-on, so
a stranger installing it from GitHub gets the same responsiveness without
having to install pyscript and copy a file.

ADVISORY ONLY, like everything else here. This module subscribes and reads. It
never calls a service and never writes anything back to Home Assistant.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from typing import Callable

from . import log

_log = log.get(__name__)

SUPERVISOR_WS = "ws://supervisor/core/websocket"

# Backoff between reconnection attempts, seconds. A Home Assistant restart
# drops the socket and then makes the Supervisor proxy refuse for a while, so
# the first few failures are entirely normal and must not flood the log.
BACKOFF_START = 1.0
BACKOFF_CAP = 60.0

# How long to block in recv before checking whether we have been asked to stop.
# Only affects shutdown latency; the library's own ping keepalive is what
# actually detects a dead peer.
RECV_TIMEOUT = 5.0

# States that mean "no reading", not "a new reading". `StoreSource.collect`
# already drops these, so waking the worker for one would rebuild a month of
# features to arrive at the answer it already published.
EMPTY_STATES = {"unknown", "unavailable", "", "none"}


def should_fire(trigger: dict) -> bool:
    """True for a real state change, false for an attribute-only one.

    A state trigger with no `to`/`from` fires on ANY change to the state
    object, and a person entity rewrites its GPS attributes every few minutes
    while somebody is driving. Comparing the state strings is what separates
    "they came home" from "the phone moved forty metres" -- and the difference
    is a full feature rebuild per event.

    Kept as a module-level pure function on purpose: it is the only part of
    this file with a decision in it, and it is testable without a socket.
    """
    if not isinstance(trigger, dict):
        return False

    def state_of(side: str) -> str | None:
        value = trigger.get(side)
        if not isinstance(value, dict):
            return None
        state = value.get("state")
        return None if state is None else str(state).strip().lower()

    old, new = state_of("from_state"), state_of("to_state")
    if new is None or new in EMPTY_STATES:
        return False
    if old is None or old in EMPTY_STATES:
        # First sighting after a restart, or recovery from unavailable. The
        # value is genuinely new to us even though nothing "changed".
        return True
    return old != new


class Listener:
    """A Home Assistant trigger subscription, on its own thread.

    `on_event` is called with no arguments for every trigger that survives
    `should_fire`. It must be cheap and must not raise -- setting a
    `threading.Event` is what this was built for. Anything slower belongs on
    the worker that the event wakes.

    A dead listener is a latency regression, never an outage: the caller keeps
    its own periodic poll, and `status` is surfaced on the ingress panel so
    the degradation is visible rather than silent.
    """

    def __init__(self, entity_ids: list[str], on_event: Callable[[], None],
                 url: str | None = None, token: str | None = None):
        supervisor = os.environ.get("SUPERVISOR_TOKEN")
        if url is None and supervisor:
            url, token = SUPERVISOR_WS, supervisor
        if url is None:
            base = os.environ.get("HA_URL", "").rstrip("/")
            if base:
                url = base.replace("https://", "wss://", 1).replace(
                    "http://", "ws://", 1) + "/api/websocket"
        self.url = url
        self.token = token or os.environ.get("HA_TOKEN", "")
        self.entity_ids = sorted(set(entity_ids))
        self.on_event = on_event

        self.connected = False
        self.last_event: str | None = None
        self.last_error: str | None = None
        self.events = 0
        self.fired = 0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Begin listening. Never raises: a failure here must not stop the add-on."""
        if not self.url or not self.token:
            self.last_error = ("no Home Assistant to listen to: expected "
                               "SUPERVISOR_TOKEN or HA_URL + HA_TOKEN")
            return
        if not self.entity_ids:
            self.last_error = "nothing to subscribe to"
            return
        self._thread = threading.Thread(target=self._run, name="occupancy-listener",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._drop_socket()

    def update_entities(self, entity_ids: list[str]) -> None:
        """Re-subscribe to a different set, after the configuration changed.

        A subscription cannot be edited in place, so this drops the socket and
        lets the run loop reconnect -- which it already knows how to do, and
        which re-reads `entity_ids` on the way through. Without this, adding a
        person on the settings page leaves them uncovered until a restart, and
        nothing says so.
        """
        wanted = sorted(set(entity_ids))
        if wanted == self.entity_ids:
            return
        self.entity_ids = wanted
        self._drop_socket()

    def _drop_socket(self) -> None:
        """Make a blocking `recv` on the listener thread return. Never raises."""
        socket, self._socket = self._socket, None
        if socket is None:
            return
        try:
            socket.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def status(self) -> dict:
        return {
            "connected": self.connected,
            "entities": len(self.entity_ids),
            "events": self.events,
            "fired": self.fired,
            "last_event": self.last_event,
            "last_error": self.last_error,
        }

    # -- the thread ---------------------------------------------------------

    def _run(self) -> None:
        backoff = BACKOFF_START
        while not self._stop.is_set():
            try:
                self._session()
                backoff = BACKOFF_START      # a clean session resets the ladder
            except Exception as err:  # noqa: BLE001
                self.last_error = f"{_now()}: {err}"
            finally:
                self.connected = False
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, BACKOFF_CAP)

    def _session(self) -> None:
        """One connection, from handshake to disconnect.

        The import is here rather than at module scope so that a missing or
        broken `websockets` degrades to "no listener, five-minute poll" with
        the reason on the status page, instead of stopping the add-on from
        importing at all.
        """
        from websockets.sync.client import connect

        with connect(self.url, open_timeout=30, close_timeout=5) as socket:
            self._authenticate(socket)
            self._subscribe(socket)
            self._socket = socket
            # Transitions only: a reconnect after a drop is worth a line, the
            # first connect at boot is worth one, and a socket that simply
            # stays up is worth none.
            _log.info("subscribed to %d Home Assistant %s",
                      len(self.entity_ids),
                      "entity" if len(self.entity_ids) == 1 else "entities")
            self.connected = True
            self.last_error = None
            try:
                while not self._stop.is_set() and self._socket is socket:
                    try:
                        raw = socket.recv(timeout=RECV_TIMEOUT)
                    except TimeoutError:
                        continue
                    self._handle(json.loads(raw))
            finally:
                if self._socket is socket:
                    self._socket = None

    def _authenticate(self, socket) -> None:
        hello = json.loads(socket.recv(timeout=30))
        if hello.get("type") != "auth_required":
            raise RuntimeError(f"expected auth_required, got {hello.get('type')!r}")
        socket.send(json.dumps({"type": "auth", "access_token": self.token}))
        reply = json.loads(socket.recv(timeout=30))
        if reply.get("type") != "auth_ok":
            raise RuntimeError(f"authentication refused: {reply.get('message') or reply}")

    def _subscribe(self, socket) -> None:
        socket.send(json.dumps({
            "id": 1,
            "type": "subscribe_trigger",
            "trigger": {"platform": "state", "entity_id": self.entity_ids},
        }))
        reply = json.loads(socket.recv(timeout=30))
        if reply.get("type") == "result" and not reply.get("success", False):
            raise RuntimeError(f"subscribe_trigger refused: {reply.get('error') or reply}")

    def _handle(self, message: dict) -> None:
        if message.get("type") != "event":
            return
        trigger = (((message.get("event") or {}).get("variables") or {})
                   .get("trigger") or {})
        self.events += 1
        if not should_fire(trigger):
            return
        self.fired += 1
        self.last_event = _now()
        try:
            self.on_event()
        except Exception as err:  # noqa: BLE001
            # The callback is the caller's problem, but it must not take the
            # subscription down with it -- that would trade a slow forecast
            # for no event-driven forecast at all.
            self.last_error = f"{_now()}: on_event: {err}"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
