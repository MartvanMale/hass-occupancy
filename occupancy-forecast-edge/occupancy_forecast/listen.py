"""Event-driven wake-ups from Home Assistant's WebSocket API; it only reads.

Via the Supervisor proxy inside an add-on, HA_URL + HA_TOKEN outside one.
`subscribe_trigger`, not `subscribe_events`: HA filters it server-side, where
`state_changed` would deliver every change in the house to be thrown away.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from typing import Callable

from . import config, log

_log = log.get(__name__)

SUPERVISOR_WS = "ws://supervisor/core/websocket"

# Seconds. An HA restart drops the socket and then the proxy refuses for a
# while, so the first few failures are normal and must not flood the log.
BACKOFF_START = 1.0
BACKOFF_CAP = 60.0

# Only affects shutdown latency; the ping keepalive detects a dead peer.
RECV_TIMEOUT = 5.0

# "No reading", not a new one: waking on it would rebuild a month of features
# to arrive at the answer already published.
EMPTY_STATES = config.EMPTY_STATES


def should_fire(trigger: dict) -> bool:
    """True for a real state change, false for an attribute-only one.

    A bare state trigger also fires on every GPS update: a feature rebuild each.
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

    `on_event` must be cheap and must not raise; setting a `threading.Event` is
    what it is for. A dead listener is a latency regression, never an outage.
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
        self.last_error_public: str | None = None
        self.events = 0
        self.fired = 0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Begin listening. Never raises: a failure here must not stop the add-on."""
        if not self.url or not self.token:
            self._record("no Home Assistant to listen to: expected "
                         "SUPERVISOR_TOKEN or HA_URL + HA_TOKEN")
            return
        if not self.entity_ids:
            self._record("nothing to subscribe to")
            return
        self._thread = threading.Thread(target=self._run, name="occupancy-listener",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._drop_socket()

    def update_entities(self, entity_ids: list[str]) -> None:
        """Re-subscribe to a different set, after the configuration changed.

        A subscription cannot be edited, so this drops the socket to reconnect.
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

    def _record(self, text: str | None, public: str | None = None) -> None:
        """Remember a reason. `public` is what `/api/status` may show, and
        defaults to `text` -- right for the reasons written here, not for an
        exception's own message."""
        self.last_error = text
        self.last_error_public = text if public is None else public

    @property
    def status(self) -> dict:
        return {
            "connected": self.connected,
            "entities": len(self.entity_ids),
            "events": self.events,
            "fired": self.fired,
            "last_event": self.last_event,
            "last_error": self.last_error_public,
        }

    # -- the thread ---------------------------------------------------------

    def _run(self) -> None:
        backoff = BACKOFF_START
        while not self._stop.is_set():
            try:
                self._session()
                backoff = BACKOFF_START      # a clean session resets the ladder
            except Exception as err:  # noqa: BLE001
                # Logged here: the status page shows only the stamp.
                _log.warning("listener session ended: %s. Retrying in %ss.",
                             err, backoff, exc_info=True)
                self._record(f"{_now()}: {err}", f"{_now()}: {log.SEE_THE_LOG}")
            finally:
                self.connected = False
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, BACKOFF_CAP)

    def _session(self) -> None:
        """One connection, from handshake to disconnect.

        Imported here: a missing `websockets` degrades to polling, not a crash.
        """
        from websockets.sync.client import connect

        with connect(self.url, open_timeout=30, close_timeout=5) as socket:
            # Published BEFORE the handshake, so an `update_entities` landing
            # during it can drop this socket and the reconnect gets the new set.
            self._socket = socket
            self._authenticate(socket)
            self._subscribe(socket)
            if self._socket is not socket:
                return                          # reconfigured mid-handshake
            # Transitions only: a connect is worth a line, a socket that stays
            # up is worth none.
            _log.info("subscribed to %d Home Assistant %s",
                      len(self.entity_ids),
                      "entity" if len(self.entity_ids) == 1 else "entities")
            self.connected = True
            self._record(None)
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
            # The callback must not take the subscription down with it.
            _log.warning("the listener's callback raised: %s", err, exc_info=True)
            self._record(f"{_now()}: on_event: {err}",
                         f"{_now()}: {log.SEE_THE_LOG}")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
