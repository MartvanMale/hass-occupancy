"""Trigger-subscription tests. The socket is not the interesting part -- the
filter is: a state trigger with no `to`/`from` fires on ANY change, and each
wake is a 32-day feature rebuild.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from occupancy_forecast import listen, runtime  # noqa: E402

from .conftest import settings as make_settings  # noqa: E402


def _trigger(old: str | None, new: str | None) -> dict:
    """A `subscribe_trigger` state payload, as Home Assistant sends it."""
    return {
        "platform": "state",
        "entity_id": "person.alice",
        "from_state": None if old is None else {"state": old, "attributes": {}},
        "to_state": None if new is None else {"state": new, "attributes": {}},
    }


# ---------------------------------------------------------------------------
# should_fire
# ---------------------------------------------------------------------------

def test_a_real_state_change_fires():
    assert listen.should_fire(_trigger("not_home", "home"))


def test_an_attribute_only_change_does_not_fire():
    """The one that matters. A person entity rewrites its GPS attributes every
    few minutes while somebody drives, and HA reports each one as a state
    trigger; waking on them would rebuild features for a whole commute."""
    assert not listen.should_fire(_trigger("not_home", "not_home"))


def test_going_unavailable_does_not_fire():
    """`unavailable` is the absence of a reading, not a new one. `StoreSource`
    drops these states already, so a re-predict would only republish."""
    for empty in ("unavailable", "unknown", ""):
        assert not listen.should_fire(_trigger("home", empty)), empty


def test_coming_back_from_unavailable_fires():
    """Recovery is genuinely new information, even though nothing 'changed'."""
    assert listen.should_fire(_trigger("unavailable", "home"))


def test_a_first_sighting_fires():
    """No `from_state` at all -- the entity was created, or HA just restarted."""
    assert listen.should_fire(_trigger(None, "home"))


def test_junk_is_survivable():
    """Nothing off the wire may take the subscription down."""
    for junk in (None, {}, "nonsense", {"to_state": "not-a-dict"}, _trigger(None, None)):
        assert listen.should_fire(junk) is False, junk


# ---------------------------------------------------------------------------
# Which entities we ask about
# ---------------------------------------------------------------------------

def test_proximity_sensors_are_not_subscribed_to():
    """They rewrite every few minutes while somebody drives, and are averaged
    over a 30-minute slot anyway. This is the difference between
    `trigger_entities` and `tracked_entities`."""
    config = make_settings()
    tracked = runtime.tracked_entities(config)
    triggers = runtime.trigger_entities(config)

    assert "sensor.home_alice_distance" in tracked
    assert "sensor.home_alice_distance" not in triggers
    assert "sensor.home_alice_direction_of_travel" not in triggers
    assert set(triggers) <= set(tracked)
    assert {"person.alice", "person.bob", "group.household",
            "zone.alice_office"} == set(triggers)


# ---------------------------------------------------------------------------
# The handshake, against a fake socket
# ---------------------------------------------------------------------------

class FakeSocket:
    """Scripted replies, and a record of what was sent."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.sent: list[dict] = []

    def recv(self, timeout=None):
        if not self.replies:
            raise TimeoutError
        return json.dumps(self.replies.pop(0))

    def send(self, raw):
        self.sent.append(json.loads(raw))


def _listener(**kwargs) -> listen.Listener:
    return listen.Listener(["person.alice"], lambda: None,
                           url="ws://test/api/websocket", token="secret", **kwargs)


def test_the_handshake_is_in_the_order_ha_expects():
    socket = FakeSocket([{"type": "auth_required", "ha_version": "2026.8.3"},
                         {"type": "auth_ok"}])
    listener = _listener()
    listener._authenticate(socket)
    assert socket.sent == [{"type": "auth", "access_token": "secret"}]


def test_a_refused_token_raises_rather_than_hanging():
    socket = FakeSocket([{"type": "auth_required"},
                         {"type": "auth_invalid", "message": "Invalid access token"}])
    with pytest.raises(RuntimeError, match="Invalid access token"):
        _listener()._authenticate(socket)


def test_the_subscription_asks_for_a_state_trigger():
    socket = FakeSocket([{"id": 1, "type": "result", "success": True, "result": None}])
    listener = _listener()
    listener._subscribe(socket)
    assert socket.sent == [{"id": 1, "type": "subscribe_trigger",
                            "trigger": {"platform": "state",
                                        "entity_id": ["person.alice"]}}]


def test_a_refused_subscription_raises():
    socket = FakeSocket([{"id": 1, "type": "result", "success": False,
                          "error": {"code": "invalid_format"}}])
    with pytest.raises(RuntimeError, match="invalid_format"):
        _listener()._subscribe(socket)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _event(trigger: dict) -> dict:
    return {"id": 1, "type": "event", "event": {"variables": {"trigger": trigger}}}


def test_a_state_change_sets_the_event():
    nudge = threading.Event()
    listener = listen.Listener(["person.alice"], nudge.set,
                               url="ws://test", token="secret")
    listener._handle(_event(_trigger("not_home", "home")))
    assert nudge.is_set()
    assert listener.fired == 1 and listener.events == 1


def test_an_attribute_change_is_counted_but_not_fired():
    """`events` minus `fired` is what tells you the filter is doing its job."""
    nudge = threading.Event()
    listener = listen.Listener(["person.alice"], nudge.set,
                               url="ws://test", token="secret")
    listener._handle(_event(_trigger("not_home", "not_home")))
    assert not nudge.is_set()
    assert listener.events == 1 and listener.fired == 0


def test_a_raising_callback_does_not_kill_the_subscription():
    """Better a recorded error than trading a slow forecast for no listener."""
    def boom():
        raise RuntimeError("callback exploded")

    listener = listen.Listener(["person.alice"], boom,
                               url="ws://test", token="secret")
    listener._handle(_event(_trigger("not_home", "home")))
    assert "callback exploded" in (listener.last_error or "")


def test_non_event_messages_are_ignored():
    """Pongs, results and anything else HA decides to send."""
    nudge = threading.Event()
    listener = listen.Listener(["person.alice"], nudge.set,
                               url="ws://test", token="secret")
    for message in ({"type": "result", "success": True}, {"type": "pong"}, {}):
        listener._handle(message)
    assert not nudge.is_set() and listener.events == 0


# ---------------------------------------------------------------------------
# Failure posture
# ---------------------------------------------------------------------------

def test_no_credentials_records_a_reason_and_starts_nothing():
    """A listener that cannot run must degrade to the poll, never raise."""
    listener = listen.Listener(["person.alice"], lambda: None, url=None, token="")
    listener.start()
    assert listener._thread is None
    assert not listener.connected
    assert listener.last_error and "SUPERVISOR_TOKEN" in listener.last_error


def test_nothing_to_subscribe_to_is_not_an_error_worth_raising():
    listener = listen.Listener([], lambda: None, url="ws://test", token="secret")
    listener.start()
    assert listener._thread is None
    assert listener.last_error == "nothing to subscribe to"


def test_the_status_holds_back_a_message_this_module_did_not_write(caplog):
    """`/api/status` needs no login and a callback's exception can name a token
    or a URL, so the status page gets the stamp and the log gets the rest. A
    reason written here is not a secret and still shows."""
    def boom():
        raise RuntimeError("rejected: token abc123 for /api/websocket")

    listener = listen.Listener(["person.alice"], boom, url="ws://test", token="secret")
    caplog.set_level("WARNING")
    listener._handle(_event(_trigger("not_home", "home")))

    assert "abc123" in (listener.last_error or ""), "kept for the log"
    assert "abc123" not in listener.status["last_error"]
    assert any(r.exc_info for r in caplog.records), "logged with its traceback"

    quiet = listen.Listener([], lambda: None, url="ws://test", token="secret")
    quiet.start()
    assert quiet.status["last_error"] == "nothing to subscribe to"


def test_connecting_logs_what_it_subscribed_to(monkeypatch, caplog):
    """`_session` is the only place `connected` is set and had no test: a log
    line naming an attribute that does not exist raised on every subscription
    attempt, and the only sign was `last_error`."""
    import contextlib

    from websockets.sync import client as ws_client

    socket = FakeSocket([
        {"type": "auth_required"},
        {"type": "auth_ok"},
        {"id": 1, "type": "result", "success": True, "result": None},
    ])

    @contextlib.contextmanager
    def fake_connect(url, **kwargs):
        yield socket

    monkeypatch.setattr(ws_client, "connect", fake_connect)
    caplog.set_level("INFO")

    listener = _listener()
    listener._stop.set()          # handshake and subscribe, then no receive loop
    listener._session()

    assert listener.connected, "the connect path did not complete"
    assert listener.last_error is None
    assert any("subscribed to 1 Home Assistant entity" in r.getMessage()
               for r in caplog.records)


def test_a_reconfigure_during_the_handshake_ends_that_session(monkeypatch):
    """`_socket` is published BEFORE the handshake. Assigned after the subscribe,
    an `update_entities` landing in that window found nothing to drop, and a
    person added on the settings page was not listened to until a restart."""
    import contextlib

    from websockets.sync import client as ws_client

    socket = FakeSocket([
        {"type": "auth_required"},
        {"type": "auth_ok"},
        {"id": 1, "type": "result", "success": True, "result": None},
    ])

    @contextlib.contextmanager
    def fake_connect(url, **kwargs):
        yield socket

    monkeypatch.setattr(ws_client, "connect", fake_connect)
    listener = _listener()
    real_authenticate = listener._authenticate

    def authenticate_then_reconfigure(sock):
        real_authenticate(sock)
        assert listener._socket is sock, "the socket is droppable during the handshake"
        listener.update_entities(["person.alice", "person.bob"])

    monkeypatch.setattr(listener, "_authenticate", authenticate_then_reconfigure)
    listener._session()

    assert not listener.connected, "the reconfigured session must not be reported up"
    assert listener._socket is None
    assert listener.entity_ids == ["person.alice", "person.bob"], \
        "the next session subscribes to the new set"
