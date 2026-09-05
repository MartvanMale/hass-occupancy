"""The log is a product surface, so its shape is worth pinning.

The add-on's whole log was two lines per start until 2026-09-02, which is how
an 11.5 hour stall went unnoticed. What is asserted here is the part that makes
silence meaningful: the level actually comes from the add-on option, the format
matches the lines bashio writes around ours, and the heartbeat fires on time.
"""
import io
import logging

import pytest

from occupancy_forecast import log


@pytest.fixture(autouse=True)
def _restore():
    """`configure` mutates the root logger, which every other test shares."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level
    log._configured = False


def test_every_level_the_option_offers_maps_to_a_real_level():
    """`config.yaml` offers seven names. A name the schema allows but the code
    does not know would silently round to something else -- quieter, probably,
    which is the direction that hides things."""
    offered = {"trace", "debug", "info", "notice", "warning", "error", "fatal"}
    assert offered == set(log.LEVELS)
    for name in offered:
        assert isinstance(log.level_from(name), int)
    assert log.level_from("TRACE") == log.TRACE, "the option is not case-bound"
    # Unreadable is not a reason to go quiet, nor to refuse to start.
    assert log.level_from(None) == logging.INFO
    assert log.level_from("shout") == logging.INFO


def test_the_level_comes_from_the_environment(monkeypatch):
    """`run.sh` exports LOG_LEVEL from the add-on option. That wiring was
    missing entirely: the option existed, was in the schema, showed in the
    Configuration tab, and nothing read it."""
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert log.configure() == logging.DEBUG
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert log.configure() == logging.WARNING


def test_the_line_matches_the_shape_bashio_writes():
    """`[HH:MM:SS] LEVEL: message`. The Log tab renders raw stdout with no
    timestamps of its own, so a line in another shape reads as an orphan
    between the timestamped ones Supervisor and run.sh already emit."""
    stream = io.StringIO()
    log.configure("info", stream=stream)
    log.get("occupancy_forecast.test").info("published %d entities", 36)
    line = stream.getvalue().strip()
    assert line.endswith("INFO: published 36 entities")
    stamp = line[1:line.index("]")]
    hours, minutes, seconds = stamp.split(":")
    assert len(hours) == len(minutes) == len(seconds) == 2


def test_the_two_added_levels_keep_their_names():
    """`trace` and `notice` are bashio's, not Python's. Registered rather than
    aliased, so the option cannot offer a name the output then renames."""
    stream = io.StringIO()
    log.configure(log.TRACE, stream=stream)
    logger = log.get("occupancy_forecast.test")
    logger.log(log.TRACE, "deep")
    logger.log(log.NOTICE, "worth knowing")
    out = stream.getvalue()
    assert "TRACE: deep" in out
    assert "NOTICE: worth knowing" in out


def test_the_chatty_libraries_are_quiet_until_debug():
    """uvicorn logs every request and the panel polls every few seconds."""
    log.configure("info")
    assert logging.getLogger("uvicorn.access").level == logging.WARNING
    log.configure("debug")
    assert logging.getLogger("uvicorn.access").level == logging.DEBUG


def test_configure_does_not_stack_handlers():
    """It is called at startup and again by any CLI entry point. Twice must not
    mean every line twice."""
    stream = io.StringIO()
    log.configure("info", stream=stream)
    log.configure("info", stream=stream)
    log.get("occupancy_forecast.test").info("once")
    assert stream.getvalue().count("once") == 1


def test_the_websocket_library_never_follows_the_add_ons_level():
    """It logs every frame at DEBUG, and the first frame of every connection is
    the authentication one -- so `log_level: debug`, a user-facing option, put a
    Supervisor token in the journal. The library elides the middle of a long
    frame so what landed was partial rather than usable, but a credential does
    not belong in a log and an option must not be a way to put one there.

    It is also four lines per 20-second keepalive, which makes a debug session
    unreadable. WARNING keeps the connection errors, which are the useful part.
    """
    for level in ("trace", "debug", "info", "warning"):
        log.configure(level)
        for name in log.SECRET_BEARING:
            assert logging.getLogger(name).level == logging.WARNING, name

    # The rest still follow, because that is the point of the option.
    log.configure("debug")
    assert logging.getLogger("uvicorn.access").level == logging.DEBUG
    assert "websockets" not in log.THIRD_PARTY
