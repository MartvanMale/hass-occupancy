"""One place that decides what the add-on says, and how loudly.

**Why this exists.** Over the two days journald kept before this was written,
the add-on emitted two lines per start -- `Starting` and `ready: N model(s)` --
and nothing else, ever. No line for a collect, a predict, a publish, a train, an
MQTT connect or a listener reconnect. A healthy add-on and a hung one produced
byte-identical logs, which is why an 11.5 hour stall on 2026-09-01 was found by
looking at a forecast-accuracy chart rather than by looking at the log. Silence
has to mean something, and it could not.

**The format matches bashio's**, `[HH:MM:SS] LEVEL: message`, because `run.sh`
and Supervisor already write that shape and the add-on's Log tab renders raw
stdout with no timestamps of its own. Lines in a different shape read as
orphans floating between timestamped ones. Local time, for the same reason: it
is what the surrounding lines use, even though everything else in this package
works in UTC.

**The level comes from the add-on option**, which existed in `config.yaml` --
`list(trace|debug|info|notice|warning|error|fatal)`, offered in the
Configuration tab -- and was read by nothing at all. `run.sh` now exports it.
Bashio has seven levels and Python has five; `trace` and `notice` are added so
the option cannot offer a value that silently rounds to another one.

**Third-party loggers are quietened deliberately.** uvicorn would log every
`/api/status` poll and the panel polls every few seconds; paho and websockets
are chatty about routine reconnects. They are pinned at WARNING until the
add-on is set to `debug`, at which point they are exactly what you want.
"""
from __future__ import annotations

import logging
import os
import sys
import time

TRACE = 5
NOTICE = 25

# bashio's vocabulary, mapped onto Python's. `fatal` is CRITICAL under another
# name; `trace` and `notice` are registered below so they are real levels
# rather than aliases that lose their name in the output.
LEVELS = {
    "trace": TRACE,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": NOTICE,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
}

# Noisy at INFO and worth every line at DEBUG.
THIRD_PARTY = ("uvicorn", "uvicorn.access", "uvicorn.error", "paho",
               "asyncio", "urllib3")

# NEVER follows the add-on's level, at any setting.
#
# `websockets` logs every frame at DEBUG, and the first frame of every
# connection is the authentication one:
#
#     DEBUG: > TEXT '{"type": "auth", "access_token": "5183aa...ebea"}'
#
# The library elides the middle of a long frame, so what lands in the journal is
# a partial token rather than a usable one -- but a credential has no business
# being in a log at all, and the add-on's own log level is a user-facing option
# that must not be a way to put one there.
#
# It is also unusable as diagnostics: the keepalive is every 20 seconds and each
# one costs four lines, so a `debug` session drowns in PING/PONG. WARNING keeps
# the connection errors, which are the part with any value.
SECRET_BEARING = ("websockets", "websockets.client", "websockets.protocol")

_configured = False
_handler: logging.Handler | None = None


class Formatter(logging.Formatter):
    """`[HH:MM:SS] LEVEL: message`, the shape bashio already writes."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        line = f"[{stamp}] {record.levelname}: {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def level_from(name: str | None) -> int:
    """A bashio level name to a Python level, tolerantly.

    An unreadable option must not decide the add-on cannot start, and must not
    silently pick something quieter than asked for either -- an unknown name
    falls back to INFO, which is the schema's own default.
    """
    return LEVELS.get((name or "").strip().lower(), logging.INFO)


def configure(level: str | int | None = None, stream=None) -> int:
    """Install the handler. Idempotent, so a CLI entry point may call it too.

    Returns the level actually applied, which is what a test asserts on.
    """
    global _configured
    logging.addLevelName(TRACE, "TRACE")
    logging.addLevelName(NOTICE, "NOTICE")

    if isinstance(level, int):
        applied = level
    else:
        applied = level_from(level if level is not None else os.environ.get("LOG_LEVEL"))

    root = logging.getLogger()
    # Replace OUR handler, never everyone's. An earlier version assigned
    # `root.handlers = [handler]`, which is the obvious way to stop a second
    # call doubling every line and is wrong: it also silently removes handlers
    # that belong to whatever is hosting us -- pytest's capture, for one, so the
    # tests could no longer see what the add-on logged.
    global _handler
    if _handler is not None:
        root.removeHandler(_handler)
    _handler = logging.StreamHandler(stream or sys.stdout)
    _handler.setFormatter(Formatter())
    root.addHandler(_handler)
    _configured = True
    root.setLevel(applied)

    for name in THIRD_PARTY:
        logging.getLogger(name).setLevel(
            applied if applied <= logging.DEBUG else logging.WARNING)
    for name in SECRET_BEARING:
        logging.getLogger(name).setLevel(logging.WARNING)
    return applied


def get(name: str) -> logging.Logger:
    """A logger for one module. `log.get(__name__)` at the top of a file."""
    return logging.getLogger(name)
