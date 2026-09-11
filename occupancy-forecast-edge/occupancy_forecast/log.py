"""One place that decides what the add-on says, and how loudly.

Lines match bashio's `[HH:MM:SS] LEVEL: message` in local time, because the Log
tab renders raw stdout beside Supervisor's own lines. The level comes from the
add-on option; third-party loggers stay at WARNING until it is set to `debug`.
"""
from __future__ import annotations

import logging
import os
import sys
import time

TRACE = 5
NOTICE = 25

# bashio's vocabulary; `trace` and `notice` are registered below as real
# levels rather than aliases that lose their name in the output.
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

# NEVER follows the add-on's level: `websockets` logs every frame at DEBUG, so
# `debug` would put the auth handshake's Supervisor token in the journal.
SECRET_BEARING = ("websockets", "websockets.client", "websockets.protocol")

# Said in place of an exception's own message, which can carry a /data path or
# a broker address: `/api/status` and the Data tab are readable by any HA user.
SEE_THE_LOG = "the add-on log has the error"

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
    """A bashio level name to a Python level; an unknown name is INFO.

    An unreadable option must neither stop the add-on starting nor silently pick
    something quieter than asked for.
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
    # Replace OUR handler, never everyone's: `root.handlers = [handler]` would
    # also drop handlers owned by whatever hosts us, pytest's capture included.
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
