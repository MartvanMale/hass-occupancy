"""The read interface every source implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Source(Protocol):
    """Three reads, keyed on entity id.

    Timestamps are returned as ISO-8601 strings because that is what
    `features.py` already parses (`pd.to_datetime(..., format="ISO8601")`), and
    keeping the shape means the feature builder does not care which source it
    is talking to.
    """

    def states(self, entity_id: str, start: str,
               stop: str | None = None) -> list[tuple[str, str]]:
        """Raw string states in [start, stop)."""

    def seeded_states(self, entity_id: str, start: str, stop: str | None = None,
                      seed_days: int = 14) -> list[tuple[str, str]]:
        """`states`, prefixed with the last value before `start`, re-stamped at `start`."""

    def numeric(self, entity_id: str, start: str,
                stop: str | None = None) -> list[tuple[str, float]]:
        """Numeric states in [start, stop). Non-numeric values are dropped, not raised.

        Home Assistant writes `unavailable` and `unknown` into numeric entities
        during restarts and integration reloads; those are absence of data, not
        an error worth stopping a six-month rebuild for.
        """
