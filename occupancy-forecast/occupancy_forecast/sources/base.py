"""The read interface every source implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Source(Protocol):
    """Three reads, keyed on entity id.

    Timestamps are ISO-8601 strings, what `features.py` parses, so the feature
    builder does not care which source it is talking to.
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

        HA writes `unavailable`/`unknown` into numeric entities during restarts;
        that is absence of data, not an error worth stopping a rebuild for.
        """
