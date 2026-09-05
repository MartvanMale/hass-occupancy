"""Where the history comes from.

Everything the feature builder needs is expressed as three reads keyed on an
**entity id**. How those land in storage is an implementation detail, which is
the whole point: the same `features.build()` runs against a local SQLite store
on a stock Home Assistant and against a six-month InfluxDB archive here.

  states(entity_id, ...)         -> [(iso, "home"), ...]        string states
  seeded_states(entity_id, ...)  -> the same, plus the value carried in from
                                    before the window, re-stamped at `start`
  numeric(entity_id, ...)        -> [(iso, 18.4), ...]          numeric states

`seeded_states` matters more than it looks. Presence transitions can be hours
apart, so a window that opens mid-episode has no idea what the state was until
the next change -- that is hours of the first day silently unlabelled.
"""

from .base import Source
from .store import HistoryStore
from .ha import HomeAssistant, StoreSource
from .influx import InfluxSource

__all__ = ["Source", "HistoryStore", "HomeAssistant", "StoreSource", "InfluxSource"]
