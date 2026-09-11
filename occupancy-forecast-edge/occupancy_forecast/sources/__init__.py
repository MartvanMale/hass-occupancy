"""Where the history comes from: three reads keyed on an entity id, so the same
`features.build()` runs against a local SQLite store or an InfluxDB archive.

`seeded_states` matters more than it looks: presence changes can be hours apart,
so unseeded, a window opening mid-episode starts hours silently unlabelled.
"""

from .base import Source
from .store import HistoryStore
from .ha import HomeAssistant, StoreSource
from .influx import InfluxSource

__all__ = ["Source", "HistoryStore", "HomeAssistant", "StoreSource", "InfluxSource"]
