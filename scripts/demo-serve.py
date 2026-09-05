#!/usr/bin/env python3
"""Serve the panel against a demo `/data`, with no Home Assistant and no broker.

`server.lifespan` calls `runtime.bootstrap()`, which builds a `HomeAssistant`
client and immediately re-reads timezone, country and the home's LATITUDE AND
LONGITUDE from it. Pointing that at a real installation would write real
coordinates into the demo's config and put them on screen -- so this replaces
the client with one that answers from a fictional household instead.

Everything else is the real add-on: the same FastAPI app, the same worker, the
same panel bundle. The MQTT broker is never configured, which the add-on already
treats as "no entities published yet" rather than an error, and the event
listener fails to connect and says so on the status page -- both are the
documented degraded paths, not special cases added for this.

    scripts/demo-serve.py --data ~/occupancy-demo/data --port 8099
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "occupancy-forecast-edge"))

from occupancy_forecast import config, runtime  # noqa: E402
from occupancy_forecast.sources import HistoryStore, StoreSource  # noqa: E402

# What Home Assistant would report. The extra unticked person and zone are here
# so the Config view shows a CHOICE rather than a list of exactly what is already
# enabled, which is what that page actually looks like on a real install.
ENTITIES = [
    ("person.alice", "home", "Alice"),
    ("person.bob", "not_home", "Bob"),
    ("person.carol", "not_home", "Carol"),
    ("group.household", "home", "Household"),
    ("zone.office", "1", "Office"),
    ("zone.workshop", "0", "Workshop"),
    ("zone.gym", "0", "Gym"),
    ("sensor.home_alice_distance", "0", "Alice distance from home"),
    ("sensor.home_alice_direction_of_travel", "stationary", "Alice direction of travel"),
    ("sensor.home_bob_distance", "8100", "Bob distance from home"),
    ("sensor.home_bob_direction_of_travel", "away_from", "Bob direction of travel"),
]


class DemoHomeAssistant:
    """Answers the three questions the add-on asks, from a household that isn't real."""

    def config(self) -> dict:
        return {"time_zone": "Europe/Amsterdam", "country": "NL",
                "latitude": 52.09, "longitude": 5.12,
                "unit_system": {"length": "km"}}

    def states(self) -> list[dict]:
        return [{"entity_id": entity, "state": state,
                 "attributes": {"friendly_name": name}}
                for entity, state, name in ENTITIES]

    def history(self, entity_ids, start, stop=None) -> list[list[dict]]:
        # The archive is already complete and static: there is nothing to top up.
        return []

    def notify(self, *args, **kwargs) -> None:
        pass

    def dismiss(self, *args, **kwargs) -> None:
        pass


def install(data: Path) -> None:
    """Repoint every `/data` path at the demo directory, then stub the client."""
    config.DATA_DIR = data
    config.CONFIG_PATH = data / "config.json"
    config.MODELS_DIR = data / "models"
    config.FEATURES_PATH = data / "features.parquet"
    config.HISTORY_DB = data / "history.db"

    settings = config.Settings.load(config.CONFIG_PATH)
    if settings is None:
        raise SystemExit(f"no config.json in {data} -- run demo-instance.py build")

    def bootstrap(path: Path = config.CONFIG_PATH):
        config.configure(settings)
        ha = DemoHomeAssistant()
        return settings, ha, StoreSource(HistoryStore(config.HISTORY_DB), ha)

    runtime.bootstrap = bootstrap
    runtime.home_assistant = DemoHomeAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    install(args.data.resolve())

    # Imported AFTER install(): `server` reads config.MODELS_DIR at import time
    # in places, and its lifespan calls the bootstrap patched above.
    import uvicorn

    from occupancy_forecast import server

    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
