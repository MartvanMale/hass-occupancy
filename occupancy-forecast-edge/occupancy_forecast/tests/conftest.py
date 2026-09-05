"""Give every test a configured installation.

`config.configure()` used to be unnecessary because the identity was a block of
literals. Now it is discovered, so a test has to say what it is testing against
-- which is itself the point: the fixture below is a synthetic household that
matches nobody's real installation, so anything that quietly depends on one
particular set of entity ids fails here rather than on a user's Home Assistant.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from occupancy_forecast import config  # noqa: E402


def settings(**overrides) -> config.Settings:
    base = dict(
        people=["person.alice", "person.bob"],
        zones=["zone.alice_office"],
        zone_names={"zone.alice_office": "Alice Office"},
        house_entity="group.household",
        proximity={"person.alice": ["sensor.home_alice_distance",
                                    "sensor.home_alice_direction_of_travel"]},
        timezone="Europe/Amsterdam",
        country="NL",
        units={"sensor.home_alice_distance": "m"},
        home_latitude=52.0, home_longitude=4.5,
    )
    base.update(overrides)
    return config.Settings(**base)


@pytest.fixture(autouse=True)
def configured():
    """Applied to every test, and reapplied after any test that reconfigures."""
    config.configure(settings())
    yield
    config.configure(settings())
