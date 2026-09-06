"""The arrival ETA, and the condition it is conditional on.

`eta.py` answers "how long until you are home, GIVEN you are on your way". The
model enforces that in training by discarding anything more than `MAX_LEAD_MIN`
from an arrival. These tests are about enforcing it at serving time too, which
is where it was missing.

`alice`, because she is the conftest household's person with a proximity entity.
An unconfigured name makes `current_row` return None for a reason that has
nothing to do with the gate -- which is how the first version of these tests
passed without exercising it at all.
"""

import datetime as dt

import pandas as pd

from occupancy_forecast import eta


def _distance_trace(kilometres):
    """A source whose distance trace ends now, one sample every two minutes."""
    now = pd.Timestamp.now(tz="UTC").floor("min")
    stamps = [now - dt.timedelta(minutes=2 * i)
              for i in range(len(kilometres) - 1, -1, -1)]

    class Source:
        def numeric(self, entity_id, start, stop=None):
            return [(t.isoformat(), km * 1000.0)
                    for t, km in zip(stamps, kilometres)]

        def seeded_states(self, entity_id, start, stop=None, seed_days=14):
            return []

    return Source()


def test_no_eta_is_served_to_somebody_sitting_still_far_from_home():
    """The bug this exists to stop, with the numbers that produced it.

    Somebody at their desk: 32.6 km out, closing 0.0 km/h, direction neither
    towards nor away. The model answered **169 minutes** -- the top of its
    trained range -- for a person who had not moved and would not be home for
    six hours. It could not answer otherwise: `MAX_LEAD_MIN` discards
    everything beyond three hours from TRAINING, so the model cannot express a
    longer wait, and nothing at serving time asked whether the question was in
    range.
    """
    assert eta.current_row(_distance_trace([32.61] * 21), "alice") is None, \
        "a stationary person is not on a journey home"


def test_an_eta_is_served_to_somebody_on_their_way():
    """The other half, and the reason the gate is a speed rather than a silence:
    while closing, this sensor has an MAE of four to five minutes."""
    approaching = [30.0 - i for i in range(21)]        # 30 km -> 10 km, ~30 km/h
    row = eta.current_row(_distance_trace(approaching), "alice")
    assert row is not None, "somebody actually driving home must still get an ETA"
    assert float(row["closing_kmh"].iloc[0]) >= eta.MIN_CLOSING_KMH


def test_it_is_the_speed_doing_the_work_and_not_a_lookup_failing():
    """Same person, same entity, same distances -- one moving and one not."""
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6)).isoformat()
    still = eta.feature_frame(
        eta.traces(_distance_trace([12.0] * 21), "alice", since))
    moving = eta.feature_frame(
        eta.traces(_distance_trace([22.0 - i * 0.5 for i in range(21)]),
                   "alice", since))
    assert not still.empty and not moving.empty, "both traces reach the model"
    assert float(still["closing_kmh"].iloc[-1]) < eta.MIN_CLOSING_KMH
    assert float(moving["closing_kmh"].iloc[-1]) >= eta.MIN_CLOSING_KMH


def test_somebody_already_home_still_gets_no_eta():
    """The older of the two refusals, and it must survive the new one."""
    assert eta.current_row(_distance_trace([0.2] * 21), "alice") is None
