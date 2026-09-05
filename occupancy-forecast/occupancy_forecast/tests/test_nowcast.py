"""Serving-time nowcast tests.

Two things are being guarded. First, that the nowcast actually buys the
responsiveness it exists for -- the comparison against the un-nowcast grid row
is the whole point, and a test that only checked the new value would pass just
as happily if the change did nothing. Second, and more important, that it
touches ONLY the origin block: everything else in the row is anchored on the
30-minute slot and drawn from a distribution the models were fitted on, so a
column quietly moved here is a train/serve skew nobody would ever see.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from occupancy_forecast import config, features, nowcast  # noqa: E402

AT = pd.Timestamp("2026-05-01T19:47:00Z")


class FakeSource:
    """A `Source` over a per-entity list of (iso, state) events."""

    def __init__(self, traces: dict[str, list[tuple[str, str]]]):
        self.traces = traces

    def seeded_states(self, entity_id, start, stop=None, seed_days=14):
        return self.traces.get(entity_id, [])

    states = seeded_states

    def numeric(self, entity_id, start, stop=None):
        return []


def _trace(departure: str | None = None) -> list[tuple[str, str]]:
    events = [("2026-05-01T12:00:00Z", "home")]
    if departure:
        events.append((f"2026-05-01T{departure}:00Z", "not_home"))
    return events


def _source(**per_person: list[tuple[str, str]]) -> FakeSource:
    traces = {f"person.{name}": trace for name, trace in per_person.items()}
    traces["group.household"] = per_person.get("alice", _trace())
    return FakeSource(traces)


def _rows() -> pd.DataFrame:
    """One serving row per subject, as `current_rows` hands them over.

    `state_now` is 0.567 -- what the 19:30 slot reports for a 19:47 departure,
    which is exactly the case this module exists to fix: over the half-way
    line, so the house reads occupied for thirteen more minutes.
    """
    return pd.DataFrame({
        "subject": ["alice", "bob", "house"],
        "time": [pd.Timestamp("2026-05-01T19:30:00Z")] * 3,
        "state_now": [0.567, 1.0, 0.567],
        "minutes_in_state": [447.0, 447.0, 447.0],
        "coverage": [1.0, 1.0, 1.0],
        "other_alice": [np.nan, 0.567, np.nan],
        "other_bob": [1.0, np.nan, np.nan],
        "other_house": [np.nan, np.nan, np.nan],
        "tgt1h_lag1d": [0.9, 0.8, 0.95],
        "tgt1h_wclim4": [0.85, 0.75, 0.9],
        "tgt1h_slot_sin": [0.13, 0.13, 0.13],
        "is_holiday": [0.0, 0.0, 0.0],
    })


# ---------------------------------------------------------------------------
# presence_fraction
# ---------------------------------------------------------------------------

def test_a_departure_reads_away_within_the_window():
    """Two and a half minutes, always -- not nought to fifteen, phase-dependent."""
    assert nowcast.presence_fraction(_trace("19:44"), AT) < 0.5


def test_a_departure_one_minute_ago_has_not_crossed_yet():
    """Deliberate. The window is a debounce, and a debounce has to cost something.

    One minute of five is 0.8, still occupied. The trade is a fixed ~2.5-minute
    worst case in exchange for never swinging on GPS jitter.
    """
    assert nowcast.presence_fraction(_trace("19:46"), AT) == pytest.approx(0.8)


def test_a_ninety_second_blip_does_not_flip_the_nowcast():
    """The jitter this window exists to absorb.

    `14:22:11 zone.office -> 14:22:49 not_home -> 14:26:16 zone.office` is the
    real trace quoted in features.py. A raw instantaneous state would have
    swung all 48 horizons on it, because `state_now` is `train.RESIDUAL_BASE`.
    """
    events = [("2026-05-01T12:00:00Z", "home"),
              ("2026-05-01T19:45:00Z", "not_home"),
              ("2026-05-01T19:46:30Z", "home")]
    assert nowcast.presence_fraction(events, AT) > 0.5


def test_no_events_is_none_not_a_guess():
    assert nowcast.presence_fraction([], AT) is None


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_the_nowcast_beats_the_grid_row_it_replaces():
    """The comparison that makes this change worth making.

    Without the un-nowcast assertion this test would pass even if `apply` did
    nothing at all.
    """
    rows = _rows()
    assert rows.loc[rows["subject"] == "alice", "state_now"].iloc[0] > 0.5

    out = nowcast.apply(rows, _source(alice=_trace("19:42"), bob=_trace()), AT)
    assert out.loc[out["subject"] == "alice", "state_now"].iloc[0] < 0.5
    assert out.loc[out["subject"] == "bob", "state_now"].iloc[0] == 1.0


def test_the_nowcast_leaves_the_lag_and_calendar_columns_alone():
    """The train/serve-skew guard.

    Everything outside the origin block is anchored on the slot and drawn from
    the distribution the models were fitted on. If one of these ever moves, the
    served row stops being the kind of row the model saw and nothing else in
    the system would say so.
    """
    rows = _rows()
    out = nowcast.apply(rows, _source(alice=_trace("19:42"), bob=_trace()), AT)
    for column in ("time", "coverage", "tgt1h_lag1d", "tgt1h_wclim4",
                   "tgt1h_slot_sin", "is_holiday"):
        pd.testing.assert_series_equal(out[column], rows[column])


def test_the_cross_subject_columns_follow_the_origin():
    """Otherwise `state_now` says "just left" and `other_alice` disagrees."""
    out = nowcast.apply(_rows(), _source(alice=_trace("19:42"), bob=_trace()), AT)
    fresh = out.loc[out["subject"] == "alice", "state_now"].iloc[0]
    assert out.loc[out["subject"] == "bob", "other_alice"].iloc[0] == fresh
    # A subject never mirrors itself -- that would just be `state_now` twice.
    assert pd.isna(out.loc[out["subject"] == "alice", "other_alice"].iloc[0])


def test_a_subject_with_no_recent_events_keeps_its_grid_value():
    """A missing nowcast degrades to today's behaviour, never to NaN.

    `state_now` is `train.RESIDUAL_BASE`, so a NaN there deletes all 48
    horizons for that subject -- the forecast would vanish rather than get
    slightly older.
    """
    rows = _rows()
    source = FakeSource({"person.alice": _trace("19:42")})   # bob and house silent
    out = nowcast.apply(rows, source, AT)
    assert out.loc[out["subject"] == "alice", "state_now"].iloc[0] < 0.5
    assert out.loc[out["subject"] == "bob", "state_now"].iloc[0] == 1.0
    assert not out["state_now"].isna().any()


def test_a_source_that_raises_costs_only_that_subject():
    """One bad entity must not cost the cycle its whole forecast."""
    class Broken(FakeSource):
        def seeded_states(self, entity_id, start, stop=None, seed_days=14):
            if entity_id == "person.bob":
                raise RuntimeError("influx said no")
            return self.traces.get(entity_id, [])

    out = nowcast.apply(_rows(), Broken({"person.alice": _trace("19:42")}), AT)
    assert out.loc[out["subject"] == "alice", "state_now"].iloc[0] < 0.5
    assert out.loc[out["subject"] == "bob", "state_now"].iloc[0] == 1.0


def test_nothing_at_all_returns_the_rows_untouched():
    rows = _rows()
    pd.testing.assert_frame_equal(nowcast.apply(rows, FakeSource({}), AT), rows)


def test_current_at_is_stamped():
    """The third clock. See the `current_at` comment in predict.predict_rows."""
    out = nowcast.apply(_rows(), _source(alice=_trace("19:42"), bob=_trace()), AT)
    assert (out["current_at"] == AT.isoformat()).all()


def test_an_empty_frame_is_survivable():
    assert nowcast.apply(pd.DataFrame(), FakeSource({}), AT).empty


# ---------------------------------------------------------------------------
# The window itself
# ---------------------------------------------------------------------------

def test_the_window_rejects_anything_shorter_than_half_of_it():
    """The property the length was chosen for, asserted rather than assumed."""
    half = nowcast.WINDOW_MINUTES / 2
    blip_end = AT - pd.Timedelta(minutes=0)
    blip_start = blip_end - pd.Timedelta(minutes=half * 0.9)
    events = [("2026-05-01T12:00:00Z", "home"),
              (blip_start.strftime("%Y-%m-%dT%H:%M:%SZ"), "not_home")]
    assert nowcast.presence_fraction(events, AT) > 0.5


def test_the_window_is_shorter_than_the_grid():
    """A window at or above GRID_MINUTES would be no faster than the row it fixes."""
    assert 0 < nowcast.WINDOW_MINUTES < config.GRID_MINUTES


def test_minutes_in_state_is_measured_at_now_not_at_the_slot():
    rows = _rows()
    out = nowcast.apply(rows, _source(alice=_trace("19:42"), bob=_trace()), AT)
    # Alice left at 19:42, so at 19:47 she has been away five minutes -- not
    # the 447 the 19:30 slot reported for an unbroken stretch at home.
    assert out.loc[out["subject"] == "alice", "minutes_in_state"].iloc[0] == pytest.approx(5.0)
