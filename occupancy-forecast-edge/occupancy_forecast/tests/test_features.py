"""Feature-table tests.

Every test here guards a way this pipeline can be silently wrong rather than
loudly broken -- which is the only failure mode that matters, given that the
implementation this replaces ran for five months writing `went_to_office=False`
on every row and nothing ever noticed. The zone tests at the bottom are the
direct descendants of that bug: they exist because the per-person zone columns
are keyed on a friendly name, and a rename has to be loud.

Runnable two ways:
    pytest app/tests/test_features.py
    python3 app/tests/test_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from occupancy_forecast import config, features, train  # noqa: E402


def _slots(n: int, start="2026-05-01T00:00:00Z") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=f"{config.GRID_MINUTES}min",
                         tz="UTC", name="time")


def _wide_table(days: int = 40) -> pd.DataFrame:
    """A real `features.build()` output, for the tests that melt it.

    Built rather than hand-written: the melt is a mapping between two sets of
    column names, and a hand-written frame would only ever prove the mapping
    agrees with itself.
    """
    start = pd.Timestamp("2026-04-01T00:00:00Z")
    events = {}
    for index, subject in enumerate(config.SUBJECTS):
        entity = subject.entity_id or "group.household"
        rows = []
        for day in range(days + 2):
            base = start + pd.Timedelta(days=day) - pd.Timedelta(days=1)
            rows.append(((base + pd.Timedelta(hours=8 + index)).isoformat(), "not_home"))
            rows.append(((base + pd.Timedelta(hours=18 - index)).isoformat(), "home"))
        events[entity] = rows

    class _Source:
        def states(self, entity, s, t=None):
            return events.get(entity, [])

        def seeded_states(self, entity, s, t=None, seed_days=14):
            return events.get(entity, [])

        def numeric(self, entity, s, t=None):
            return []

    stop = (start + pd.Timedelta(days=days)).isoformat()
    return features.build(_Source(), start.isoformat(), stop)


# ---------------------------------------------------------------------------
# slot_fraction: the debounce
# ---------------------------------------------------------------------------

def test_full_slot_at_home_is_one():
    slots = _slots(2)
    events = [("2026-04-30T23:00:00Z", "home")]
    out = features.slot_fraction(events, slots, "home")
    assert out["frac"].iloc[0] == 1.0
    assert out["coverage"].iloc[0] == 1.0


def test_half_a_slot_away_is_a_half():
    """The whole reason the target is a fraction and not a binary."""
    slots = _slots(1)
    events = [("2026-04-30T23:00:00Z", "home"),
              ("2026-05-01T00:15:00Z", "not_home")]
    out = features.slot_fraction(events, slots, "home")
    assert abs(out["frac"].iloc[0] - 0.5) < 1e-9


def test_a_ninety_second_blip_barely_moves_the_slot():
    """GPS jitter must not flip a slot.

    Measured on a real installation: 19% of one person's episodes are under five
    minutes, and 11 of 28 workplace-zone episodes are under two. Under a
    last-observation-carried-forward resample a blip landing on a grid point
    flips the whole slot; here it costs its own duration and nothing more.
    """
    slots = _slots(1)
    events = [("2026-04-30T23:00:00Z", "home"),
              ("2026-05-01T00:10:00Z", "not_home"),
              ("2026-05-01T00:11:30Z", "home")]
    out = features.slot_fraction(events, slots, "home")
    assert out["frac"].iloc[0] > 0.94, out["frac"].iloc[0]


def test_an_unobserved_slot_is_nan_not_zero():
    """"No data" and "not at home" must never look the same."""
    slots = _slots(4)
    events = [("2026-05-01T01:30:00Z", "home")]  # starts after slots 0-2
    out = features.slot_fraction(events, slots, "home")
    assert np.isnan(out["frac"].iloc[0])
    assert out["coverage"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# observability: the bug that fabricated three weeks of labels
# ---------------------------------------------------------------------------

def test_a_long_silence_is_not_observed():
    """The 653-hour hole of 2026-06-26..07-23, in miniature.

    Without this the step function carries the last known state across the
    entire outage, and the table reports three straight weeks of
    `home_frac == 1.00` for every subject -- a fold whose base rate is literally
    1.000. The first build of this table did exactly that.
    """
    slots = _slots(96)  # two days
    events = pd.DatetimeIndex(["2026-05-01T00:00:00Z", "2026-05-02T23:30:00Z"])
    observable = features.observability(events, slots)
    assert not observable.any(), "a 47-hour silence must not count as observation"


def test_the_nightly_doze_is_still_observed():
    """The counterweight: phones go quiet 21:00-03:00 every single night.

    Blanking those would delete 27% of the history and every night in it, which
    is when occupancy is most predictable. The measured worst case is 10.3 h, so
    a 12 h threshold has to accept 6 h comfortably.
    """
    slots = _slots(24)  # 12 hours
    events = pd.DatetimeIndex(["2026-05-01T00:00:00Z", "2026-05-01T06:00:00Z",
                               "2026-05-01T12:00:00Z"])
    assert features.observability(events, slots).all()


def test_the_threshold_sits_between_the_two():
    assert 10.3 < config.MAX_SILENCE_H < 653, (
        f"MAX_SILENCE_H={config.MAX_SILENCE_H} no longer separates the nightly "
        "doze (max 10.3 h observed) from the real outage (653 h)")


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_daily_lags_never_reach_past_the_origin():
    """`tgt{h}h_lag{k}d` is home_frac at t + h - 24k. It must not be in the future.

    At 36 h the target's "yesterday" is twelve hours ahead of the origin.
    Including it would train a model that cannot be served -- and it would score
    beautifully while doing so, which is what makes this worth a test.
    """
    for horizon in config.HORIZONS_H:
        for days in features.safe_daily_lags(horizon):
            assert 24 * days >= horizon, (
                f"{horizon}h admits lag{days}d, which is "
                f"{horizon - 24 * days}h into the future")


def test_the_gate_is_actually_binding_somewhere():
    """A gate that never excludes anything is not a gate."""
    excluded = [h for h in config.HORIZONS_H
                if set(features.safe_daily_lags(h)) != set(features.DAILY_LAGS)]
    assert excluded, "no horizon excludes any daily lag -- is the gate wired up?"


def test_the_long_frame_has_no_value_in_a_lag_it_may_not_see():
    """**The most important test in this file.**

    The gate used to be enforced by a feature LIST: `features_for(36)` simply
    never named `tgt36h_lag1d`, so the leaky column sat in the parquet and no
    model could reach it. One pooled model has one feature list, so that
    protection is gone and `lag1d` is a real column on every row -- it must
    carry NO VALUE above +24 h.

    Asserting on the feature list here would now pass vacuously while the model
    trained on twelve hours of the future. So this asserts on the data.
    """
    wide = _wide_table()
    long = features.long_frame(wide)
    for days in features.DAILY_LAGS:
        column = f"lag{days}d"
        illegal = long[long[features.HORIZON_COLUMN] > 24 * days]
        assert illegal[column].isna().all(), (
            f"{column} has a value at horizons past {24 * days}h -- "
            f"that is the future, and the model would score beautifully on it")
        # And the converse, or the gate could be satisfied by an empty column.
        legal = long[long[features.HORIZON_COLUMN] <= 24 * days]
        assert legal[column].notna().any(), f"{column} is never populated at all"


def test_the_long_frame_carries_the_same_numbers_as_the_wide_one():
    """The melt is a rename, not a recomputation."""
    wide = _wide_table()
    long = features.long_frame(wide)
    keyed = long.set_index(["subject", "time", features.HORIZON_COLUMN])
    for horizon in (1, 24, 25, 48):
        row = wide.iloc[3]
        got = keyed.loc[(row["subject"], row["time"], float(horizon))]
        assert got[features.TARGET_COLUMN] == pytest.approx(
            row[f"y_{horizon}h"], nan_ok=True)
        assert got["tgt_is_weekend"] == pytest.approx(
            row[f"tgt{horizon}h_is_weekend"], nan_ok=True)
        assert got[f"wclim{features.CLIMATOLOGY_WEEKS}"] == pytest.approx(
            row[features.climatology_column(horizon)], nan_ok=True)


def test_the_long_frame_says_which_lag_the_cross_subject_column_holds():
    """`other_{slug}_lag` steps from a 1 d to a 2 d offset at the h=24/25 line.

    One name, two meanings, so the offset travels as a column of its own rather
    than leaving the model to infer it from `horizon_h`.
    """
    long = features.long_frame(_wide_table())
    for horizon, expected in ((24, 1.0), (25, 2.0)):
        part = long[long[features.HORIZON_COLUMN] == float(horizon)]
        assert (part[features.OTHER_LAG_DAYS_COLUMN] == expected).all()


def test_no_feature_is_a_target():
    """Nothing named y_* may appear in the feature list."""
    assert not [c for c in train.base_features() if c.startswith("y_")]
    assert features.TARGET_COLUMN not in train.base_features()


def test_every_column_a_horizon_reads_has_a_family():
    """`column_family` is what lets the panel summarise a thousand-column table it
    cannot show as a table. An unclassified column falls into "unknown" and is
    silently mis-summarised, which at this width nobody would spot -- so the
    classifier is asserted against the names the code actually mints."""
    minted = {"time", "subject", "home_frac", *features.BUILT_NOT_SHIPPED}
    minted.update(train.origin_features())
    for horizon in config.HORIZONS_H:
        minted.add(f"y_{horizon}h")
        minted.update(features.target_calendar_columns(horizon))
        minted.update(features.cross_subject_lag_columns(horizon))
        minted.add(features.climatology_column(horizon))
        minted.add(features.slot_climatology_column(horizon))
        minted.update(f"tgt{horizon}h_lag{days}d" for days in features.DAILY_LAGS)

    for column in minted:
        family = features.column_family(column)
        assert family != "unknown", f"{column} has no family"
        assert family in features.FAMILIES
        assert family in features.FAMILY_WORDS


def test_phone_sensors_are_built_but_not_shipped():
    """Companion-app sensors are usually enabled long after the recorder started.

    A column that is NaN for all but the last few days of the history trains as
    "unknown" on every row and is worse than not having the feature at all.
    Shipping one should be a deliberate tuple edit, not an accident.
    """
    assert not set(train.base_features()) & set(features.BUILT_NOT_SHIPPED)


# ---------------------------------------------------------------------------
# Which way a slot faces
# ---------------------------------------------------------------------------

def test_a_slot_faces_forward_from_its_left_edge():
    """Slot `t` covers `[t, t+30min)`, not the half hour before it.

    Worth its own test because the whole table reads as half an hour stale if
    you assume otherwise -- `observed_at` on a published forecast is a slot's
    LEFT EDGE, and the newest row is the in-progress slot with the present
    state carried through the part that has not happened yet. Somebody
    (reasonably) read that as staleness and phase-shifted the grid to "fix" it,
    which made the newest row a 30-minute trailing average and strictly less
    responsive. See nowcast.py.
    """
    slots = _slots(1, start="2026-05-01T19:30:00Z")
    events = [("2026-05-01T18:00:00Z", "home"),
              ("2026-05-01T19:40:00Z", "not_home")]
    # Home for [19:30, 19:40) of [19:30, 20:00) -- a third. Under a trailing
    # reading this slot would be entirely `home` and the assert would be 1.0.
    assert abs(features.slot_fraction(events, slots, "home")["frac"].iloc[0]
               - 1 / 3) < 1e-9


def test_responsiveness_is_phase_dependent_which_is_why_nowcast_exists():
    """The measured motivation for the nowcast, pinned so it cannot drift.

    A departure before the slot's midpoint reads away at once; one after it
    keeps the house occupied until the slot turns over. 0-15 minutes, and the
    caller does not get to know which.
    """
    slots = _slots(1, start="2026-05-01T19:30:00Z")

    def state_now(departure: str) -> float:
        events = [("2026-05-01T18:00:00Z", "home"),
                  (f"2026-05-01T{departure}:00Z", "not_home")]
        return features.slot_fraction(events, slots, "home")["frac"].iloc[0]

    assert state_now("19:31") < 0.5, "an early departure reads away at once"
    assert state_now("19:46") > 0.5, "a late one still reads home"


def test_a_shorter_window_is_the_same_arithmetic():
    """`minutes` exists for nowcast, and must not change the default behaviour."""
    slots = _slots(1)
    events = [("2026-04-30T23:00:00Z", "home"),
              ("2026-05-01T00:15:00Z", "not_home")]
    assert (features.slot_fraction(events, slots, "home")["frac"].iloc[0]
            == features.slot_fraction(events, slots, "home",
                                      config.GRID_MINUTES)["frac"].iloc[0])
    # Same events, five-minute window: entirely inside the `home` stretch.
    short = features.slot_fraction(events, slots, "home", 5)
    assert short["frac"].iloc[0] == 1.0 and short["coverage"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Cyclical encodings
# ---------------------------------------------------------------------------

def test_midnight_wraps():
    """23:30 and 00:00 must be neighbours, which a raw hour column cannot say."""
    times = pd.Series(pd.to_datetime(
        ["2026-05-01T21:30:00Z", "2026-05-01T22:00:00Z", "2026-05-01T09:00:00Z"],
        utc=True)).dt.tz_convert(config.TIMEZONE)
    out = features._cyclical(times)
    near = np.hypot(out["slot_sin"][0] - out["slot_sin"][1],
                    out["slot_cos"][0] - out["slot_cos"][1])
    far = np.hypot(out["slot_sin"][0] - out["slot_sin"][2],
                   out["slot_cos"][0] - out["slot_cos"][2])
    assert near < far


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print("\nall passed" if not failures else f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


# ---------------------------------------------------------------------------
# Zones
#
# A zone used to be a work zone belonging to one person, read from the zone
# entity's numeric person-count. It is now a plain place the user ticked, read
# per person from that person's own state -- which is the only per-person zone
# signal history contains, and the one thing the old count could never give:
# one housemate standing in another's work zone is a fact a household count
# cannot hold.
#
# The join is on the zone's FRIENDLY NAME, which is the thing that broke the
# previous implementation. These tests exist to make that break loud.
# ---------------------------------------------------------------------------

class _Events:
    """A Source that answers with a canned event list, for zone tests."""

    def __init__(self, by_entity: dict[str, list[tuple[str, str]]]):
        self.by_entity = by_entity

    def states(self, entity, start, stop=None):
        return self.by_entity.get(entity, [])

    def seeded_states(self, entity, start, stop=None, seed_days=14):
        return self.by_entity.get(entity, [])

    def numeric(self, entity, start, stop=None):
        return []


def _zoned(**overrides):
    """Reconfigure with two ticked zones. Undone by the autouse fixture."""
    from .conftest import settings
    base = dict(zones=["zone.alice_office", "zone.market"],
                zone_names={"zone.alice_office": "Alice Office",
                            "zone.market": "Supermarket"})
    base.update(overrides)
    return config.configure(settings(**base))


def test_a_ticked_zone_becomes_a_column_named_after_the_zone():
    """Not after the person. That rename is the whole point of the change."""
    _zoned()
    assert features.zone_columns() == ("zone_alice_office", "zone_market",
                                       "zone_other")


def test_the_zone_columns_and_home_partition_a_covered_slot():
    """home + every zone + other == 1, because they are one state machine.

    The columns are built by the same `slot_fraction` integration as
    `home_frac`, so this is true by construction -- and asserting it is what
    catches a future column that double-counts or drops time on the floor.
    """
    _zoned()
    slots = _slots(1)
    events = [("2026-04-30T23:00:00Z", "home"),
              ("2026-05-01T00:10:00Z", "Supermarket"),
              ("2026-05-01T00:20:00Z", "not_home")]
    frame = pd.DataFrame(index=slots)
    features._add_zones(frame, events, slots)
    home = features.slot_fraction(events, slots, config.HOME_STATE)["frac"]

    total = home.iloc[0] + sum(frame[c].iloc[0] for c in features.zone_columns())
    assert total == 1.0
    # Ten of the thirty minutes in the zone, ten away from it and unaccounted.
    assert frame["zone_market"].iloc[0] == 1 / 3
    assert frame["zone_other"].iloc[0] == 1 / 3


def test_half_a_slot_in_a_zone_is_a_half():
    _zoned()
    slots = _slots(1)
    events = [("2026-04-30T23:00:00Z", "Alice Office"),
              ("2026-05-01T00:15:00Z", "not_home")]
    frame = pd.DataFrame(index=slots)
    features._add_zones(frame, events, slots)
    assert frame["zone_alice_office"].iloc[0] == 0.5


def test_an_unticked_zone_is_lumped_in_with_everywhere_else():
    """Unticking means "stop distinguishing this place", not "call it home"."""
    _zoned(zones=["zone.market"], zone_names={"zone.market": "Supermarket"})
    slots = _slots(1)
    events = [("2026-04-30T23:00:00Z", "Alice Office")]
    frame = pd.DataFrame(index=slots)
    features._add_zones(frame, events, slots)
    assert "zone_alice_office" not in frame.columns
    assert frame["zone_other"].iloc[0] == 1.0


def test_a_renamed_zone_is_counted_rather_than_silently_swallowed():
    """The scar this whole diagnostic exists for.

    Rename `zone.alice_office` in Home Assistant and every historical row still
    says the OLD name. Those rows land in `zone_other`, which is defensible --
    but it must not happen quietly, because that is exactly how the previous
    implementation spent five months writing a column that meant nothing.
    """
    _zoned()
    events = [("2026-05-01T00:00:00Z", "Alice Office"),      # current name
              ("2026-05-01T01:00:00Z", "Kantoor Alice"),     # what it was called
              ("2026-05-01T02:00:00Z", "Kantoor Alice"),
              ("2026-05-01T03:00:00Z", "home"),
              ("2026-05-01T04:00:00Z", "not_home")]
    source = _Events({"person.alice": events})

    unmatched = features.unmatched_away_states(source, "2026-05-01T00:00:00Z", None)
    assert unmatched == {"Kantoor Alice": 2}
    # `not_home` is Home Assistant's own word for "away, in no zone", not a
    # name that failed to resolve -- reporting it would cry wolf on every house.
    assert "not_home" not in unmatched
    assert "home" not in unmatched


def test_the_house_sees_a_zone_any_of_the_people_are_in():
    """The house has no state of its own, so it gets the union.

    This is the household reading the old `office_{slug}` columns carried, and
    losing it would quietly make the house model blinder than the person models.
    """
    _zoned()
    # Events inside the window as well as before it: `_liveness` builds the
    # observability mask from state changes, and a slot with no evidence that
    # anything was recording is NaN by design rather than zero.
    source = _Events({
        "person.alice": [("2026-04-30T23:00:00Z", "Supermarket"),
                         ("2026-05-01T00:10:00Z", "Supermarket")],
        "person.bob": [("2026-04-30T23:00:00Z", "home"),
                       ("2026-05-01T00:10:00Z", "home")],
        "group.household": [("2026-04-30T23:00:00Z", "home"),
                            ("2026-05-01T00:10:00Z", "home")],
    })
    table = features.build(source, "2026-05-01T00:00:00Z", "2026-05-01T00:30:00Z")
    house = table[table["subject"] == config.HOUSE_SLUG]
    assert house["zone_market"].iloc[0] == 1.0
    assert table[table["subject"] == "bob"]["zone_market"].iloc[0] == 0.0

    # And the house is a UNION, not a partition: Bob home while Alice is at the
    # supermarket sums past one, which is correct and must not be "fixed".
    # Measured on the real history: 2229 of 6871 covered house rows do this.
    zoned = sum(house[c].iloc[0] for c in features.zone_columns())
    assert house["home_frac"].iloc[0] + zoned > 1.0


def test_every_zone_column_lands_in_the_zone_family():
    _zoned()
    for column in features.zone_columns():
        assert features.column_family(column) == "zone"


def test_an_unserved_candidate_does_not_widen_the_serving_reach():
    """`predict.LOOKBACK_DAYS` derives from `deepest_lookback_days`, and every
    serving cycle is a full lookback rebuild -- every five minutes.

    Building the wide climatology unconditionally while nothing served it took
    that from 32 to 60 days for a feature no model reads. The reach has to track
    what is SERVED. The other direction still has to hold: the moment a
    candidate IS served, the reach widens with it, or production would average
    fewer weeks than training did and nothing would say so.
    """
    before = features.SHIPPED_EXTRAS
    try:
        features.SHIPPED_EXTRAS = ()
        narrow = features.deepest_lookback_days()
        assert narrow == 7 * features.CLIMATOLOGY_WEEKS

        features.SHIPPED_EXTRAS = ("wclim_wide",)
        assert features.deepest_lookback_days() == 7 * features.WIDE_CLIMATOLOGY_WEEKS
        assert features.deepest_lookback_days() > narrow

        # A candidate that needs no extra reach must not buy any.
        features.SHIPPED_EXTRAS = ("int_calendar",)
        assert features.deepest_lookback_days() == narrow
    finally:
        features.SHIPPED_EXTRAS = before


# --- the next alarm -------------------------------------------------------

def test_the_next_alarm_becomes_hours_ahead_on_the_grid():
    """The one feature that knows about tomorrow, so the arithmetic is worth
    pinning: a 07:00 alarm read from a 05:00 slot is two hours ahead."""
    import datetime as dt

    import numpy as np
    import pandas as pd

    from occupancy_forecast import config, features

    slots = pd.date_range("2026-09-01T03:00Z", periods=6, freq="30min",
                          tz="UTC", name="time")
    frame = pd.DataFrame(index=slots)

    class Source:
        def seeded_states(self, entity_id, start, stop=None, seed_days=14):
            return [("2026-08-31T20:00:00+00:00", "2026-09-01T07:00:00+00:00")]

    subject = config.Subject(slug="alice", entity_id="person.alice",
                             is_person=True, next_alarm_entity="sensor.a_next_alarm")
    features._add_next_alarm(frame, Source(), subject, "2026-08-31", None, slots)

    assert frame["next_alarm_h"].iloc[0] == pytest.approx(4.0)   # 03:00 -> 07:00
    assert frame["next_alarm_h"].iloc[4] == pytest.approx(2.0)   # 05:00 -> 07:00
    # 07:00 itself and after: the alarm has fired, so there is no next one.
    assert np.isnan(frame["next_alarm_h"].iloc[-1]) or frame["next_alarm_h"].iloc[-1] > 0


def test_absent_clears_the_alarm_rather_than_being_skipped():
    """`source.numeric` would drop the `absent` row and carry a cancelled alarm
    forward forever. That is the bug this column is parsed by hand to avoid."""
    import numpy as np
    import pandas as pd

    from occupancy_forecast import config, features

    slots = pd.date_range("2026-09-01T03:00Z", periods=4, freq="30min",
                          tz="UTC", name="time")
    frame = pd.DataFrame(index=slots)

    class Source:
        def seeded_states(self, entity_id, start, stop=None, seed_days=14):
            return [("2026-08-31T20:00:00+00:00", "2026-09-01T07:00:00+00:00"),
                    ("2026-09-01T03:45:00+00:00", "absent")]

    subject = config.Subject(slug="alice", entity_id="person.alice",
                             is_person=True, next_alarm_entity="sensor.a_next_alarm")
    features._add_next_alarm(frame, Source(), subject, "2026-08-31", None, slots)

    values = frame["next_alarm_h"].to_numpy()
    assert not np.isnan(values[0]), "an alarm was set"
    assert np.isnan(values[-1]), "it was cancelled, so there is no next alarm"


def test_a_person_with_no_alarm_entity_gets_the_column_anyway():
    """A household with no companion app is not an error -- the column goes NaN
    and the ship gate prices what is left, same as proximity."""
    import numpy as np
    import pandas as pd

    from occupancy_forecast import config, features

    slots = pd.date_range("2026-09-01T03:00Z", periods=3, freq="30min", tz="UTC")
    frame = pd.DataFrame(index=slots)
    subject = config.Subject(slug="bob", entity_id="person.bob", is_person=True)
    features._add_next_alarm(frame, None, subject, "2026-08-31", None, slots)

    assert frame["next_alarm_h"].isna().all()
