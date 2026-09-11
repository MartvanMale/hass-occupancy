"""The tests that exist because this has to run on somebody else's house.

Every one of these guards a way the code could be wired to one specific
installation. They are the difference between "it works here" and "it works".
"""

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from occupancy_forecast import config, discover, evaluate, features, train  # noqa: E402
from occupancy_forecast.sources import HistoryStore  # noqa: E402

from occupancy_forecast.tests.conftest import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Identity is any number of people
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_any_number_of_people(count):
    """One person is a real household. So is five."""
    people = [f"person.p{i}" for i in range(count)]
    subjects = config.configure(settings(people=people, zones=[], proximity={}))
    assert len(subjects) == count + 1                 # people + the house
    assert [s.slug for s in config.PEOPLE] == [f"p{i}" for i in range(count)]


def test_no_people_is_refused_clearly():
    with pytest.raises(ValueError, match="no people configured"):
        config.configure(settings(people=[]))


def test_renaming_a_person_does_not_silently_drop_their_rows():
    """The bug this port exists to kill: a hardcoded exemption list and a
    derived feature list that disagree on a rename drop every row for the
    renamed person."""
    config.configure(settings(people=["person.zoe", "person.quentin"],
                              zones=["zone.lab"],
                              zone_names={"zone.lab": "Lab"}))
    optional = train.may_be_nan()
    for slug in config.all_slugs():
        assert f"other_{slug}" in optional, f"other_{slug} would be required"
    for column in features.zone_columns():
        assert column in optional, f"{column} would be required"

    # And the feature list and the exemptions must agree.
    served = set(train.base_features())
    for column in optional:
        if column in served:
            assert column in train.nan_allowed()


def test_a_person_called_house_does_not_collide():
    """`person.house` must not give two subjects the slug `house`: that surfaces
    much later as a duplicate-label reindex error inside the horizon join."""
    subjects = config.configure(settings(people=["person.house", "person.other"],
                                         zones=[], proximity={}))
    slugs = [s.slug for s in subjects]
    assert len(slugs) == len(set(slugs)), slugs
    assert config.HOUSE_SLUG in slugs


def test_two_people_slugifying_the_same_are_kept_apart():
    subjects = config.configure(settings(people=["person.alice", "person.Alice"],
                                         zones=[], proximity={}))
    assert len({s.slug for s in subjects}) == 3


# ---------------------------------------------------------------------------
# Optional everything
# ---------------------------------------------------------------------------

def _store_with(days: int = 70) -> HistoryStore:
    """A store holding one person alternating home/away, and nothing else."""
    store = HistoryStore(Path(tempfile.mkdtemp()) / "h.db")
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rows = []
    for hour in range(days * 24):
        when = start + dt.timedelta(hours=hour)
        state = "home" if (hour % 24) < 8 or (hour % 24) > 17 else "not_home"
        rows.append(("person.solo", int(when.timestamp() * 1000), state))
    store.append(rows)
    return store


class _StoreOnly:
    """A Source over a bare HistoryStore -- no Home Assistant, no network."""

    def __init__(self, store):
        self.store = store

    def states(self, e, s, t=None):
        return self.store.states(e, s, t)

    def seeded_states(self, e, s, t=None, seed_days=14):
        return self.store.seeded_states(e, s, t, seed_days)

    def numeric(self, e, s, t=None):
        return self.store.numeric(e, s, t)


def test_builds_with_nothing_optional_present():
    """One person, no zone, no house group, no Proximity, no country. The bar for
    "installable by anyone": a full-width table with the optional columns NaN
    rather than missing."""
    config.configure(settings(people=["person.solo"], zones=[],
                              house_entity=None, proximity={}, country=None,
                              units={}))
    source = _StoreOnly(_store_with())
    table = features.build(source, "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z")

    assert not table.empty
    assert set(table["subject"]) == {"solo", "house"}

    # The origin block has to be in the WIDE table; the horizon-relative
    # columns are minted by the melt, which `test_features` covers.
    for column in train.origin_features():
        assert column in table.columns, f"{column} missing entirely"

    # The optional ones are present-but-empty, which is what lets the model
    # ignore them rather than crash on them.
    for column in features.PROXIMITY_COLUMNS:
        assert table[column].isna().all()
    assert table["is_holiday"].eq(0).all(), "no country should mean no holidays, not a crash"

    # And the house, with no group configured, is the OR over the people.
    house = table[table["subject"] == "house"]["home_frac"]
    solo = table[table["subject"] == "solo"]["home_frac"]
    assert house.notna().any()
    assert np.allclose(house.dropna().to_numpy(), solo.dropna().to_numpy(), atol=1e-9)


def test_an_unsupported_country_does_not_crash_the_build():
    """`holidays.country_holidays` raises NotImplementedError for countries it
    does not cover, and that must not abort the whole feature build."""
    config.configure(settings(people=["person.solo"], zones=[],
                              house_entity=None, proximity={}, country="ZZ"))
    source = _StoreOnly(_store_with(days=40))
    table = features.build(source, "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z")
    assert table["is_holiday"].eq(0).all()


# ---------------------------------------------------------------------------
# The holiday calendar is the household's, not the country's
# ---------------------------------------------------------------------------
# HA's country says where the house is, not which holidays the people keep.

def _flagged_dates(table) -> set:
    rows = table[table["is_holiday"] == 1]
    return set(rows["time"].dt.tz_convert("Europe/Amsterdam").dt.date)


def _holiday_table(**overrides):
    config.configure(settings(people=["person.solo"], zones=[],
                              house_entity=None, proximity={}, **overrides))
    source = _StoreOnly(_store_with(days=70))
    return features.build(source, "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z")


def test_the_chosen_calendar_wins_over_home_assistants_country():
    """Living in NL, keeping the Indian calendar. Disjoint dates on purpose, so
    this cannot pass by accident."""
    dutch = _flagged_dates(_holiday_table(country="NL", holiday_country=None))
    assert dt.date(2026, 1, 1) in dutch
    assert dt.date(2026, 1, 26) not in dutch

    indian = _flagged_dates(_holiday_table(country="NL", holiday_country="IN"))
    assert {dt.date(2026, 1, 14), dt.date(2026, 1, 26)} <= indian
    assert dt.date(2026, 1, 1) not in indian


def test_no_calendar_is_a_choice_distinct_from_not_having_chosen():
    """"" means "no holidays"; None means "nobody has picked, use HA's".
    Collapsing them re-seeds an explicit "none" on the next restart."""
    assert _flagged_dates(_holiday_table(country="NL", holiday_country="")) == set()
    assert dt.date(2026, 1, 1) in _flagged_dates(
        _holiday_table(country="NL", holiday_country=None))


def test_an_unsupported_chosen_calendar_does_not_crash_the_build():
    table = _holiday_table(country="NL", holiday_country="ZZ")
    assert table["is_holiday"].eq(0).all()


def test_a_chosen_calendar_survives_a_restart():
    """`refresh_environment` runs on every boot and must not overwrite this, or
    the setting is useless."""
    from occupancy_forecast import runtime

    class _HA:
        def config(self):
            return {"time_zone": "Europe/Amsterdam", "country": "NL",
                    "latitude": 52.0, "longitude": 4.5}

        def states(self):
            return []

    chosen = runtime.refresh_environment(
        settings(holiday_country="IN", proximity={}), _HA())
    assert chosen.holiday_country == "IN"
    assert chosen.country == "NL", "HA's own country should still be refreshed"

    # But an install that has never chosen gets seeded, so nobody who ignores
    # the setting sees any change.
    seeded = runtime.refresh_environment(
        settings(holiday_country=None, proximity={}), _HA())
    assert seeded.holiday_country == "NL"

    # And an explicit "none" is a choice, not an absence.
    none = runtime.refresh_environment(
        settings(holiday_country="", proximity={}), _HA())
    assert none.holiday_country == ""


def test_the_next_alarm_sensor_is_matched_per_person_and_a_miss_is_fine():
    """Matched on the person's slug, like the proximity pair. Optional, always."""
    states = [
        {"entity_id": "sensor.alices_pixel_next_alarm"},
        {"entity_id": "sensor.alices_pixel_battery_level"},
        {"entity_id": "binary_sensor.alices_pixel_next_alarm"},   # wrong domain
    ]
    assert discover.match_next_alarm("person.alice", states) == \
        "sensor.alices_pixel_next_alarm"
    assert discover.match_next_alarm("person.bob", states) is None


def test_a_next_alarm_sensor_is_collected_but_never_served():
    """In the tracked set and out of the feature list, on purpose: a companion-app
    sensor is enabled long after the recorder started, so it must accumulate now
    to be trainable later."""
    from occupancy_forecast import features, runtime, train

    configured = settings(next_alarm={"person.alice": "sensor.alices_pixel_next_alarm"})
    assert "sensor.alices_pixel_next_alarm" in runtime.tracked_entities(configured)
    assert "sensor.alices_pixel_next_alarm" in runtime.absence_entities(configured)

    assert not set(train.base_features()) & set(features.BUILT_NOT_SHIPPED)

    # And it is not woken on: an alarm being set is not news about where
    # anybody is right now.
    assert "sensor.alices_pixel_next_alarm" not in runtime.trigger_entities(configured)


def test_no_alarm_set_is_recorded_rather_than_dropped(tmp_path):
    """`unavailable` on a next-alarm sensor means "no alarm", which is data -- and
    the commoner of the two readings. Distinct from the presence entities, which
    keep the same words for the opposite reason; see `test_presence_gaps`."""
    from occupancy_forecast.sources.ha import ABSENT, StoreSource
    from occupancy_forecast.sources.store import HistoryStore

    alarm, person = "sensor.alices_pixel_next_alarm", "person.alice"

    class _HA:
        def history(self, entity_ids, start, stop):
            return [
                [{"entity_id": alarm, "state": "2026-09-02T06:30:00+00:00",
                  "last_changed": "2026-09-01T21:00:00+00:00"},
                 {"entity_id": alarm, "state": "unavailable",
                  "last_changed": "2026-09-01T22:00:00+00:00"}],
                [{"entity_id": person, "state": "home",
                  "last_changed": "2026-09-01T21:00:00+00:00"},
                 {"entity_id": person, "state": "unavailable",
                  "last_changed": "2026-09-01T22:00:00+00:00"}],
            ]

    store = HistoryStore(tmp_path / "history.db")
    StoreSource(store, _HA()).collect([alarm, person],
                                      absence_is_a_reading=[alarm])

    alarms = [state for _, state in store.states(alarm, "2026-09-01T00:00:00Z")]
    assert alarms == ["2026-09-02T06:30:00+00:00", ABSENT]

    # Named in neither list, so the gap is dropped and nothing ends the `home`.
    people = [state for _, state in store.states(person, "2026-09-01T00:00:00Z")]
    assert people == ["home"]


def test_never_having_looked_for_a_next_alarm_differs_from_finding_none():
    """Same three-state reasoning as the holiday calendar: None is "never looked",
    {} is "looked, found nothing" and must survive a restart."""
    from occupancy_forecast import runtime

    class _HA:
        def config(self):
            return {"time_zone": "Europe/Amsterdam", "country": "NL",
                    "latitude": 52.0, "longitude": 4.5}

        def states(self):
            return [{"entity_id": "sensor.alices_pixel_next_alarm"}]

    looked = runtime.refresh_environment(
        settings(next_alarm=None, proximity={}), _HA())
    assert looked.next_alarm == {"person.alice": "sensor.alices_pixel_next_alarm"}

    cleared = runtime.refresh_environment(
        settings(next_alarm={}, proximity={}), _HA())
    assert cleared.next_alarm == {}, "an empty mapping is a choice, not an absence"


def test_the_picker_offers_real_countries():
    countries = discover.holiday_countries()
    codes = {c["code"] for c in countries}
    assert {"NL", "IN", "US"} <= codes
    assert len(codes) == len(countries), "a duplicate code cannot be told apart"
    assert "UK" not in codes, "the GB alias would show up as a second Britain"
    assert dict((c["code"], c["name"]) for c in countries)["NL"] == "Netherlands"

    assert discover.is_supported_country("NL")
    assert not discover.is_supported_country("ZZ")


def test_a_bad_timezone_says_which_one():
    config.configure(settings(people=["person.solo"], timezone="Mars/Olympus"))
    source = _StoreOnly(_store_with(days=10))
    with pytest.raises(ValueError, match="Mars/Olympus"):
        features.build(source, "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z")


# ---------------------------------------------------------------------------
# Synthesised proximity
# ---------------------------------------------------------------------------

def test_haversine_matches_a_proximity_reading():
    """Checked against HA's own Proximity sensors: within 4-10 m, which is GPS
    accuracy rather than the formula's. Coordinates are the fixture's synthetic
    offshore point."""
    home = (52.0, 4.5)
    away = (52.3, 4.9)
    assert abs(discover.haversine_m(home, home)) < 1
    assert abs(discover.haversine_m(home, away) - 43_100) < 50


def test_direction_is_derived_when_proximity_is_absent():
    """No Proximity integration means no `direction_of_travel` entity, so the
    sign of the distance delta has to stand in for it."""
    config.configure(settings(people=["person.solo"], proximity={}, zones=[],
                              house_entity=None))
    store = _store_with(days=20)
    entity = discover.synthetic_distance_entity("solo")
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    # Walk in from 20 km to 0 over ten hours.
    store.append([(entity, int((start + dt.timedelta(hours=h)).timestamp() * 1000),
                   str(max(0, 20_000 - h * 2_000))) for h in range(240)])

    table = features.build(_StoreOnly(store), "2026-01-01T00:00:00Z",
                           "2026-01-10T00:00:00Z")
    solo = table[table["subject"] == "solo"]
    assert solo["distance_km"].notna().any()
    assert solo["dir_towards"].fillna(0).sum() > 0, "closing distance should read as towards"


# ---------------------------------------------------------------------------
# Model or nothing
# ---------------------------------------------------------------------------

def _newest_rows(days: int = 30):
    """The serving frame a fresh install would have: one row per subject."""
    config.configure(settings(people=["person.solo"], zones=[],
                              house_entity=None, proximity={}))
    source = _StoreOnly(_store_with(days=days))
    table = features.build(source, "2026-01-01T00:00:00Z", "2026-01-25T00:00:00Z")
    return (table.dropna(subset=["state_now"]).sort_values("time")
            .groupby("subject", as_index=False).tail(1))


def test_a_record_is_published_with_no_models_at_all():
    """The first seven weeks of every install: no curve, but still a record.
    `current`, `current_at`, `predicted_at` and the ETA are observations, not
    model output -- do not "fix" the empty curve by putting a baseline in it."""
    from occupancy_forecast import predict as predict_mod

    results = predict_mod.predict_rows({}, _newest_rows())  # no models whatsoever
    assert results, "a fresh install must still publish a record"
    for result in results:
        assert result["curve"] == {}, "no model, so no horizon may be published"
        assert "sources" not in result, "provenance died with the baselines"
        assert result["current"] is not None
        assert result["predicted_at"]
        assert result["next_departure_h"] is None
        assert result["next_arrival_h"] is None


def _dedicated(horizon: int, ships: bool, **metrics) -> dict:
    """One artifact entry as `load_models` would hand it to `predict_rows`."""
    return {"model": object(), "kind": "dedicated" if ships else None,
            "version": "test", "horizon_h": horizon, "features": [],
            "metrics": {"ships": ships, "kind": "dedicated" if ships else None,
                        **metrics}}


def test_a_horizon_whose_baseline_won_is_not_published(monkeypatch):
    """The gate's verdict is acted on, not merely recorded. The artifact still
    carries a usable `fallback` spec for a horizon that lost; this pins that
    serving IGNORES it."""
    from occupancy_forecast import predict as predict_mod, train as train_mod

    rows = _newest_rows()
    monkeypatch.setattr(train_mod, "predict_dedicated",
                        lambda model, frame, horizon: np.array([0.5]))
    fallback = {"which": "persistence", "column": "state_now",
                "weight": 0.9, "base": 0.4}
    models = {
        1: _dedicated(1, True),
        2: _dedicated(2, False, best_baseline="persistence", fallback=fallback),
    }

    for result in predict_mod.predict_rows(models, rows):
        assert 1 in result["curve"], "the shipping horizon still serves"
        assert 2 not in result["curve"], "a horizon the baseline won publishes nothing"


def test_a_shipping_horizon_the_row_cannot_answer_is_a_hole(monkeypatch, caplog):
    """A stale sensor deletes an hour of the forecast, and says so. `_model_curve`
    swallows the failure with a bare `except`, so the warning is the only place
    it can surface."""
    import logging
    from occupancy_forecast import predict as predict_mod, train as train_mod

    rows = _newest_rows()

    def flaky(model, frame, horizon):
        if horizon == 2:
            raise KeyError("tgt2h_lag1d")     # a column the row does not carry
        return np.array([0.5])

    monkeypatch.setattr(train_mod, "predict_dedicated", flaky)
    models = {1: _dedicated(1, True), 2: _dedicated(2, True)}

    with caplog.at_level(logging.WARNING):
        results = predict_mod.predict_rows(models, rows)

    assert all(1 in r["curve"] for r in results)
    assert all(2 not in r["curve"] for r in results), "no number without a model"
    assert any("produced no value" in r.message for r in caplog.records)


class _Recorder:
    """Just enough MQTT client to read back what `publish` sent."""

    def __init__(self):
        self.sent: dict[str, str] = {}

    def publish(self, topic, payload, retain=False, qos=0):
        self.sent[topic] = payload


def _published_state(curve: dict) -> dict:
    from occupancy_forecast import predict as predict_mod

    config.configure(settings(people=["person.solo"], zones=[],
                              house_entity=None, proximity={}))
    client = _Recorder()
    predict_mod.publish([{
        "subject": "solo", "observed_at": "2026-01-01T00:00:00+00:00",
        "current_at": None, "predicted_at": "2026-01-01T00:00:00+00:00",
        "current": 1.0, "curve": curve, "next_departure_h": None,
        "next_arrival_h": None, "eta_minutes": None, "model_version": "test",
    }], client)
    topic = f"{predict_mod.state_prefix()}/solo/state"
    return json.loads(client.sent[topic])


def test_an_unserved_horizon_publishes_an_explicit_null():
    """HA IGNORES an empty MQTT payload, so an absent key leaves the sensor
    holding its last number forever with a fresh `predicted_at` beside it.
    `null` renders as "None", which `PAYLOAD_NONE` maps to unknown."""
    state = _published_state({1: 0.9})
    assert state["p_home_1h"] == 90.0
    assert "p_home_24h" in state, "an absent key is an empty payload, which HA ignores"
    assert state["p_home_24h"] is None, "a hole must clear the sensor, not freeze it"


def test_the_published_state_carries_every_sensor_horizon():
    """Including with nothing trained at all: discovery configs are retained and
    static, so a config can outlive the model that justified it and the entity
    would go stale rather than unknown."""
    for curve in ({}, {1: 0.9, 6: 0.2}):
        state = _published_state(curve)
        assert {f"p_home_{h}h" for h in config.SENSOR_HORIZONS_H} <= set(state)


# ---------------------------------------------------------------------------
# A fresh install trains, instead of waiting seven weeks
# ---------------------------------------------------------------------------

def _times(days: int, n_subjects: int = 2) -> pd.Series:
    """A `time` column as the feature table would have it: one row per subject
    per 30-minute slot."""
    slots = pd.date_range("2026-01-01", periods=days * (1440 // config.GRID_MINUTES),
                          freq=f"{config.GRID_MINUTES}min", tz="UTC")
    return pd.Series(np.repeat(slots.to_numpy(), n_subjects))


def test_a_mature_history_keeps_the_published_geometry():
    """The numbers in baseline.py were measured with 45/7/200. If this changes,
    they stop describing the code that produced them."""
    geometry = evaluate.fold_geometry(_times(90), n_subjects=3)
    assert geometry == {"test_days": evaluate.TEST_DAYS,
                        "min_train_days": evaluate.MIN_TRAIN_DAYS,
                        "min_test_rows": evaluate.MIN_TEST_ROWS}


def test_ten_days_yields_folds_rather_than_none():
    """The whole point of the taper: below 52 days the fixed geometry produces
    no folds at all, and training died with "no folds"."""
    times = _times(10, n_subjects=3)
    embargo = evaluate.embargo_for(1)

    assert evaluate.calendar_folds(times, embargo=embargo) == [], "premise"

    geometry = evaluate.fold_geometry(times, n_subjects=3)
    folds = evaluate.calendar_folds(times, embargo=embargo, **geometry)
    assert len(folds) >= 2, f"{len(folds)} folds from {geometry}"


def test_a_single_person_household_can_clear_the_row_floor():
    """MIN_TEST_ROWS = 200 was silently a three-subject assumption: one person
    yields 48 rows a day."""
    times = _times(10, n_subjects=1)
    geometry = evaluate.fold_geometry(times, n_subjects=1)
    assert geometry["min_test_rows"] < 200
    folds = evaluate.calendar_folds(times, embargo=evaluate.embargo_for(1), **geometry)
    assert len(folds) >= 2, f"{len(folds)} folds from {geometry}"


def test_few_folds_demand_a_bigger_effect():
    """At two folds "beat the baseline more often than not" is 2-of-2, which
    luck reaches. The skill bar rises to compensate."""
    assert train.min_ship_skill_pct(2) > train.min_ship_skill_pct(15)
    assert train.min_ship_skill_pct(15) == train.MIN_SHIP_SKILL_PCT


# ---------------------------------------------------------------------------
# Two builds of this add-on run side by side, so identity derives from the slug
# ---------------------------------------------------------------------------
# Anything that names something -- a topic, a client id, a unique_id, a
# notification, a log line -- or the two builds silently share it.

def _with_prefix(monkeypatch, prefix):
    """Pin `topic_prefix()` without a Supervisor to ask."""
    monkeypatch.setattr(config, "_topic_prefix", prefix)


def test_the_notification_id_separates_two_builds(monkeypatch):
    from occupancy_forecast import server

    _with_prefix(monkeypatch, "occupancy_forecast")
    stable = server.notify_collecting_id()
    _with_prefix(monkeypatch, "occupancy_forecast_edge")
    edge = server.notify_collecting_id()

    assert stable != edge, (
        "stable and edge would share one persistent notification, and each "
        "would dismiss the other's")
    assert "edge" in edge


def test_the_notification_title_says_which_build_wrote_it(monkeypatch):
    _with_prefix(monkeypatch, "occupancy_forecast")
    assert config.display_name() == "Occupancy Forecast"
    _with_prefix(monkeypatch, "occupancy_forecast_edge")
    assert config.display_name() == "Occupancy Forecast Edge"


# How the prefix is LEARNED. A failure must not be cached as an answer, or a
# Supervisor still coming up leaves edge running as a second stable build.

def _unresolved(monkeypatch):
    monkeypatch.setattr(config, "_topic_prefix", None)
    monkeypatch.setattr(config, "_topic_prefix_error", None)


def test_outside_an_add_on_the_default_is_the_answer(monkeypatch):
    _unresolved(monkeypatch)
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    assert config.topic_prefix() == config.DEFAULT_TOPIC_PREFIX
    assert config.topic_prefix_resolved(), "no Supervisor, no second instance"


def test_a_failed_slug_lookup_is_not_remembered(monkeypatch):
    import io
    import urllib.request

    _unresolved(monkeypatch)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "t")

    def refuse(*_args, **_kwargs):
        raise OSError("supervisor not up yet")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    assert config.resolve_topic_prefix() is False
    assert config.topic_prefix() == config.DEFAULT_TOPIC_PREFIX, \
        "names can still be formed for a log line"
    assert not config.topic_prefix_resolved(), "but nothing may be published under it"
    assert "not up yet" in config.topic_prefix_error()

    class Answer:
        def __enter__(self):
            return io.BytesIO(json.dumps(
                {"data": {"slug": "local_occupancy_forecast_edge"}}).encode())

        def __exit__(self, *_exc):
            return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: Answer())

    assert config.resolve_topic_prefix() is True, "the next ask succeeds"
    assert config.topic_prefix() == "occupancy_forecast_edge"
    assert config.topic_prefix_resolved()
    assert config.topic_prefix_error() is None


def test_every_discovery_payload_says_who_created_it(monkeypatch):
    """Home Assistant's MQTT discovery shows `origin` on the device page and in
    its diagnostics, which is where a user goes when wondering where forty-odd
    sensors came from."""
    from occupancy_forecast import predict

    _with_prefix(monkeypatch, "occupancy_forecast_edge")
    payloads = predict._discovery_payloads("alice")
    assert payloads
    for _topic, payload in payloads:
        assert payload["origin"]["name"]
        assert payload["origin"]["sw_version"]
        assert payload["origin"]["support_url"].startswith("https://")


def test_retracting_a_subject_clears_every_retained_topic_it_owned(monkeypatch):
    from occupancy_forecast import predict

    _with_prefix(monkeypatch, "occupancy_forecast_edge")

    class Client:
        def __init__(self):
            self.published = []

        def publish(self, topic, payload, retain=False, qos=0):
            self.published.append((topic, payload, retain))

    client = Client()
    n = predict.retract("alice", client)
    discovery = {topic for topic, _ in predict._discovery_payloads("alice")}
    assert n == len(client.published) == len(discovery) + 2
    assert {t for t, _, _ in client.published} == discovery | {
        "occupancy_forecast_edge/alice/state", "occupancy_forecast_edge/alice/attributes"}
    assert all(payload == "" and retain for _, payload, retain in client.published), \
        "an EMPTY retained payload is how a retained message is deleted"


def test_nothing_connects_to_mqtt_under_a_guessed_prefix(monkeypatch):
    """`Broker.client()` is where a guessed prefix would become a client id, a
    set of discovery topics and a pile of retained states on top of whichever
    build owns the default. It refuses, says why, and the worker asks again."""
    from occupancy_forecast import predict

    _unresolved(monkeypatch)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "t")
    monkeypatch.setattr(config, "_topic_prefix_error", "supervisor not up yet")

    broker = predict.Broker()
    assert broker.client() is None
    assert not broker.connected
    assert "not up yet" in broker.last_error
    assert broker.last_error_public == broker.last_error, \
        "written here, so the Setup tab keeps saying why"


def test_a_broker_failure_is_logged_rather_than_served(monkeypatch):
    """The reason above is this add-on's own sentence. A paho failure is not:
    it can name the broker and its port, and `/api/status` needs no login. The
    full text stays on `last_error` because that is the key the transition-only
    log line compares against."""
    from occupancy_forecast import log, predict

    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)

    def boom():
        raise OSError("core-mosquitto:1883 connection refused")

    monkeypatch.setattr(predict, "connect", boom)

    broker = predict.Broker()
    assert broker.client() is None
    assert "core-mosquitto" in broker.last_error
    assert broker.last_error_public == log.SEE_THE_LOG


# A literal that ESCAPES the add-on must derive from the slug; one that stays
# inside /data need not, since /data is private per add-on.
STORE_LOCAL_NAMES = {
    # Row keys in /data/history.db, never sent to HA. They keep the historical
    # `occupancy_ml.` namespace because the archive cannot be re-supplied.
    '"occupancy_ml.collector"',              # sources/ha.py, the liveness heartbeat
    'f"occupancy_ml.{slug}_distance"',       # discover.py, a synthesised distance
}


def test_no_escaping_identity_string_is_hardcoded():
    """Grep for the default prefix outside the places allowed to spell it. A
    literal that reaches MQTT, a unique_id or a notification does not move when
    the slug does."""
    package = Path(__file__).resolve().parents[1]
    prefix = config.DEFAULT_TOPIC_PREFIX     # derived, so a rename cannot outrun it
    offenders = []
    for path in package.rglob("*.py"):
        if path.name == "config.py" or "tests" in path.parts:
            continue                          # DEFAULT_TOPIC_PREFIX lives in config.py
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if f'"{prefix}' not in code and f"'{prefix}" not in code:
                continue
            if any(name in code for name in STORE_LOCAL_NAMES):
                continue
            offenders.append(f"{path.relative_to(package)}:{n}: {line.strip()}")
    assert not offenders, (
        "identity strings that leave the add-on must derive from "
        "config.topic_prefix() or config.display_name(). If this one stays "
        "inside /data, add it to STORE_LOCAL_NAMES with a note saying why:\n  "
        + "\n  ".join(offenders))


def test_a_config_written_before_the_crossing_cuts_still_loads():
    """No `_migrate` entry is needed: `from_json` filters to the dataclass fields,
    so a missing key takes the default. `_migrate` exists to RENAME a key."""
    text = json.dumps({
        "people": ["person.alice"],
        "zones": [],
        "timezone": "Europe/Amsterdam",
    })
    loaded = config.Settings.from_json(text)
    assert loaded.departure_threshold == config.DEFAULT_DEPARTURE_THRESHOLD
    assert loaded.arrival_threshold == config.DEFAULT_ARRIVAL_THRESHOLD
    assert loaded.crossing_min_hours == config.DEFAULT_CROSSING_MIN_HOURS


def test_an_office_zones_config_migrates_to_a_ticked_list():
    """The upgrade path off the per-person work zone. `office_zones`' values are
    exactly the zones that household cared about, so they become the ticked list;
    the old key is dropped, because two spellings of one setting drift apart."""
    text = json.dumps({
        "people": ["person.alice", "person.bob"],
        "office_zones": {"person.alice": "zone.alice_office",
                         "person.bob": "zone.bob_office"},
        "timezone": "Europe/Amsterdam",
    })
    loaded = config.Settings.from_json(text)
    assert loaded.zones == ["zone.alice_office", "zone.bob_office"]
    assert not hasattr(loaded, "office_zones")
    # Round-trips without resurrecting the old key.
    assert "office_zones" not in json.loads(loaded.to_json())


def test_an_explicit_zone_list_wins_over_the_legacy_key():
    """A config already migrated is not re-migrated over the top of itself."""
    text = json.dumps({"people": ["person.alice"],
                       "office_zones": {"person.alice": "zone.old"},
                       "zones": ["zone.new"]})
    assert config.Settings.from_json(text).zones == ["zone.new"]


def test_history_always_asks_for_an_end_time():
    """Without `end_time`, Home Assistant returns ONE DAY from `start` and says
    nothing about having done so. The same call bootstraps the archive on a fresh
    install, where a silent truncation is not recoverable later."""
    from occupancy_forecast.sources import ha as ha_mod

    asked = []

    class Recording(ha_mod.HomeAssistant):
        def __init__(self):
            pass

        def _get(self, path):
            asked.append(path)
            return []

    Recording().history(["schedule.day_time"], "2026-08-26T00:00:00Z")
    assert asked and "end_time=" in asked[0], (
        "a history call with no end_time silently returns one day")

    asked.clear()
    Recording().history(["x.y"], "2026-08-26T00:00:00Z", "2026-08-27T00:00:00Z")
    assert "end_time=2026-08-27" in asked[0].replace("%3A", ":"), \
        "an explicit stop must still win"
