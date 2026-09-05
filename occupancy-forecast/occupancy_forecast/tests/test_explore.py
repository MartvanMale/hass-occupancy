"""What the Data tab reads, and the two promises it has to keep.

The promises are that nothing here recomputes a feature its own way, and that
nothing here reads a 134 MiB table to answer a question about its columns. Both
are easy to break by accident and neither is visible when it breaks -- a second
definition of `home_frac` shows a plausible wrong number, and a full read just
makes the tab slow. So both are asserted rather than described.

Everything runs against the synthetic household in conftest.
"""

import datetime as dt

import pandas as pd
import pytest

from occupancy_forecast import config, explore, features
from occupancy_forecast.sources.ha import HEARTBEAT_ENTITY
from occupancy_forecast.sources.store import HistoryStore
from occupancy_forecast.tests.conftest import settings as make_settings


def _ms(when: dt.datetime) -> int:
    return int(when.timestamp() * 1000)


class FakeSource:
    """A source that owns a store.

    `explore` mostly reaches through `.store`, but `verification` goes via
    `features.presence_events`, which asks the SOURCE -- so this delegates the
    two reads that protocol requires, exactly as `sources.ha.StoreSource` does.
    A fake thinner than the protocol would have let a broken call site pass.
    """

    def __init__(self, store):
        self.store = store

    def seeded_states(self, entity_id, start, stop=None, seed_days=14):
        return self.store.seeded_states(entity_id, start, stop, seed_days)

    def states(self, entity_id, start, stop=None):
        return self.store.states(entity_id, start, stop)


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "history.db")


@pytest.fixture
def stocked(store):
    """A day of alice coming and going, plus a heartbeat and a stray entity."""
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    rows = []
    for hour, state in ((10, "not_home"), (12, "home"), (18, "not_home"), (20, "home")):
        rows.append(("person.alice", _ms(now - dt.timedelta(hours=24 - hour)), state))
    rows.append((HEARTBEAT_ENTITY, _ms(now - dt.timedelta(hours=1)), "ok"))
    # Nothing in the configuration reads this one. That is the case the
    # inventory exists to make visible.
    rows.append(("sensor.something_else", _ms(now - dt.timedelta(hours=2)), "41.5"))
    store.append(rows)
    return store


# --- the archive ----------------------------------------------------------

def test_the_inventory_names_every_entity_and_says_which_are_tracked(stocked):
    """`tracked` is the field the card exists for: an entity in the archive that
    nothing reads is otherwise invisible."""
    result = explore.archive_inventory(FakeSource(stocked), make_settings())
    assert result["available"] is True

    by_id = {e["entity_id"]: e for e in result["entities"]}
    assert by_id["person.alice"]["tracked"] is True
    assert by_id["person.alice"]["rows"] == 4
    assert by_id["sensor.something_else"]["tracked"] is False


def test_the_collectors_own_heartbeat_is_not_in_the_inventory(stocked):
    """It is bookkeeping, not a signal, and `tracked` is false for it -- so
    listing it drew an "unused" chip and, worse, made it the entity the card's
    "N not read" count was counting, under a sentence saying such entities are
    left over from an earlier configuration. Neither is true of the heartbeat.
    """
    result = explore.archive_inventory(FakeSource(stocked), make_settings())
    assert HEARTBEAT_ENTITY not in {e["entity_id"] for e in result["entities"]}
    assert not [e for e in result["entities"] if e["role"] == "heartbeat"]
    # The genuinely unread entity is still there; this hides one row, not the
    # question the card exists to answer.
    assert not next(e for e in result["entities"]
                    if e["entity_id"] == "sensor.something_else")["tracked"]


def test_a_configured_entity_with_no_rows_still_appears(stocked):
    """The worse half of the same question. person.bob is configured and has
    never produced a row, so a GROUP BY over the table cannot mention him -- and
    a household training with a member missing is exactly what wants saying."""
    result = explore.archive_inventory(FakeSource(stocked), make_settings())
    bob = next(e for e in result["entities"] if e["entity_id"] == "person.bob")
    assert bob["rows"] == 0
    assert bob["tracked"] is True
    assert bob["first"] is None


def test_kinds_are_read_off_the_values_not_the_entity_id(stocked):
    """Naming is the user's business. A sensor called anything at all is numeric
    if its values are numbers."""
    result = explore.archive_inventory(FakeSource(stocked), make_settings())
    by_id = {e["entity_id"]: e for e in result["entities"]}
    assert by_id["sensor.something_else"]["kind"] == "numeric"
    assert by_id["person.alice"]["kind"] == "presence"


def test_the_heartbeat_is_its_own_kind_and_not_a_presence_series(stocked):
    """Its value is "ok" on every row, so shape alone calls it presence -- and
    charting it would draw the fraction of each slot the collector spent at
    home. This package writes that row, so it may name it.

    Asserted through `entity_series` rather than the inventory, which no longer
    lists it: the endpoint is still reachable by entity id, so the guard has to
    hold there. Dropping it from one card is not the same as deleting it.
    """
    result = explore.entity_series(FakeSource(stocked), make_settings(),
                                   HEARTBEAT_ENTITY, days=2)
    assert result["available"] is True
    assert result["kind"] == "heartbeat"
    assert result["role"] == "heartbeat"


def test_every_kind_reported_is_one_the_panel_types(stocked):
    """`kind` is a union in `panel/src/types.ts`. A value not in it renders as
    whatever the card's fallback happens to be, and tsc cannot catch it."""
    result = explore.archive_inventory(FakeSource(stocked), make_settings())
    for entity in result["entities"]:
        assert entity["kind"] in {"presence", "numeric", "heartbeat", "other"}


def test_a_missing_archive_is_an_answer_not_an_error():
    """An Influx installation keeps no local archive. That is a sentence to
    render, not a 404 for the console."""
    class Influx:
        pass

    result = explore.archive_inventory(Influx(), make_settings())
    assert result["available"] is False
    assert "InfluxDB" in result["reason"]


# --- one entity -----------------------------------------------------------

def test_a_presence_entity_grids_to_the_same_home_frac_the_features_do(stocked):
    """The promise that matters most. If this ever fails, the panel is showing a
    number the model never saw."""
    result = explore.entity_series(FakeSource(stocked), make_settings(),
                                   "person.alice", days=2)
    assert result["available"] is True

    stop = pd.Timestamp(result["stop"])
    start = pd.Timestamp(result["start"])
    slots = features.grid(start, stop)
    events = stocked.seeded_states("person.alice", start.isoformat(), stop.isoformat())
    expected = features.slot_fraction(events, slots, config.HOME_STATE)

    got = [g["v"] for g in result["gridded"]]
    want = [None if pd.isna(v) else round(float(v), 4)
            for v in expected["frac"].to_numpy()]
    assert got == want


def test_an_unobserved_slot_is_null_and_never_a_zero(stocked):
    """`slot_fraction` returns NaN for a slot it did not see, because an
    unobserved slot is not an empty house. It has to reach the browser as null:
    a zero would draw as "away" and read as fact."""
    result = explore.entity_series(FakeSource(stocked), make_settings(),
                                   "person.alice", days=30)
    values = [g["v"] for g in result["gridded"]]
    assert None in values
    assert result["summary"]["nulls"] > 0


def test_a_distance_is_reported_in_the_kilometres_the_model_sees(store):
    """The archive holds metres; `distance_km` is what the feature table holds.
    Charting the raw number would put an axis on screen that no feature uses."""
    now = dt.datetime.now(dt.timezone.utc)
    store.append([("sensor.home_alice_distance",
                   _ms(now - dt.timedelta(minutes=30)), "4200.0")])
    result = explore.entity_series(FakeSource(store), make_settings(),
                                   "sensor.home_alice_distance", days=1)
    assert result["kind"] == "numeric"
    assert result["unit"] == "km"
    assert result["summary"]["max"] == pytest.approx(4.2)


def test_an_unknown_entity_is_an_answer_not_a_crash(stocked):
    result = explore.entity_series(FakeSource(stocked), make_settings(),
                                   "person.nobody", days=7)
    assert result["available"] is False


@pytest.mark.parametrize("asked, expected", [
    (None, explore.DEFAULT_DAYS),
    (0, explore.DEFAULT_DAYS),
    (-5, 1),
    (100_000, explore.MAX_DAYS),
    (14, 14),
])
def test_a_window_cannot_be_widened_past_the_cap(stocked, asked, expected):
    """The cap is about the response, not the database: it is what keeps a
    careless `?days=100000` from building a quarter-million-point payload."""
    result = explore.entity_series(FakeSource(stocked), make_settings(),
                                   "person.alice", days=asked)
    span = pd.Timestamp(result["stop"]) - pd.Timestamp(result["start"])
    assert round(span.total_seconds() / 86400) == expected


def test_the_raw_events_are_capped_from_the_recent_end(store):
    """When there are more transitions than fit, the tail is what anyone
    inspecting wants -- taking the head would show the oldest data and call it
    a truncation."""
    now = dt.datetime.now(dt.timezone.utc)
    store.append([("person.alice", _ms(now - dt.timedelta(seconds=i * 10)),
                   "home" if i % 2 else "not_home")
                  for i in range(explore.MAX_EVENTS + 500)])
    result = explore.entity_series(FakeSource(store), make_settings(),
                                   "person.alice", days=1)
    assert result["truncated"] is True
    assert len(result["events"]) == explore.MAX_EVENTS
    assert result["raw_rows"] > explore.MAX_EVENTS
    # The last event returned is the newest one in the store.
    assert result["events"][-1]["t"] >= result["events"][0]["t"]


# --- the feature table ----------------------------------------------------

@pytest.fixture
def parquet(tmp_path):
    """A small table with one column from each family that matters."""
    slots = pd.date_range("2026-01-01", periods=96, freq="30min", tz="UTC")
    frame = pd.DataFrame({
        "time": list(slots) * 2,
        "subject": ["alice"] * 96 + ["house"] * 96,
        "home_frac": [0.0, 1.0, None, 0.5] * 48,
        "state_now": [0.0, 1.0, 0.0, 0.5] * 48,
        "coverage": [1.0] * 192,
        "minutes_in_state": [10.0] * 192,
        "distance_km": [1.5] * 192,
        "is_weekend": [0.0] * 192,
        "zone_alice_office": [None] * 192,
        "other_house": [0.5] * 192,
        "y_36h": [1.0] * 192,
        "tgt36h_lag1d": [1.0] * 192,
        "tgt36h_lag2d": [1.0] * 192,
        "tgt36h_is_weekend": [0.0] * 192,
        "tgt36h_wclim4": [0.5] * 192,
        "next_alarm_h": [None] * 192,
    })
    path = tmp_path / "features.parquet"
    features.write(frame, path)
    return path


def test_the_column_inventory_never_reads_the_table(parquet, monkeypatch):
    """The promise that keeps the tab fast. Everything the inventory reports is
    in the parquet footer; if a future edit reaches for the data pages, this is
    what catches it -- not a stopwatch on a 134 MiB file nobody has in a test."""
    def boom(*_args, **_kwargs):
        raise AssertionError("feature_inventory must not read the table")

    monkeypatch.setattr(pd, "read_parquet", boom)

    result = explore.feature_inventory(parquet)
    assert result["available"] is True
    assert result["rows"] == 192
    assert result["columns"] == 16
    assert result["statistics"] is True
    # And the statistics really did come through, not just the schema.
    home_frac = next(c for c in result["browsable"] if c["name"] == "home_frac")
    assert home_frac["null_frac"] == pytest.approx(0.25)
    assert home_frac["max"] == pytest.approx(1.0)


def test_the_families_account_for_every_column(parquet):
    result = explore.feature_inventory(parquet)
    assert sum(f["columns"] for f in result["families"]) == result["columns"]
    assert "unknown" not in {f["family"] for f in result["families"]}


def test_the_per_horizon_columns_are_summarised_and_never_listed(parquet):
    """A thousand columns is not a dropdown. Only the origin families are browsable."""
    names = {c["name"] for c in result_browsable(parquet)}
    assert "home_frac" in names
    assert "distance_km" in names
    assert "tgt36h_lag1d" not in names
    assert "y_36h" not in names


def result_browsable(path):
    return explore.feature_inventory(path)["browsable"]


def test_a_missing_feature_table_says_it_has_not_been_built(tmp_path):
    result = explore.feature_inventory(tmp_path / "nothing.parquet")
    assert result["available"] is False
    assert "training run" in result["reason"]


def test_a_feature_series_reads_three_columns_not_the_table(parquet, monkeypatch):
    """It does read data -- but only the three columns it names, which is the
    same trick `train.load` uses and which the changelog records as three
    quarters of the training time."""
    seen = {}
    real = pd.read_parquet

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    result = explore.feature_series(parquet, "alice", "home_frac", days=30)

    assert result["available"] is True
    assert seen["columns"] == ["time", "subject", "home_frac"]
    assert result["summary"]["nulls"] > 0


def test_an_unknown_column_is_refused_before_it_reaches_the_reader(parquet, monkeypatch):
    """An unvalidated name reaching pyarrow is a stack trace where an answer
    belongs."""
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("validated too late"))
    result = explore.feature_series(parquet, "alice", "'; DROP TABLE", days=7)
    assert result["available"] is False


def test_an_unknown_subject_is_refused(parquet):
    result = explore.feature_series(parquet, "nobody", "home_frac", days=7)
    assert result["available"] is False


def test_a_lag_column_says_which_horizon_may_not_use_it(parquet):
    """Charting `tgt36h_lag1d` without the warning would present a column the
    model is forbidden to read as though it were a live feature."""
    result = explore.feature_series(parquet, "alice", "tgt36h_lag1d", days=7)
    assert result["safe_for"] == {"horizon_h": 36, "days": 1, "safe": False,
                                  "why": "home_frac 12 hours after the moment being "
                                         "predicted, so it cannot be known at +36 h"}

    safe = explore.feature_series(parquet, "alice", "tgt36h_lag2d", days=7)
    assert safe["safe_for"]["safe"] is True


# --- what one horizon uses ------------------------------------------------

@pytest.mark.parametrize("horizon", config.HORIZONS_H)
def test_the_recipe_agrees_with_the_leakage_gate_at_every_horizon(horizon):
    """One assertion covering all 48. `safe_daily_lags` owns the rule; this
    proves the page reports that rule rather than a second copy of it."""
    recipe = explore.horizon_recipe(horizon, {})
    safe = {lag["days"] for lag in recipe["daily_lags"] if lag["safe"]}
    assert safe == set(features.safe_daily_lags(horizon))

    for lag in recipe["daily_lags"]:
        assert (lag["why"] is None) is lag["safe"]


@pytest.mark.parametrize("horizon", config.HORIZONS_H)
def test_the_recipe_matches_what_the_model_actually_fits(horizon):
    """Guards against the page describing a feature list that has drifted from
    the one `train` builds and the pickle carries."""
    from occupancy_forecast import train

    recipe = explore.horizon_recipe(horizon, {})
    assert recipe["features"] == train.base_features()
    assert recipe["n_features"] == len(train.base_features())


@pytest.mark.parametrize("horizon", config.HORIZONS_H)
def test_the_recipe_names_the_anchor_the_fit_actually_uses(horizon):
    """The card must name the anchor the fit uses, whatever that turns out to be.

    `train.residual_base` answers `state_now` at every horizon today. Asserting
    against the function rather than against that string is the point: if it is
    ever made to vary again, this test follows it instead of quietly describing
    half the horizons as anchored on something they are not."""
    from occupancy_forecast import train

    recipe = explore.horizon_recipe(horizon, {})
    assert recipe["residual_base"] == train.residual_base(horizon)


def test_the_recipe_reads_nothing_from_disk(monkeypatch):
    """It is pure config plus arithmetic, and staying that way is what makes it
    safe to fetch on every change of a dropdown."""
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("read the table"))
    assert explore.horizon_recipe(24, {})["available"] is True


def test_a_horizon_reports_what_is_serving_it():
    """Three values, and the third is not the second.

    "none" is a model that was trained here and lost; `None` is a horizon
    nothing has ever been fitted for. Both publish nothing, so both draw the
    same grey cell -- but only one of them has a bake-off to show, which is
    what `best_baseline` and the Data tab hang off.
    """
    models = {6: {"metrics": {"ships": True}},
              24: {"metrics": {"ships": False, "best_baseline": "persistence"}}}
    assert explore.horizon_recipe(6, models)["served_by"] == "model"
    assert explore.horizon_recipe(24, models)["served_by"] == "none"
    assert explore.horizon_recipe(12, models)["served_by"] is None


def test_a_horizon_outside_the_set_is_an_answer():
    assert explore.horizon_recipe(999, {})["available"] is False


# --- model quality --------------------------------------------------------

def _metrics(horizon: int, ships: bool = True) -> dict:
    return {
        "horizon_h": horizon, "evaluation": "rolling-origin-embargoed",
        "n_folds": 4, "n_scored": 800, "n_train_final": 5000, "base_rate": 0.6,
        "brier": 0.05, "log_loss": 0.2, "auc": 0.9, "mae_frac": 0.08,
        "brier_fold_min": 0.04, "brier_fold_max": 0.07,
        "best_baseline": "persistence", "best_baseline_brier": 0.06,
        "skill_vs_best_baseline_pct": 16.7, "folds_beating_best_baseline": 3,
        "sign_test_p": 0.31, "ships": ships,
        "fallback": {"which": "persistence", "column": "state_now"},
        "baselines": {"persistence": 0.06, "base_rate": 0.24},
        "per_fold": [{"n": 200, "base_rate": 0.6, "brier": 0.05,
                      "log_loss": 0.2, "auc": 0.9, "mae_frac": 0.08}] * 4,
        "reliability": [{"bin_low": 0.0, "bin_high": 0.1, "n": 40,
                         "predicted": 0.05, "observed": 0.04}],
    }


@pytest.fixture
def models_dir(tmp_path):
    import json
    path = tmp_path / "models"
    path.mkdir()
    (path / "metrics.json").write_text(json.dumps({
        "model_version": "0.1.0",
        "trained_at": "2026-08-31T04:03:52+00:00",
        "duration_s": 121.4,
        "evaluation": "rolling-origin-embargoed",
        "horizons": {"1": _metrics(1), "24": _metrics(24, ships=False)},
        "failed": {},
    }))
    return path


def test_the_summary_reports_every_horizon_against_its_baseline(models_dir):
    result = explore.metrics_summary(models_dir)
    assert result["available"] is True
    assert result["shipping"] == 1
    assert [h["horizon_h"] for h in result["horizons"]] == [1, 24]


def test_the_summary_leaves_the_bulky_series_behind(models_dir):
    """All 48 with their fold lists and calibration curves attached is about a
    quarter of a megabyte for a card that shows one at a time."""
    result = explore.metrics_summary(models_dir)
    for row in result["horizons"]:
        assert "per_fold" not in row
        assert "reliability" not in row
        assert "baselines" not in row


def test_the_detail_carries_the_two_series_nothing_has_ever_drawn(models_dir):
    result = explore.metrics_detail(models_dir, 1)
    assert result["available"] is True
    assert len(result["per_fold"]) == 4
    assert result["reliability"][0]["observed"] == 0.04
    assert result["baselines"]["persistence"] == 0.06


def test_metrics_survive_a_corrupt_summary(tmp_path):
    """Mirrors `train.last_summary`: a truncated file is no answer rather than a
    JSONDecodeError out of an endpoint."""
    path = tmp_path / "models"
    path.mkdir()
    (path / "metrics.json").write_text('{"horizons": {"1": ')

    assert explore.metrics_summary(path)["available"] is False
    assert explore.metrics_detail(path, 1)["available"] is False


def test_the_pickles_answer_when_the_summary_is_lost(tmp_path):
    """`metrics.json` can go missing on a /data that has kept its models, and
    each artifact carries the same dict."""
    path = tmp_path / "models"
    path.mkdir()
    models = {1: {"metrics": _metrics(1)}}

    summary = explore.metrics_summary(path, models)
    assert summary["available"] is True
    assert summary["shipping"] == 1
    assert explore.metrics_detail(path, 1, models)["available"] is True


def test_no_models_at_all_is_an_answer(tmp_path):
    result = explore.metrics_summary(tmp_path / "nothing")
    assert result["available"] is False
    assert "trained" in result["reason"]


def test_an_untrained_horizon_says_so(models_dir):
    assert explore.metrics_detail(models_dir, 36)["available"] is False


def test_nan_never_reaches_the_json(stocked):
    """`float('nan')` is not valid JSON and a bare NaN in the body is a
    `JSON.parse` failure in the browser rather than a visible error here."""
    result = explore.entity_series(FakeSource(stocked), make_settings(),
                                   "person.alice", days=30)
    for point in result["gridded"]:
        assert point["v"] is None or point["v"] == point["v"]
    for value in result["summary"].values():
        assert value is None or value == value


# --- was it right? --------------------------------------------------------

def _slots(days: int):
    """The exact grid `verification` will build, so a test can write a forecast
    onto a slot rather than near one."""
    stop = pd.Timestamp.now(tz="UTC")
    return features.grid(stop - pd.Timedelta(days=days), stop)


@pytest.fixture
def home_all_week(store):
    """alice at home throughout, seeded before the window so every slot is
    observed. Truth is then 1.0 everywhere and any hole in the output is the
    forecast's, which is what these tests are about."""
    now = dt.datetime.now(dt.timezone.utc)
    store.append([("person.alice", _ms(now - dt.timedelta(days=30)), "home")])
    return store


def test_verification_puts_the_forecast_on_the_slot_it_was_about(home_all_week):
    """The join is an equality on the grid. If the write side and the read side
    disagreed about which slot a forecast was for, everything would still render
    -- just never line up -- so this asserts the alignment directly."""
    slots = _slots(2)
    home_all_week.append_forecasts(
        [("alice", int(t.timestamp() * 1000), 6, 0.75) for t in slots])

    result = explore.verification(FakeSource(home_all_week), make_settings(),
                                  "alice", 6, days=2)
    assert result["available"] is True
    assert result["served"] == len(slots)
    assert all(p["forecast"] == 0.75 for p in result["points"])
    # Home throughout, so truth is 1.0 and a 0.75 forecast is 0.0625 off.
    assert result["scored"] > 0
    assert result["brier"] == pytest.approx(0.0625, abs=1e-6)


def test_a_slot_nothing_was_published_for_is_a_null_and_not_a_zero(home_all_week):
    """The Part A pairing. An unserved horizon must reach the chart as a hole,
    because a 0.0 there reads as "certainly away" -- the sharpest lie the panel
    can tell, and the one the serving rule was changed to stop telling."""
    slots = _slots(2)
    published, withheld = slots[:-6], slots[-6:]
    home_all_week.append_forecasts(
        [("alice", int(t.timestamp() * 1000), 6, 0.75) for t in published])

    result = explore.verification(FakeSource(home_all_week), make_settings(),
                                  "alice", 6, days=2)
    by_t = {p["t"]: p for p in result["points"]}
    for t in withheld:
        assert by_t[t.isoformat()]["forecast"] is None
    assert result["served"] == len(published)
    # The score is over what was published, and says how much that was.
    assert result["scored"] <= len(published)
    assert str(len(published)) in result["summary"]


def test_the_holes_lead_the_summary_rather_than_the_score(home_all_week):
    """A horizon published a third of the time has a Brier over its good days.
    Stating the score first would invite reading it as the horizon's record."""
    slots = _slots(2)
    home_all_week.append_forecasts(
        [("alice", int(t.timestamp() * 1000), 6, 0.75) for t in slots[:10]])

    result = explore.verification(FakeSource(home_all_week), make_settings(),
                                  "alice", 6, days=2)
    assert result["summary"].startswith("published for 10 of")


def test_before_anything_has_come_due_it_says_so(home_all_week):
    """The normal state for the first `horizon` hours after a deploy, and
    forever for a horizon the model never earns. Not a 404, not an empty chart
    with a 0.000 Brier under it."""
    result = explore.verification(FakeSource(home_all_week), make_settings(),
                                  "alice", 6, days=2)
    assert result["available"] is False
    assert "come due" in result["reason"]


def test_a_forecast_about_the_future_is_not_scored_yet(home_all_week):
    """A +48 h forecast made now is about a slot two days out. It belongs in the
    table the moment it is published and on the chart only when it arrives."""
    ahead = pd.Timestamp.now(tz="UTC").ceil("30min") + pd.Timedelta(hours=6)
    home_all_week.append_forecasts(
        [("alice", int(ahead.timestamp() * 1000), 6, 0.75)])

    result = explore.verification(FakeSource(home_all_week), make_settings(),
                                  "alice", 6, days=2)
    assert result["available"] is False, "nothing in the window has come due"


def test_an_unknown_subject_or_horizon_is_an_answer(home_all_week):
    source = FakeSource(home_all_week)
    stranger = explore.verification(source, make_settings(), "carol", 6, days=2)
    assert stranger["available"] is False and "carol" in stranger["reason"]

    impossible = explore.verification(source, make_settings(), "alice", 99, days=2)
    assert impossible["available"] is False and "99" in impossible["reason"]


def test_an_influx_installation_keeps_no_such_record():
    """`explore` asks a source for its store and gets None. The reason has to
    say why rather than reading as a fault, because it never will have one."""
    class Storeless:
        pass

    result = explore.verification(Storeless(), make_settings(), "alice", 6)
    assert result["available"] is False
    assert "InfluxDB" in result["reason"]
