"""The contract between the panel and the API.

`panel/src/types.ts` declares the shape of every response the panel consumes;
this file asserts that a live response carries those fields. Nothing checks the
two against each other automatically: **add or rename a field in both places.**
"""

import numpy as np
import pytest

from occupancy_forecast import config, discover, server
from occupancy_forecast.tests.conftest import settings as make_settings

# --- panel/src/types.ts, transcribed -------------------------------------

STATUS_KEYS = {
    "display_name", "history", "days_until_training", "usable_presence_days",
    "people", "feature_groups",
    "served_by", "model_kind", "best_baseline", "worker",
    "mqtt", "listener", "last_train", "last_train_seconds",
    "next_train", "train_cadence", "training_in_progress",
    "training_started_at", "last_collect", "last_predict", "last_error",
}
MQTT_KEYS = {"connected", "error"}
FORECAST_KEYS = {"available", "predicted_at", "house", "horizons", "subjects", "night"}
OUT_ROUTINE_KEYS = {
    "probability", "weekday", "n_weekday", "n_out_weekday",
    "departure_hour", "departure_sd", "departure_from",
    "return_hour", "return_sd", "return_from", "fitted_at",
}
SUBJECT_FORECAST_KEYS = {"subject", "current", "observed_at", "curve",
                         "next_departure_h",
                         "next_arrival_h", "eta_minutes", "next_change", "out"}
NEXT_CHANGE_KEYS = {"direction", "in_hours", "at", "at_from"}
WORKER_KEYS = {"phase", "cycles", "seconds_since_phase", "stalled",
               "stalled_since", "stalled_in", "stalls"}
LISTENER_KEYS = {"connected"}          # the rest are optional in the type
FEATURE_GROUP_KEYS = {"active", "detail"}
CANDIDATE_KEYS = {"people", "zones", "groups", "countries", "has_proximity",
                  "schedules"}
SETTINGS_KEYS = {"people", "zones", "house_entity", "holiday_country",
                 "day_schedule",
                 "departure_threshold", "arrival_threshold", "crossing_min_hours",
                 "forecast_retention_days"}

# The Data tab. Each is a discriminated union on the panel side (`available`
# false-with-reason or true), so tsc refuses a view that forgets the empty state.
UNAVAILABLE_KEYS = {"available", "reason"}
ARCHIVE_KEYS = {"available", "span", "entities"}
ARCHIVE_SPAN_KEYS = {"first", "last", "rows", "days", "bytes"}
ARCHIVE_ENTITY_KEYS = {"entity_id", "rows", "first", "last", "kind", "role", "tracked"}
ENTITY_SERIES_KEYS = {
    "available", "entity_id", "kind", "role", "start", "stop", "unit",
    "raw_rows", "truncated", "events", "grid_minutes", "gridded",
    "gridded_label", "min_coverage", "summary",
}
SERIES_SUMMARY_KEYS = {"n", "nulls", "min", "max", "mean", "last"}
VERIFICATION_KEYS = {
    "available", "subject", "horizon_h", "grid_minutes", "start", "stop",
    "points", "slots", "served", "scored", "brier", "mae", "retention_days",
    "summary",
}
VERIFICATION_POINT_KEYS = {"t", "actual", "forecast"}
FEATURE_INVENTORY_KEYS = {"available", "path", "built_at", "bytes", "rows",
                          "columns", "row_groups", "grid_minutes", "statistics", "families",
                          "browsable"}
FEATURE_FAMILY_KEYS = {"family", "words", "columns", "null_frac"}
COLUMN_STAT_KEYS = {"name", "family", "null_frac", "min", "max"}
FEATURE_SERIES_KEYS = {"available", "subject", "column", "family", "words",
                       "grid_minutes", "points", "thinned", "safe_for",
                       "start", "stop", "summary"}
HORIZON_RECIPE_KEYS = {"available", "horizon_h", "target", "residual_base",
                       "n_features", "features", "families", "daily_lags",
                       "climatology", "columns_read", "embargo_hours",
                       "served_by", "ships", "kind"}
HORIZON_LAG_KEYS = {"days", "column", "safe", "why"}
METRICS_KEYS = {"available", "trained_at", "model_version", "evaluation",
                "duration_s", "shipping", "horizons", "failed"}
HORIZON_METRICS_KEYS = {
    "horizon_h", "brier", "log_loss", "auc", "mae_frac", "base_rate",
    "n_folds", "n_scored", "n_train_final", "best_baseline",
    "best_baseline_brier", "skill_vs_best_baseline_pct",
    "folds_beating_best_baseline", "sign_test_p", "ships",
    "brier_fold_min", "brier_fold_max",
    "kind", "rival_brier", "rival_kind",
}
FOLD_SCORE_KEYS = {"n", "base_rate", "brier", "log_loss", "auc", "mae_frac"}
RELIABILITY_BIN_KEYS = {"bin_low", "bin_high", "n", "predicted", "observed"}


@pytest.fixture
def fresh(monkeypatch):
    """Day one: no settings applied, no models, nothing connected."""
    monkeypatch.setitem(server._state, "settings", None)
    monkeypatch.setitem(server._state, "models", {})
    return server._status()


@pytest.fixture
def mature(monkeypatch):
    """A trained installation with some horizons publishing nothing."""
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._state, "models", {
        # One of each family and one that lost its bake-off: the shape a real
        # installation has, with the far end falling off the gate.
        1: {"metrics": {"ships": True, "kind": "dedicated"}},
        6: {"metrics": {"ships": True, "kind": "pooled"}},
        24: {"metrics": {"ships": False, "kind": None,
                         "best_baseline": "persistence"}},
        # 36 deliberately absent: the panel tells trained from untrained by
        # whether `best_baseline` has an entry, so one of each has to be here.
    })
    return server._status()


def test_status_carries_every_field_the_panel_reads(fresh, mature):
    for status in (fresh, mature):
        assert STATUS_KEYS <= set(status)
        assert MQTT_KEYS <= set(status["mqtt"])
        assert WORKER_KEYS <= set(status["worker"])
        assert LISTENER_KEYS <= set(status["listener"])


def test_the_forecast_the_now_tab_reads_is_the_one_that_was_published(monkeypatch):
    """The Overview tab reads what went to MQTT rather than recomputing: if it
    recomputed, the panel and the HA entities could disagree about the same
    instant."""
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._state, "last_predict", "2026-09-02T10:00:00+00:00")
    monkeypatch.setitem(server._state, "forecast", [{
        "subject": "alice", "current": 1.0, "curve": {"1": 0.9, "3": 0.4},
        # Older than `predicted_at`, and the chart's clock anchor: horizons are
        # measured from the feature row's slot, not from when the maths ran.
        "observed_at": "2026-09-02T09:30:00+00:00",
        "next_departure_h": 3, "next_arrival_h": None, "eta_minutes": None,
    }])

    payload = server.api_forecast()
    assert FORECAST_KEYS <= set(payload)
    assert payload["available"] is True
    assert payload["predicted_at"] == "2026-09-02T10:00:00+00:00"
    assert payload["house"] == config.HOUSE_SLUG
    assert payload["horizons"] == list(config.HORIZONS_H)
    for subject in payload["subjects"]:
        assert SUBJECT_FORECAST_KEYS <= set(subject)
    # JSON has no integer keys and the panel indexes the curve by String(h).
    assert all(isinstance(k, str) for k in payload["subjects"][0]["curve"])
    # The axis anchor, and it is NOT `predicted_at`.
    assert payload["subjects"][0]["observed_at"] == "2026-09-02T09:30:00+00:00"
    # The combined answer the card renders, present even when the forecast in
    # memory predates it -- `.get` on the server side, null here.
    assert "next_change" in payload["subjects"][0]
    # And it arrives SPARSE: the panel draws a hole at the unserved +2 h, and a
    # densified curve would put a number there that no model produced.
    assert payload["subjects"][0]["curve"] == {"1": 0.9, "3": 0.4}


def test_a_forecast_that_has_not_run_yet_says_so_rather_than_answering_empty(monkeypatch):
    """Day one, and the first few minutes of every restart. An empty chart with
    no explanation is the state this panel exists to stop rendering."""
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._state, "forecast", [])
    payload = server.api_forecast()
    assert payload["available"] is False
    assert payload["subjects"] == []


def test_the_panel_can_name_the_add_on_that_served_it(monkeypatch):
    """The header and the browser tab. Two add-ons, one bundle, so the name has
    to arrive over the API rather than be baked in at build time."""
    monkeypatch.setattr(config, "_topic_prefix", "occupancy_forecast_edge")
    assert server._status()["display_name"] == "Occupancy Forecast Edge"


def test_served_by_is_a_horizon_to_verdict_mapping(fresh, mature):
    """What the horizon strip draws. Keyed by the WHOLE grid, not the loaded
    artifacts -- a fresh install would otherwise send an empty map, and a
    failed pickle would drop out of the denominator."""
    for status in (fresh, mature):
        assert set(status["served_by"]) == {str(h) for h in config.HORIZONS_H}
        # The line that fails if "baseline:<name>" ever comes back. Nothing
        # downstream may have to split a status value to find out what it says.
        assert set(status["served_by"].values()) <= {"model", "none"}

    served = mature["served_by"]
    assert served["1"] == "model"
    assert served["24"] == "none", "the baseline won, so nothing is published"
    assert served["36"] == "none", "never trained, so nothing is published"
    assert set(fresh["served_by"].values()) == {"none"}


def test_best_baseline_names_the_winner_only_where_a_model_lost(fresh, mature):
    """Two horizons publish nothing for two different reasons and draw the same
    grey cell, so the tooltip tells them apart. Absence is the signal; a null
    would mean "trained, winner has no name", which is not a state that
    exists."""
    assert fresh["best_baseline"] == {}

    beaten = mature["best_baseline"]
    assert beaten == {"24": "persistence"}
    assert "1" not in beaten and "6" not in beaten, "a model serves these"
    assert "36" not in beaten, "nothing was ever trained for it"


def test_model_kind_names_the_family_without_disturbing_served_by(fresh, mature):
    """`served_by` keeps the two values the strip counts; WHICH family answered
    rides beside it. Folding the family in as "model:pooled" would break every
    `=== 'model'` silently."""
    assert fresh["model_kind"] == {}

    kinds = mature["model_kind"]
    assert set(kinds) == {h for h, v in mature["served_by"].items() if v == "model"}
    assert set(kinds.values()) <= {"dedicated", "pooled"}
    assert kinds == {"1": "dedicated", "6": "pooled"}
    # A horizon nothing serves is absent rather than null: there is no family
    # to name, and `undefined` is what the panel's Record lookup gives.
    assert "24" not in kinds


def test_feature_group_details_keep_the_three_shapes_the_panel_formats(mature):
    """`detail` is a list, a mapping or a sentence, which is why the panel has a
    `formatDetail`. The old page interpolated it and rendered a Python list on
    screen."""
    groups = mature["feature_groups"]
    assert groups
    for info in groups.values():
        assert FEATURE_GROUP_KEYS <= set(info)
        assert isinstance(info["detail"], (str, list, dict))

    assert isinstance(groups["presence"]["detail"], list)
    # A sentence, not a person->zone mapping: a zone belongs to the household,
    # and the row carries a rename count that no mapping could.
    assert isinstance(groups["zones"]["detail"], str)


def test_candidates_carries_every_field_the_pickers_read():
    states = [
        {"entity_id": "person.alice", "attributes": {"friendly_name": "Alice"}},
        {"entity_id": "zone.office", "attributes": {"friendly_name": "Office"}},
        {"entity_id": "group.household",
         "attributes": {"friendly_name": "Household",
                        "entity_id": ["person.alice"]}},
    ]
    candidates = discover.candidates(states)
    assert CANDIDATE_KEYS <= set(candidates)
    for entity in candidates["people"] + candidates["zones"] + candidates["groups"]:
        assert {"entity_id", "name"} <= set(entity)
    for country in candidates["countries"]:
        assert {"code", "name"} <= set(country)


def test_the_saved_settings_round_trip_through_the_form():
    """GET /api/config seeds the form; POST sends back these same keys."""
    from dataclasses import asdict
    assert SETTINGS_KEYS <= set(asdict(make_settings()))


def test_every_settable_field_the_form_sends_is_on_a_validator_allowlist():
    """`typed_patch` drops what it does not recognise, silently: a field added
    to the form and forgotten here never saves and never errors."""
    from occupancy_forecast import server

    settable = SETTINGS_KEYS - {"holiday_country"}          # optional in the patch
    payload = {"people": [], "zones": [], "house_entity": None,
               "day_schedule": None, "departure_threshold": 0.5,
               "arrival_threshold": 0.5, "crossing_min_hours": 2,
               "forecast_retention_days": 30}
    applied = {**server.typed_patch(payload),
               **server.crossing_patch(payload, make_settings())}
    assert settable <= set(applied), settable - set(applied)


# --- the Data tab ---------------------------------------------------------

def test_the_archive_carries_every_field_the_data_tab_reads(tmp_path):
    from occupancy_forecast import explore
    from occupancy_forecast.sources.store import HistoryStore

    class Source:
        store = HistoryStore(tmp_path / "history.db")

    Source.store.append([("person.alice", 1_700_000_000_000, "home")])
    archive = explore.archive_inventory(Source(), make_settings())

    assert ARCHIVE_KEYS <= set(archive)
    assert ARCHIVE_SPAN_KEYS <= set(archive["span"])
    for entity in archive["entities"]:
        assert ARCHIVE_ENTITY_KEYS <= set(entity)


def test_an_entity_series_carries_every_field_the_chart_reads(tmp_path):
    import datetime as dt

    from occupancy_forecast import explore
    from occupancy_forecast.sources.store import HistoryStore

    class Source:
        store = HistoryStore(tmp_path / "history.db")

    now = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    Source.store.append([("person.alice", now - 3_600_000, "home")])
    series = explore.entity_series(Source(), make_settings(), "person.alice", days=1)

    assert ENTITY_SERIES_KEYS <= set(series)
    assert SERIES_SUMMARY_KEYS <= set(series["summary"])
    for point in series["gridded"]:
        assert {"t", "v", "coverage"} <= set(point)


def test_a_forecast_without_an_anchor_still_serves(monkeypatch):
    """A missing `observed_at` costs the chart its clock labels, not the whole
    endpoint. The panel's `at` prop is optional for exactly this reason."""
    monkeypatch.setitem(server._state, "settings", make_settings())
    monkeypatch.setitem(server._state, "last_predict", "2026-09-02T10:00:00+00:00")
    monkeypatch.setitem(server._state, "forecast", [{
        "subject": "alice", "current": 1.0, "curve": {"1": 0.9},
        "next_departure_h": None, "next_arrival_h": None, "eta_minutes": None,
    }])
    payload = server.api_forecast()
    assert payload["available"] is True
    assert payload["subjects"][0]["observed_at"] is None


def test_next_change_carries_every_field_the_row_reads():
    """One answer per person: the model's verdict, timed by the routine."""
    import pandas as pd

    from occupancy_forecast import config, predict

    config.configure(make_settings())
    at = pd.Timestamp("2026-09-07T06:00", tz=config.TIMEZONE).tz_convert("UTC")
    change = predict._next_change(None, "alice", at, 3, None)
    assert NEXT_CHANGE_KEYS <= set(change)
    # No routine at all still answers, from the crossing.
    assert change["at_from"] == "crossing" and change["at"] is not None


def test_the_out_routine_carries_every_field_the_card_reads():
    """The card shows a median beside the count behind it, on purpose. Dropping
    a count would leave two very different claims rendering identically."""
    import datetime as dt

    import pandas as pd

    from occupancy_forecast import config, departure, outing
    from occupancy_forecast.tests.test_outing import _table, _labelled

    config.configure(make_settings(zones=["zone.alice_office"],
                                   zone_names={"zone.alice_office": "Alice Office"}))
    begin = dt.date(2026, 1, 5)
    went = {(begin + dt.timedelta(days=7 * w + d)).isoformat()
            for w in range(12) for d in (1, 4)}
    routine = outing.fit_routine(_labelled(_table(went, days=84)))

    today = outing.today(routine, "alice",
                         pd.Timestamp("2026-03-31T06:00Z"))
    assert OUT_ROUTINE_KEYS <= set(today)
    # Nullable and never coerced: the panel's type says `| null` for both hours
    # and the card renders a sentence rather than a zero when they are absent.
    assert today["departure_hour"] is None or isinstance(today["departure_hour"], float)
    assert today["departure_from"] in {"weekday", "overall"}


def test_verification_carries_every_field_the_was_it_right_card_reads(tmp_path):
    import datetime as dt

    import pandas as pd

    from occupancy_forecast import config, explore, features
    from occupancy_forecast.sources.store import HistoryStore

    store = HistoryStore(tmp_path / "history.db")

    class Source:
        def __init__(self):
            self.store = store

        def seeded_states(self, entity_id, start, stop=None, seed_days=14):
            return self.store.seeded_states(entity_id, start, stop, seed_days)

    now = dt.datetime.now(dt.timezone.utc)
    store.append([("person.alice",
                   int((now - dt.timedelta(days=5)).timestamp() * 1000), "home")])
    stop = pd.Timestamp.now(tz="UTC")
    slots = features.grid(stop - pd.Timedelta(days=1), stop)
    store.append_forecasts([("alice", int(t.timestamp() * 1000), 6, 0.8)
                            for t in slots])

    result = explore.verification(Source(), store, make_settings(), "alice", 6,
                                  days=1)
    assert VERIFICATION_KEYS <= set(result)
    for point in result["points"]:
        assert VERIFICATION_POINT_KEYS <= set(point)
    # `forecast` is nullable and `actual` is nullable; the panel's type says so
    # and the chart's whole behaviour turns on it, so neither may be coerced.
    assert all(p["forecast"] is None or isinstance(p["forecast"], float)
               for p in result["points"])
    assert config.FORECAST_RETENTION_DAYS == result["retention_days"]


def test_an_unavailable_verification_still_answers_in_the_explorable_shape(tmp_path):
    from occupancy_forecast import explore
    from occupancy_forecast.sources.store import HistoryStore

    class Source:
        store = HistoryStore(tmp_path / "empty.db")

    result = explore.verification(Source(), Source.store, make_settings(),
                                  "alice", 6, days=1)
    assert UNAVAILABLE_KEYS <= set(result)
    assert result["available"] is False


def test_the_horizon_recipe_carries_every_field_the_leakage_card_reads():
    from occupancy_forecast import explore

    recipe = explore.horizon_recipe(36, {})
    assert HORIZON_RECIPE_KEYS <= set(recipe)
    for family in recipe["families"]:
        assert {"family", "words", "columns"} <= set(family)
    for lag in recipe["daily_lags"]:
        assert HORIZON_LAG_KEYS <= set(lag)


def test_the_feature_inventory_and_series_carry_what_the_cards_read(tmp_path):
    import pandas as pd

    from occupancy_forecast import explore, features

    path = tmp_path / "features.parquet"
    features.write(pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC"),
        "subject": ["alice"] * 4,
        "home_frac": [0.0, 1.0, None, 0.5],
    }), path)

    inventory = explore.feature_inventory(path)
    assert FEATURE_INVENTORY_KEYS <= set(inventory)
    for family in inventory["families"]:
        assert FEATURE_FAMILY_KEYS <= set(family)
    for column in inventory["browsable"]:
        assert COLUMN_STAT_KEYS <= set(column)

    series = explore.feature_series(path, "alice", "home_frac", days=30)
    assert FEATURE_SERIES_KEYS <= set(series)
    assert SERIES_SUMMARY_KEYS <= set(series["summary"])


def test_an_empty_fold_reaches_the_panel_as_a_null_and_not_a_zero():
    """`_scores_by_fold` emits an entry for every fold INDEX, because `ships`
    walks the list positionally. The padding is NaN, which serialises to null
    -- the fold list is the one place in this API that is legitimately sparse.
    """
    import math

    from occupancy_forecast import evaluate

    empty = evaluate.summarize([evaluate.score(np.array([]), np.array([]))])
    fold = empty["per_fold"][0]
    assert FOLD_SCORE_KEYS <= set(fold)
    assert fold["n"] == 0, "an empty fold still reports its row count"
    assert math.isnan(fold["brier"]), (
        "an empty fold must not score 0.0 -- a zero Brier is a perfect week, "
        "which is the opposite of what no data means")


def test_the_metrics_carry_every_field_the_quality_cards_read(tmp_path):
    import json

    from occupancy_forecast import explore

    metrics = {name: 0.5 for name in HORIZON_METRICS_KEYS}
    metrics.update(horizon_h=1, ships=True, best_baseline="persistence",
                   n_folds=2, n_scored=10, n_train_final=99,
                   folds_beating_best_baseline=2,
                   per_fold=[{name: 0.5 for name in FOLD_SCORE_KEYS}],
                   reliability=[{name: 0.5 for name in RELIABILITY_BIN_KEYS}],
                   baselines={"persistence": 0.6}, fallback={})

    models = tmp_path / "models"
    models.mkdir()
    (models / "metrics.json").write_text(json.dumps({
        "model_version": "0.1.0", "trained_at": None, "duration_s": None,
        "evaluation": "rolling-origin-embargoed",
        "horizons": {"1": metrics}, "failed": {},
    }))

    summary = explore.metrics_summary(models)
    assert METRICS_KEYS <= set(summary)
    for row in summary["horizons"]:
        assert HORIZON_METRICS_KEYS <= set(row)

    detail = explore.metrics_detail(models, 1)
    assert HORIZON_METRICS_KEYS <= set(detail)
    assert {"per_fold", "reliability", "baselines", "fallback"} <= set(detail)
    for fold in detail["per_fold"]:
        assert FOLD_SCORE_KEYS <= set(fold)
    for b in detail["reliability"]:
        assert RELIABILITY_BIN_KEYS <= set(b)


def test_a_nan_in_the_metrics_reaches_the_panel_as_null_not_as_a_500(tmp_path):
    """`metrics.json` is written with `allow_nan=True`; FastAPI renders with
    `allow_nan=False`, so a NaN reaching a handler's return value is a 500 for
    the whole endpoint. Rendered through the app's own response class, as the
    suite ships no HTTP client."""
    import json

    from fastapi.responses import JSONResponse

    from occupancy_forecast import evaluate, explore

    scored = evaluate.score(np.array([0.0, 1.0, 1.0]), np.array([0.2, 0.8, 0.9]))
    empty = evaluate.score(np.array([]), np.array([]))
    pooled = evaluate.summarize([scored, empty])
    metrics = {name: 0.5 for name in HORIZON_METRICS_KEYS}
    metrics.update(horizon_h=1, ships=True, kind="pooled", n_folds=2,
                   n_scored=3, n_train_final=3, folds_beating_best_baseline=1,
                   best_baseline="none", best_baseline_brier=float("nan"),
                   skill_vs_best_baseline_pct=float("nan"),
                   brier_fold_min=pooled["brier_fold_min"],
                   per_fold=pooled["per_fold"], reliability=[], baselines={},
                   fallback={}, rival_brier=None, rival_kind=None)
    models = tmp_path / "models"
    models.mkdir()
    (models / "metrics.json").write_text(json.dumps({
        "model_version": "0.1.0", "trained_at": None, "duration_s": None,
        "evaluation": "rolling-origin-embargoed",
        "horizons": {"1": metrics}, "failed": {},
    }))
    assert "NaN" in (models / "metrics.json").read_text(), "the fixture must carry the literal"

    summary = explore.metrics_summary(models)
    detail = explore.metrics_detail(models, 1)
    for payload in (summary, detail):
        JSONResponse(content=payload).body          # raises ValueError before the fix
    assert summary["horizons"][0]["best_baseline_brier"] is None
    assert detail["per_fold"][1]["brier"] is None, "the empty fold is null, not 0.0"
    assert detail["per_fold"][0]["brier"] == pooled["per_fold"][0]["brier"], \
        "a real number passes through unrounded"


def test_every_explorer_endpoint_answers_rather_than_raising_on_day_one():
    """The convention the whole tab depends on: a fresh install has no archive
    and no models, and that is a sentence to render, not a 404 for the console."""
    from occupancy_forecast import explore

    class NoStore:
        """An Influx source: no local archive at all."""

    from pathlib import Path
    nowhere = Path("/nonexistent")

    for answer in (explore.archive_inventory(NoStore(), make_settings()),
                   explore.entity_series(NoStore(), make_settings(),
                                         "person.alice", days=7),
                   explore.feature_inventory(nowhere / "features.parquet"),
                   explore.feature_series(nowhere / "features.parquet",
                                          "alice", "home_frac", days=7),
                   explore.metrics_summary(nowhere),
                   explore.metrics_detail(nowhere, 1)):
        assert UNAVAILABLE_KEYS <= set(answer)
        assert answer["available"] is False
        assert answer["reason"]

    # The one that does answer on day one, because it reads nothing: what a
    # horizon WOULD use is knowable before anything has been collected.
    assert explore.horizon_recipe(24, {})["available"] is True
