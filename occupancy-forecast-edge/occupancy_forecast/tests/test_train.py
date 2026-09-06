"""Training and evaluation tests, on a synthetic table with planted structure.

The table is built from an RNG and sized off `train.origin_features()`, so it
cannot drift from the feature list. The signal is planted, so the assertions are
not tautological: the model is asked to recover something we put there.

Runnable two ways:
    pytest app/tests/test_train.py
    python3 app/tests/test_train.py
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from occupancy_forecast import baseline, config, evaluate, features, train  # noqa: E402

HORIZON = 6

# The pooled fit is ONE model over every horizon, so a test that trains on all
# 48 pays for all 48. Four is enough to exercise the machinery -- one short, one
# mid, and the pair either side of the h=24/25 lag-gate boundary -- and keeps
# the suite in seconds. `train_base` takes the subset as an argument precisely
# so this is a parameter and not a fixture that lies about what it built.
TEST_HORIZONS = (1, HORIZON, 24, 25, 48)


def _feature_table(days: int = 160) -> Path:
    """A synthetic occupancy table with a real daily rhythm in it."""
    rng = np.random.default_rng(0)
    slots = pd.date_range("2026-01-01", periods=days * config.SLOTS_PER_DAY,
                          freq=f"{config.GRID_MINUTES}min", tz="UTC", name="time")

    frames = []
    for subject in config.all_slugs():
        local = slots.tz_convert(config.TIMEZONE)
        hour = local.hour + local.minute / 60
        # Home overnight, out in the middle of the day, with noise.
        p = np.where((hour < 8) | (hour > 18), 0.95, 0.25)
        p = np.clip(p + rng.normal(0, 0.15, len(slots)), 0, 1)
        frame = pd.DataFrame({"time": slots, "subject": subject, "home_frac": p})
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    table["state_now"] = table["home_frac"]
    table["coverage"] = 1.0
    table["minutes_in_state"] = rng.uniform(0, 600, len(table))

    local = table["time"].dt.tz_convert(config.TIMEZONE)
    table = pd.concat([table, features._cyclical(local)], axis=1)
    table = features._add_cross_subject(table)
    for column in features.zone_columns():
        table[column] = 0.0
    # Proximity, planted with the real relationship: far away implies not home.
    table["distance_km"] = np.where(table["home_frac"] > 0.5,
                                    rng.uniform(0, 0.2, len(table)),
                                    rng.uniform(1, 40, len(table)))
    table["distance_delta_30m"] = rng.normal(0, 2, len(table))
    table["distance_delta_60m"] = rng.normal(0, 3, len(table))
    table["dir_towards"] = rng.uniform(0, 1, len(table))
    table["dir_away"] = rng.uniform(0, 1, len(table))

    table = features._add_horizon_columns(table)
    for column in features.BUILT_NOT_SHIPPED:
        table[column] = np.nan

    # Backstop so this table can never fall behind the feature list again: any
    # ORIGIN column the construction above did not produce gets noise. The
    # horizon-relative columns are not listed here because `_add_horizon_columns`
    # above mints all of them; only the origin block is hand-built and can drift.
    # A missing column would otherwise fail collection with a ValueError from
    # `read_wide`, which is how the proximity features announced themselves.
    for column in train.origin_features():
        if column not in table.columns and column != "subject":
            table[column] = rng.normal(0, 1, len(table))

    path = Path(tempfile.mkdtemp()) / "features.parquet"
    features.write(table.sort_values(["subject", "time"]).reset_index(drop=True), path)
    return path


# Built lazily so importing the module does not do 15 folds of real work, and
# so the conftest fixture has configured an installation first.
_CACHE: dict = {}


def _fitted():
    """Both families, on shared windows, plus the gate's verdicts.

    Trained once for the whole module: two families over five horizons is real
    work, and every test below reads the same run.
    """
    if not _CACHE:
        path = _feature_table()
        windows, geometry = train.shared_windows(path)
        wide = train.read_wide(path)
        rungs = {h: baseline.run(wide, h, geometry=geometry, windows=windows)
                 for h in TEST_HORIZONS}

        ded_est, ded_scored, ded_rows = train.train_dedicated(path, HORIZON, windows)
        pool_est, pool_scored, pool_rows = train.train_pooled(
            path, windows, horizons=TEST_HORIZONS, n_jobs=1)

        dedicated = train._candidate(
            HORIZON, "dedicated", ded_scored, f"y_{HORIZON}h", len(windows),
            ded_rows, rungs[HORIZON], wide)
        part = pool_scored[pool_scored[features.HORIZON_COLUMN] == float(HORIZON)]
        pooled = train._candidate(
            HORIZON, "pooled", part, features.TARGET_COLUMN, len(windows),
            pool_rows, rungs[HORIZON], wide)

        _CACHE.update(path=path, windows=windows, geometry=geometry, rungs=rungs,
                      estimator=pool_est, pooled_scored=pool_scored,
                      dedicated=dedicated, pooled=pooled, wide=wide)
    return _CACHE


def _metrics():
    """The winning candidate at HORIZON, which is what actually serves."""
    return train.choose(_fitted()["dedicated"], _fitted()["pooled"])


# ---------------------------------------------------------------------------
# The validation design
# ---------------------------------------------------------------------------

def test_validation_is_rolling_origin_not_a_single_holdout():
    assert _metrics().n_folds > 1, "one fold: back on a single holdout"
    assert _metrics().evaluation == "rolling-origin-embargoed"
    assert len(_metrics().per_fold) == _metrics().n_folds


def test_the_reported_number_carries_a_spread():
    assert _metrics().brier_fold_min <= _metrics().brier <= _metrics().brier_fold_max


def test_the_embargo_is_at_least_the_horizon():
    """A row just before a test window has its target inside that window.

    Without a gap the model trains on the answer -- and because the table is
    sorted by (subject, time) across three subjects, a row-count embargo would
    protect nothing. The gap must be measured in time.
    """
    for horizon in config.HORIZONS_H:
        assert evaluate.embargo_for(horizon) > pd.Timedelta(hours=horizon)


def test_folds_do_not_overlap_and_respect_the_embargo():
    times = pd.Series(pd.date_range("2026-01-01", periods=4000, freq="30min",
                                    tz="UTC"))
    embargo = evaluate.embargo_for(HORIZON)
    folds = evaluate.calendar_folds(times, embargo=embargo)
    assert folds
    for fold in folds:
        latest_train = times.iloc[fold.train_idx].max()
        earliest_test = times.iloc[fold.test_idx].min()
        assert earliest_test - latest_train >= embargo
    for a, b in zip(folds, folds[1:]):
        assert a.test_stop <= b.test_start


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------

def test_brier_is_a_proper_score():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    perfect = evaluate.score(y, np.array([1.0, 1.0, 0.0, 0.0]))
    hedged = evaluate.score(y, np.array([0.5, 0.5, 0.5, 0.5]))
    wrong = evaluate.score(y, np.array([0.0, 0.0, 1.0, 1.0]))
    assert perfect.brier < hedged.brier < wrong.brier


def test_mae_is_not_used_to_choose():
    """MAE ranks a confident wrong guess above a calibrated one; Brier does not.

    Measured on the real data, ranking the baselines by MAE on the fractional
    target inverted the Brier ranking entirely. This test pins the disagreement
    so nobody "simplifies" the metric back to MAE.
    """
    y = np.array([1.0, 0.0])
    confident = evaluate.score(y, np.array([1.0, 1.0]))   # right once, wrong once
    calibrated = evaluate.score(y, np.array([0.5, 0.5]))
    assert confident.mae_frac == calibrated.mae_frac      # MAE cannot separate them
    assert confident.brier > calibrated.brier             # Brier can


def test_sign_test_knows_eight_folds_prove_little():
    """The reason TEST_DAYS moved from 14 to 7. See evaluate.TEST_DAYS."""
    assert evaluate.sign_test(6, 8) > 0.05, "6/8 must not read as significant"
    assert evaluate.sign_test(8, 8) < 0.05
    assert evaluate.sign_test(12, 15) < 0.05


def test_auc_is_half_for_a_constant_prediction():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert abs(evaluate.score(y, np.full(4, 0.7)).auc - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# The model and the gate
# ---------------------------------------------------------------------------

def test_the_model_learns_the_planted_rhythm():
    assert _metrics().brier < _metrics().base_rate * (1 - _metrics().base_rate), (
        f"Brier {_metrics().brier:.3f} no better than predicting the base rate")


def test_the_baselines_are_scored_on_the_same_folds():
    assert _metrics().baselines, "no baseline ladder recorded"
    assert "persistence" in _metrics().baselines
    assert _metrics().best_baseline in _metrics().baselines


def test_the_gate_refuses_a_model_that_does_not_beat_the_baseline():
    """A gate that always says yes is decoration."""
    poor = train.Metrics(
        horizon_h=HORIZON, evaluation="x", n_folds=15, n_scored=100,
        n_train_final=100, base_rate=0.5, brier=0.30, log_loss=0.0, auc=0.5,
        mae_frac=0.0, brier_fold_min=0.0, brier_fold_max=0.0,
        best_baseline="persistence", best_baseline_brier=0.20,
        skill_vs_best_baseline_pct=-50.0, folds_beating_best_baseline=2,
        sign_test_p=0.5, ships=False)
    assert not poor.ships


def test_the_fold_gate_refuses_a_minority_but_not_an_undecided_record():
    """The rule is "was it proven a minority", not "did it prove a majority".

    The strict majority it replaced was refusing models with +11..+15% skill at
    9 of 19 folds -- where the sign test says p = 1.000, i.e. nothing at all.
    Measuring the SERVED curve prequentially, that cost 5.07% Brier overall and
    9.0% past +30 h. That measurement was made when a refused horizon still
    served its baseline, so it compares two dense curves; the cost is now
    larger, because a refused horizon is not published at all. See
    `train.fold_record_allows`."""
    # Proven worse than a coin flip: refused, which is the one-good-fortnight
    # case the majority rule existed to catch.
    assert not train.fold_record_allows(4, 19)
    assert evaluate.sign_test(15, 19) < 0.05

    # Undecided: 9/19 proves nothing either way, so the skill bar decides.
    assert train.fold_record_allows(9, 19)
    assert evaluate.sign_test(10, 19) == pytest.approx(1.0)
    assert train.fold_record_allows(10, 19)

    # A majority is never refused, at any fold count.
    for n in range(1, 25):
        for beat in range(n // 2 + 1, n + 1):
            assert train.fold_record_allows(beat, n), (beat, n)

    assert not train.fold_record_allows(0, 19)
    assert not train.fold_record_allows(0, 0), "no folds is no evidence"


def test_the_residual_target_is_what_is_fitted():
    """The model must be learning a delta off state_now, not the level.

    Direct-target fitting measured *worse than persistence* at 24 h and beyond,
    because at multiples of 24 h the baseline is essentially the identity on
    state_now and a tree approximates identity badly.
    """
    assert train.RESIDUAL_BASE == "state_now"
    frame = train.to_long(train.read_wide(_fitted()["path"]))
    frame = frame[frame[features.HORIZON_COLUMN] == float(HORIZON)].head(500)
    residual = frame[features.TARGET_COLUMN] - frame[train.RESIDUAL_BASE]
    assert abs(residual.mean()) < abs(frame[features.TARGET_COLUMN].mean()), (
        "the residual should be smaller than the level it is taken from")


def test_the_feature_list_travels_with_the_model():
    """Otherwise a feature added here desynchronises silently from serving."""
    verdicts = {HORIZON: _metrics()}
    with tempfile.TemporaryDirectory() as tmp:
        path = train.save(_fitted()["estimator"], verdicts, Path(tmp),
                          train.POOLED_NAME, train.base_features())
        with path.open("rb") as fh:
            artifact = pickle.load(fh)
    assert artifact["features"] == train.base_features()
    assert artifact["kind"] == "pooled"
    assert artifact["metrics"][HORIZON]["ships"] == _metrics().ships


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
# The training summary on disk
#
# `metrics.json` is the only record that survives a restart. `last_train` used to
# live in memory alone, so every restart claimed the add-on had never trained --
# while sitting on 48 models with the timestamp written beside them.
# ---------------------------------------------------------------------------

def test_the_summary_records_when_and_how_long(tmp_path):
    train.write_summary({"1": {"ships": True}}, tmp_path, duration_s=12.5)
    summary = train.last_summary(tmp_path)
    assert summary is not None
    assert summary["duration_s"] == 12.5
    assert summary["trained_at"]


def test_the_duration_can_be_stamped_after_the_fact(tmp_path):
    """The number is not knowable when the summary is written: fitting the models
    is only the middle of the job, with the feature table and the ETA models
    either side."""
    train.write_summary({"1": {"ships": True}}, tmp_path)
    assert train.last_summary(tmp_path)["duration_s"] is None

    train.stamp_duration(251.44, tmp_path)
    assert train.last_summary(tmp_path)["duration_s"] == 251.4
    # Stamping must not lose what was already there.
    assert train.last_summary(tmp_path)["trained_at"]


def test_no_models_and_an_unreadable_summary_are_both_just_no_answer(tmp_path):
    """A corrupt file is not a reason to refuse to start."""
    assert train.last_summary(tmp_path) is None
    train.summary_path(tmp_path).write_text("{ this is not json")
    assert train.last_summary(tmp_path) is None


# ---------------------------------------------------------------------------
# Two families, one table
#
# The wide parquet feeds both: `load_for` selects one horizon's columns for a
# dedicated fit, `to_long` melts every horizon for the pooled one. What is
# asserted below is that the melt is lossless, that both families see every
# subject including the house, that the gate reaches a verdict per horizon, and
# that a stale artifact is refused rather than unpickled.
# ---------------------------------------------------------------------------

def test_a_table_behind_the_feature_list_still_explains_itself(tmp_path):
    """The friendly error has to survive reading the schema instead of the table."""
    path = tmp_path / "thin.parquet"
    pd.DataFrame({"time": pd.to_datetime(["2026-01-01"], utc=True),
                  "subject": ["alice"]}).to_parquet(path)
    with pytest.raises(ValueError, match="feature list has moved ahead"):
        train.read_wide(path)


def test_the_house_is_a_training_subject_in_both_families():
    """It was briefly a combiner over the people's forecasts. That measured
    0/48 shipping and lost to its own baselines, so it is a subject again."""
    frame = train.to_long(train.read_wide(_fitted()["path"]))
    assert config.HOUSE_SLUG in set(frame["subject"])
    assert config.HOUSE_SLUG in set(train.load_for(_fitted()["path"], HORIZON)["subject"])
    assert f"other_{config.HOUSE_SLUG}" in train.base_features()


def test_every_horizon_gets_a_verdict_from_one_model(tmp_path):
    """One fit, 48 gates. The gate is a property of the evaluation, and
    `horizons_shipping`/`served_by` are what a promotion is judged on."""
    path = _fitted()["path"]
    summary = train.train_all(path, tmp_path, horizons=TEST_HORIZONS, n_jobs=1)

    assert list(summary) == [str(h) for h in TEST_HORIZONS]
    assert (tmp_path / train.POOLED_NAME).exists()
    for horizon in TEST_HORIZONS:
        m = summary[str(horizon)]
        assert m["horizon_h"] == horizon
        assert isinstance(m["ships"], bool)
        # Either a family won it, or nothing is published and none may claim it.
        assert (m["kind"] in ("dedicated", "pooled")) == m["ships"]


def test_a_stale_artifact_is_refused_rather_than_unpickled(tmp_path):
    """The artifact SHAPE changed, so an old pickle does not degrade -- it
    raises somewhere unhelpful. Refusing it here publishes nothing for those
    horizons and says so in the log."""
    from occupancy_forecast import predict as predict_mod

    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / train.POOLED_NAME).open("wb") as fh:
        pickle.dump({"model": object(), "version": "0.2.0",
                     "metrics": {1: {"ships": True}}}, fh)
    assert predict_mod.load_models(tmp_path) == {}


# ---------------------------------------------------------------------------
# The three-way gate
#
# MEASURED on 173 days of real history, the two families cross at h=16: a
# dedicated fit wins h=1..15 (+71% at h=1) and the pooled fit wins h=16..48
# (-18% at h=48). The crossover is deliberately NOT hardcoded -- it belongs to
# this household at this much history -- so what is pinned here is that the
# gate picks by measurement, not that it picks any particular horizon.
# ---------------------------------------------------------------------------

def _metric(kind: str, brier: float, ships: bool, baseline_brier: float = 0.20):
    return train.Metrics(
        horizon_h=HORIZON, evaluation=train.EVALUATION, kind=kind,
        n_folds=10, n_scored=100, n_train_final=100, base_rate=0.5,
        brier=brier, log_loss=0.0, auc=0.5, mae_frac=0.0,
        brier_fold_min=brier, brier_fold_max=brier,
        best_baseline="persistence", best_baseline_brier=baseline_brier,
        skill_vs_best_baseline_pct=100.0 * (1 - brier / baseline_brier),
        folds_beating_best_baseline=8, sign_test_p=0.1, ships=ships)


def test_the_gate_picks_the_lower_brier_of_two_shipping_families():
    winner = train.choose(_metric("dedicated", 0.10, True),
                          _metric("pooled", 0.12, True))
    assert winner.kind == "dedicated"
    assert winner.brier == 0.10
    # The loser's number travels, so the crossover is visible on the Data tab.
    assert winner.rival_brier == 0.12
    assert winner.rival_kind == "pooled"

    other = train.choose(_metric("dedicated", 0.15, True),
                         _metric("pooled", 0.11, True))
    assert other.kind == "pooled"


def test_a_family_that_beats_its_rival_but_loses_to_the_baseline_does_not_ship():
    """The bar against the LADDER is absolute and comes first.

    Winning the head-to-head is not winning. Serving a model that loses to
    persistence would be strictly worse than serving persistence, which is the
    whole reason the gate exists.
    """
    winner = train.choose(_metric("dedicated", 0.19, False),
                          _metric("pooled", 0.21, False))
    assert not winner.ships
    assert winner.kind is None, "nothing is published, so no family may claim it"


def test_one_family_missing_is_not_a_failure():
    """A pooled fit that raised must not cost the dedicated verdicts, or the
    other way round."""
    assert train.choose(_metric("dedicated", 0.10, True), None).kind == "dedicated"
    assert train.choose(None, _metric("pooled", 0.10, True)).kind == "pooled"


def test_the_ladder_slice_carries_everything_the_rungs_read():
    """`train_all` fans the ladder out and hands each worker only
    `baseline.columns_for(h)`, so the slice has to reproduce the full table's
    answer exactly. A rung that grows a new column and is not named there would
    otherwise score on a frame missing it -- loudly if it KeyErrors, silently if
    the column merely goes NaN."""
    wide, windows = _fitted()["wide"], _fitted()["windows"]
    for horizon in TEST_HORIZONS:
        whole = baseline.run(wide, horizon, windows=windows)
        sliced = baseline.run(wide[baseline.columns_for(horizon)], horizon,
                              windows=windows)
        assert set(whole) == set(sliced)
        for rung, stats in whole.items():
            assert stats["brier"] == pytest.approx(sliced[rung]["brier"],
                                                   nan_ok=True), rung


def test_both_families_are_cut_on_the_same_windows():
    """`ships` walks the model's per-fold list positionally against the
    ladder's, so three candidates scored on three different fold sets would
    compare fold 7 against fold 8 and nobody would see it."""
    windows = _fitted()["windows"]
    assert windows == sorted(windows)
    for m in (_fitted()["dedicated"], _fitted()["pooled"]):
        assert m.n_folds == len(windows)
        assert len(m.per_fold) == len(windows)


def test_a_mixed_models_dict_serves_both_families(tmp_path):
    """`load_models` hides the split, and `_model_curve` answers for both."""
    from occupancy_forecast import predict as predict_mod

    path = _fitted()["path"]
    train.train_all(path, tmp_path, horizons=TEST_HORIZONS, n_jobs=1)
    models = predict_mod.load_models(tmp_path)
    assert models, "nothing loaded"
    assert set(models) <= set(TEST_HORIZONS)
    for artifact in models.values():
        # None is a normal verdict: it means both families lost to the ladder
        # and nothing is published. Tying the assertion to `ships` rather than to
        # the family keeps this test about the mixture instead of about which
        # horizons happen to clear the gate on a synthetic table.
        assert artifact["kind"] in ("dedicated", "pooled", None)
        assert bool(artifact["metrics"]["ships"]) == (artifact["kind"] is not None)

    row = train.read_wide(path).iloc[-1]
    curve = predict_mod._model_curve(models, row)
    shipping = {h for h, a in models.items() if a["metrics"]["ships"]}
    assert set(curve) == shipping, "a shipping horizon produced no value"
    assert all(0.0 <= v <= 1.0 for v in curve.values())


def test_one_call_answers_for_a_dedicated_and_a_pooled_horizon(tmp_path):
    """The mixture itself, which the test above cannot pin: on synthetic data
    every horizon may come out the same family, and an all-pooled result would
    satisfy it. Here the two artifacts are built by hand, so `_model_curve` is
    made to take both branches in one call whatever the fit decided."""
    from occupancy_forecast import predict as predict_mod

    path = _fitted()["path"]
    train.train_all(path, tmp_path, horizons=TEST_HORIZONS, n_jobs=1)
    models = predict_mod.load_models(tmp_path)
    # Off disk rather than out of `models`, which names a pooled horizon only
    # if the gate happened to give the pooled fit one.
    pooled = predict_mod._load_artifact(tmp_path / train.POOLED_NAME)
    assert pooled is not None, "the pooled fit produced no artifact"

    one, two = sorted(models)[0], sorted(models)[-1]
    mixed = {
        one: {**models[one], "kind": "dedicated",
              "model": train.fit_dedicated(train._dedicated_estimator(),
                                           train.load_for(path, one), one),
              "metrics": {**models[one]["metrics"], "ships": True}},
        two: {**models[two], "kind": "pooled", "model": pooled["model"],
              "metrics": {**models[two]["metrics"], "ships": True}},
    }
    curve = predict_mod._model_curve(mixed, train.read_wide(path).iloc[-1])
    assert set(curve) == {one, two}, "one family answered and the other did not"
    assert all(0.0 <= v <= 1.0 for v in curve.values())


def test_a_worker_is_handed_the_feature_switch_rather_than_inheriting_it():
    """`features.SHIPPED_EXTRAS` decides what a fit reads, and a loky worker is
    a fresh interpreter that inherits no module global.

    Found by the bug it causes: a probe set the switch in the parent, the parent
    reported the wider feature list, every worker fitted the narrow one, and two
    arms came back byte-identical -- which reads as an honest null on the
    candidate rather than as a broken harness. Same shape as the
    `config.configure` hazard `_pooled_fold` already documents, which is why the
    comment there names both.
    """
    before = features.SHIPPED_EXTRAS
    try:
        stamp = pd.Timestamp("2026-01-01", tz="UTC")
        frame = pd.DataFrame({
            "time": [stamp],
            features.HORIZON_COLUMN: [1.0],
            features.TARGET_COLUMN: [1.0],
        })
        # No rows before `start`, so it returns early -- the switch must already
        # have been applied by then, or a real fold would fit the wrong list.
        assert train._pooled_fold(frame, 0, stamp, stamp + pd.Timedelta(hours=1),
                                  None, ("int_calendar",)) is None
        assert features.SHIPPED_EXTRAS == ("int_calendar",)
    finally:
        features.SHIPPED_EXTRAS = before


def test_the_switch_changes_what_the_model_is_fed_and_nothing_else():
    """Built is not served: the melt always mints the candidate columns so they
    can be measured, and only `base_features` decides whether they are fed."""
    before = features.SHIPPED_EXTRAS
    try:
        features.SHIPPED_EXTRAS = ()
        narrow = set(train.base_features())
        features.SHIPPED_EXTRAS = ("int_calendar",)
        wide = set(train.base_features())
        added = wide - narrow
        assert added, "the switch fed the model nothing"
        assert added == {*features.INTEGER_CALENDAR_COLUMNS,
                         *(f"tgt_{n}" for n in features.INTEGER_CALENDAR_COLUMNS)}
        # The table's shape does not move with the switch, only the diet.
        assert set(features.long_columns()) <= set(features.long_shipped_columns())
    finally:
        features.SHIPPED_EXTRAS = before
