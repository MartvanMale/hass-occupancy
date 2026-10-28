"""Training and evaluation tests, on a synthetic table with planted structure.

Sized off `train.origin_features()`, so the table cannot drift from the feature
list; the signal is planted, so the model is asked to recover something we put
there and the assertions are not tautological.
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

# The pooled fit is ONE model over every horizon, so a test pays for each one
# it trains: short, mid, the far end and the pair either side of the lag gate.
TEST_HORIZONS = (1, HORIZON, 24, 25, 48)


def _feature_table(days: int = 80) -> Path:
    """A synthetic occupancy table with a real daily rhythm in it. 80 days is
    five full-geometry folds; the verdicts do not move at 160, the cost triples."""
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

    # Backstop so the table cannot fall behind the feature list: any ORIGIN
    # column not built above gets noise. Only the origin block is hand-built.
    for column in train.origin_features():
        if column not in table.columns and column != "subject":
            table[column] = rng.normal(0, 1, len(table))

    path = Path(tempfile.mkdtemp()) / "features.parquet"
    features.write(table.sort_values(["subject", "time"]).reset_index(drop=True), path)
    return path


# Built lazily so importing the module does not do five folds of real work, and
# so the conftest fixture has configured an installation first.
_CACHE: dict = {}


def _fitted():
    """Both families on shared windows, plus the gate's verdicts. Trained once
    for the whole module; every test below reads the same run."""
    if not _CACHE:
        path = _feature_table()
        windows, geometry = train.shared_windows(path)
        wide = train.read_wide(path)
        rungs = {h: baseline.run(wide, h, geometry=geometry, windows=windows)
                 for h in TEST_HORIZONS}

        ded_est, ded_scored, ded_rows = train.train_dedicated(path, HORIZON, windows)
        pool_est, pool_scored, pool_rows = train.train_pooled(
            path, windows, scratch=path.parent, horizons=TEST_HORIZONS, n_jobs=1)

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
    """A row just before a test window has its target inside that window. The
    table is sorted by (subject, time) across subjects, so a row-count embargo
    would protect nothing: the gap must be measured in time."""
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
    """MAE cannot tell a confident wrong guess from a calibrated one; Brier can.
    Pinned so nobody "simplifies" the metric back to MAE."""
    y = np.array([1.0, 0.0])
    confident = evaluate.score(y, np.array([1.0, 1.0]))   # right once, wrong once
    calibrated = evaluate.score(y, np.array([0.5, 0.5]))
    assert confident.mae_frac == calibrated.mae_frac      # MAE cannot separate them
    assert confident.brier > calibrated.brier             # Brier can


def test_sign_test_knows_eight_folds_prove_little():
    """Why a test window is a week, not a fortnight. See evaluate.TEST_DAYS."""
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
    """The rule is "was it proven a minority", not "did it prove a majority": a
    strict majority refuses models the sign test cannot tell from noise. See
    `train.fold_record_allows`."""
    # Proven worse than a coin flip: refused (the one-good-fortnight case).
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
    """The model must learn a delta off state_now, not the level: at multiples
    of 24 h the baseline is essentially the identity on state_now, and a tree
    approximates identity badly."""
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
# `metrics.json` is the only record of a train that survives a restart.
# ---------------------------------------------------------------------------

def test_the_summary_records_when_and_how_long(tmp_path):
    train.write_summary({"1": {"ships": True}}, tmp_path, duration_s=12.5)
    summary = train.last_summary(tmp_path)
    assert summary is not None
    assert summary["duration_s"] == 12.5
    assert summary["trained_at"]


def test_the_duration_can_be_stamped_after_the_fact(tmp_path):
    """The number is not knowable when the summary is written: fitting the models
    is only the middle of the job."""
    train.write_summary({"1": {"ships": True}}, tmp_path)
    assert train.last_summary(tmp_path)["duration_s"] is None

    train.stamp_run(251.44, tmp_path)
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
# dedicated fit, `to_long` melts every horizon for the pooled one.
# ---------------------------------------------------------------------------

def test_a_table_behind_the_feature_list_still_explains_itself(tmp_path):
    """The friendly error has to survive reading the schema instead of the table."""
    path = tmp_path / "thin.parquet"
    pd.DataFrame({"time": pd.to_datetime(["2026-01-01"], utc=True),
                  "subject": ["alice"]}).to_parquet(path)
    with pytest.raises(ValueError, match="feature list has moved ahead"):
        train.read_wide(path)


def test_the_house_is_a_training_subject_in_both_families():
    """Not a combiner over the people's forecasts, which lost to its own
    baselines."""
    frame = train.to_long(train.read_wide(_fitted()["path"]))
    assert config.HOUSE_SLUG in set(frame["subject"])
    assert config.HOUSE_SLUG in set(train.load_for(_fitted()["path"], HORIZON)["subject"])
    assert f"other_{config.HOUSE_SLUG}" in train.base_features()


def _trained() -> tuple[Path, dict]:
    """One `train_all` for every test that only reads what it wrote. Lazy like
    `_fitted`, not a module fixture: those run before conftest configures."""
    if "trained" not in _CACHE:
        models_dir = Path(tempfile.mkdtemp())
        summary = train.train_all(_fitted()["path"], models_dir,
                                  horizons=TEST_HORIZONS, n_jobs=1)
        _CACHE["trained"] = (models_dir, summary)
    return _CACHE["trained"]


def test_every_horizon_gets_a_verdict_from_one_model():
    """One fit, 48 gates. The gate is a property of the evaluation, and
    `horizons_shipping`/`served_by` are what a promotion is judged on."""
    models_dir, summary = _trained()

    assert list(summary) == [str(h) for h in TEST_HORIZONS]
    assert (models_dir / train.POOLED_NAME).exists()
    for horizon in TEST_HORIZONS:
        m = summary[str(horizon)]
        assert m["horizon_h"] == horizon
        assert isinstance(m["ships"], bool)
        # Either a family won it, or nothing is published and none may claim it.
        assert (m["kind"] in ("dedicated", "pooled")) == m["ships"]


def test_a_dedicated_artifact_from_an_earlier_train_does_not_survive_a_failed_horizon(
        tmp_path, monkeypatch):
    """Workers write a dedicated pickle only on success, so a horizon that
    failed THIS train must lose last train's: it would pass the version check
    and serve while metrics.json says it has no candidate."""
    path = _fitted()["path"]
    horizons = (1, HORIZON)          # one to fail, one to survive it
    failing = horizons[0]
    stale = tmp_path / train.DEDICATED_NAME.format(horizon=failing)
    tmp_path.mkdir(parents=True, exist_ok=True)
    with stale.open("wb") as fh:
        pickle.dump({"model": object(), "version": train.MODEL_VERSION, "kind": "dedicated",
                     "sklearn": train.sklearn.__version__,
                     "metrics": {failing: {"ships": True, "kind": "dedicated"}},
                     "features": []}, fh)

    real = train.train_dedicated

    def flaky(path_, horizon, windows):
        if horizon == failing:
            raise RuntimeError("simulated: this horizon did not fit")
        return real(path_, horizon, windows)
    monkeypatch.setattr(train, "train_dedicated", flaky)

    summary = train.train_all(path, tmp_path, horizons=horizons, n_jobs=1)
    assert not stale.exists(), "last train's pickle for the failed horizon is gone"
    assert f"{failing}h dedicated" in train.last_summary(tmp_path)["failed"]
    # The others were written by this run and rewritten with their verdicts.
    for horizon in horizons[1:]:
        assert (tmp_path / train.DEDICATED_NAME.format(horizon=horizon)).exists()
        assert str(horizon) in summary


def test_the_served_extras_are_nan_allowed_for_the_dedicated_family_too(monkeypatch):
    """`features_for` names the served extras, so `nan_allowed_for` must too, or
    `load_for` requires them and drops the dedicated arm's warm-up rows. Patched
    non-empty, because an empty `SHIPPED_EXTRAS` hides the gap."""
    horizon = HORIZON
    baseline_required = {c for c in train.features_for(horizon)
                         if c not in train.nan_allowed_for(horizon)}
    monkeypatch.setattr(features, "SHIPPED_EXTRAS", ("wclim_wide", "wclim_slope"))
    with_extras = {c for c in train.features_for(horizon)
                   if c not in train.nan_allowed_for(horizon)}
    assert with_extras == baseline_required, \
        f"an extra became required: {sorted(with_extras - baseline_required)}"
    assert set(features.extra_target_columns(horizon)) <= train.nan_allowed_for(horizon)


def test_an_empty_fold_is_not_a_lost_fold():
    """`_scores_by_fold` pads an empty fold with NaN to keep the positional walk
    aligned; the sign test and the fold record must not count it as a loss."""
    rungs = {"persistence": {"brier": 0.25,
                             "per_fold": [{"brier": 0.25}] * 4}}
    scored = pd.DataFrame({
        "subject": ["alice"] * 30,
        "time": pd.date_range("2026-01-01", periods=30, freq="30min", tz="UTC"),
        "fold": [0] * 10 + [1] * 10 + [2] * 10,            # fold 3 never scored
        f"y_{HORIZON}h": ([1.0, 0.0] * 15),
        "p": ([0.9, 0.1] * 15),
    })
    wide = pd.DataFrame({f"y_{HORIZON}h": [1.0, 0.0] * 15,
                         train.RESIDUAL_BASE: [0.9, 0.1] * 15})
    metrics = train._candidate(HORIZON, "dedicated", scored, f"y_{HORIZON}h",
                               4, 30, rungs, wide)
    assert len(metrics.per_fold) == 4, "the padded entry is still there for the walk"
    assert metrics.n_folds == 3, "but only the scored folds are trials"
    assert metrics.folds_beating_best_baseline == 3
    assert metrics.sign_test_p == pytest.approx(evaluate.sign_test(3, 3))


def test_the_ladder_is_scored_on_the_rows_the_model_is_scored_on():
    """The model drops any row missing a required origin feature; passing
    `required` makes the ladder drop the same rows, so denominators match."""
    wide, windows = _fitted()["wide"], _fitted()["windows"]
    holed = wide.copy()
    holed.loc[holed.index[::7], "coverage"] = np.nan     # coverage is required
    whole = baseline.run(holed, HORIZON, windows=windows)
    same_rows = baseline.run(holed, HORIZON, windows=windows,
                             required=train.required_origin_columns())
    assert "coverage" in train.required_origin_columns()
    assert same_rows["persistence"]["n"] < whole["persistence"]["n"]
    assert same_rows["persistence"]["n"] == int(
        holed.dropna(subset=[f"y_{HORIZON}h", *train.required_origin_columns()])
        .pipe(lambda f: sum(((f["time"] >= s) & (f["time"] < e)).sum() for s, e in windows)))


def test_a_stale_artifact_is_refused_rather_than_unpickled(tmp_path):
    """An old-shape pickle does not degrade, it raises somewhere unhelpful.
    Refused here, its horizons publish nothing and the log says so."""
    from occupancy_forecast import predict as predict_mod

    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / train.POOLED_NAME).open("wb") as fh:
        pickle.dump({"model": object(), "version": "0.2.0",
                     "metrics": {1: {"ships": True}}}, fh)
    assert predict_mod.load_models(tmp_path) == {}


def test_an_artifact_pickled_by_another_scikit_learn_is_stale(tmp_path):
    """Counted stale, not just skipped: that is what makes the worker retrain
    at once after an upgrade instead of at the next scheduled train."""
    from occupancy_forecast import predict as predict_mod

    def write(sklearn_version):
        with (tmp_path / train.POOLED_NAME).open("wb") as fh:
            pickle.dump({"model": object(), "version": train.MODEL_VERSION,
                         "sklearn": sklearn_version, "kind": "pooled",
                         "metrics": {1: {"ships": True, "kind": "pooled"}}}, fh)

    write("1.7.2")
    assert predict_mod.load_models(tmp_path) == {}
    assert predict_mod.stale_artifacts() == [(train.POOLED_NAME, "scikit-learn 1.7.2")]
    write(train.sklearn.__version__)
    assert 1 in predict_mod.load_models(tmp_path)
    assert predict_mod.stale_artifacts() == []


# ---------------------------------------------------------------------------
# The three-way gate
#
# The crossover between the families is deliberately NOT hardcoded: it belongs
# to a household at a given amount of history, so the gate picks by measurement.
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
    """The bar against the LADDER is absolute and comes first: winning the
    head-to-head is not winning."""
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
    """A ladder worker reads `train.ladder_columns(h)` and nothing else, so a
    rung reaching for a column not named there scores on a frame missing it,
    silently if the column merely goes NaN."""
    wide, windows = _fitted()["wide"], _fitted()["windows"]
    for horizon in TEST_HORIZONS:
        assert set(baseline.columns_for(horizon)) <= set(
            train.ladder_columns(horizon)), "the worker reads a narrower frame"
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


def test_a_mixed_models_dict_serves_both_families():
    """`load_models` hides the split, and `_model_curve` answers for both."""
    from occupancy_forecast import predict as predict_mod

    path = _fitted()["path"]
    models = predict_mod.load_models(_trained()[0])
    assert models, "nothing loaded"
    assert set(models) <= set(TEST_HORIZONS)
    for artifact in models.values():
        # None is a normal verdict: both families lost to the ladder. Tied to
        # `ships`, so this is about the mixture, not which horizons clear.
        assert artifact["kind"] in ("dedicated", "pooled", None)
        assert bool(artifact["metrics"]["ships"]) == (artifact["kind"] is not None)

    row = train.read_wide(path).iloc[-1]
    curve = predict_mod._model_curve(models, row)
    shipping = {h for h, a in models.items() if a["metrics"]["ships"]}
    assert set(curve) == shipping, "a shipping horizon produced no value"
    assert all(0.0 <= v <= 1.0 for v in curve.values())


def test_one_call_answers_for_a_dedicated_and_a_pooled_horizon():
    """The mixture itself: synthetic data may give every horizon one family, so
    the two artifacts are built by hand to make `_model_curve` take both
    branches in one call."""
    from occupancy_forecast import predict as predict_mod

    path, models_dir = _fitted()["path"], _trained()[0]
    models = predict_mod.load_models(models_dir)
    # Off disk rather than out of `models`, which names a pooled horizon only
    # if the gate happened to give the pooled fit one.
    pooled = predict_mod._load_artifact(models_dir / train.POOLED_NAME)
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
    a fresh interpreter that inherits no module global. Same hazard as the
    `config.configure` one `_pooled_fold` documents."""
    before = features.SHIPPED_EXTRAS
    try:
        features.SHIPPED_EXTRAS = ("int_calendar",)
        wide = train.read_wide(_feature_table(days=12))
        long_path = Path(tempfile.mkdtemp()) / train.LONG_NAME
        train.write_long(wide, long_path, horizons=(1,))
        features.SHIPPED_EXTRAS = before

        stamp = pd.Timestamp("2026-01-01", tz="UTC")
        # No rows before `start`, so it returns early -- the switch must already
        # have been applied by then, or a real fold would fit the wrong list.
        part, _peak = train._pooled_fold(long_path, 0, stamp,
                                         stamp + pd.Timedelta(hours=1),
                                         None, ("int_calendar",))
        assert part is None
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


@pytest.mark.parametrize("days, blank", [
    (25, ()),
    (80, ("other_bob",)),
    (80, features.PROXIMITY_COLUMNS),
], ids=["young archive", "person with no history", "no distance entity"])
def test_a_feature_with_no_values_does_not_stop_training(days, blank):
    """#23: scikit-learn 1.9 raised on a column blank across a fit's rows, and one
    such fold stopped every horizon in both families."""
    wide = train.read_wide(_feature_table(days=days))
    for column in blank:
        wide[column] = np.nan
    path = Path(tempfile.mkdtemp()) / "features.parquet"
    features.write(wide, path)
    windows, _ = train.shared_windows(wide)

    train.train_pooled(wide, windows, scratch=path.parent,
                       horizons=(1, HORIZON), n_jobs=1)
    train.train_dedicated(path, HORIZON, windows)


def test_a_failure_every_dedicated_horizon_shares_is_logged_once(monkeypatch, tmp_path,
                                                                 caplog):
    """A worker's own warning reached no handler, once per horizon."""
    def broken(path, horizon, windows):
        raise ValueError("the same reason")

    monkeypatch.setattr(train, "train_dedicated", broken)
    caplog.set_level("DEBUG")
    train.train_all(_fitted()["path"], tmp_path, horizons=(1, HORIZON), n_jobs=1)

    lines = [r.getMessage() for r in caplog.records
             if "dedicated horizon" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "+1h" in lines[0] and f"+{HORIZON}h" in lines[0]
    assert "the same reason" in lines[0]
    assert any("Traceback" in r.getMessage() for r in caplog.records), \
        "logged with its traceback"


def _snapshot(models_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(models_dir.iterdir())}


def _previous_train(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for artifact in _trained()[0].iterdir():
        (models_dir / artifact.name).write_bytes(artifact.read_bytes())
    return models_dir


def test_a_train_that_fails_late_leaves_the_previous_models_untouched(tmp_path,
                                                                      monkeypatch):
    """The workers write as they finish, so a train killed or failing after them
    used to leave half a new set beside the old metrics."""
    models_dir = _previous_train(tmp_path)
    before = _snapshot(models_dir)

    def broken(*args, **kwargs):
        raise RuntimeError("simulated: the gate fell over")
    monkeypatch.setattr(train, "_run_gate", broken)

    with pytest.raises(RuntimeError, match="the gate fell over"):
        train.train_all(_fitted()["path"], models_dir, horizons=(1, HORIZON), n_jobs=1)
    assert _snapshot(models_dir) == before
    assert not train._staging(models_dir).exists()


@pytest.mark.parametrize("dies_in", [0, 1], ids=["dedicated pool", "pooled pool"])
def test_a_lost_worker_stops_the_train_and_says_why(tmp_path, monkeypatch, dies_in):
    """An OOM-killed worker must not read as "pooled fit: skipped" and publish a
    set with a family missing."""
    models_dir = _previous_train(tmp_path)
    before = _snapshot(models_dir)
    real, calls = train.Parallel, []

    def parallel(*args, **kwargs):
        calls.append(None)
        if len(calls) - 1 == dies_in:
            def killed(tasks):
                raise train.TerminatedWorkerError("simulated SIGKILL(-9)")
            return killed
        return real(*args, **kwargs)
    monkeypatch.setattr(train, "Parallel", parallel)

    with pytest.raises(train.OutOfMemory, match="ran out of memory"):
        train.train_all(_fitted()["path"], models_dir, horizons=(1, HORIZON), n_jobs=1)
    assert _snapshot(models_dir) == before


def test_only_a_pool_worker_volunteers_to_the_oom_killer(tmp_path, monkeypatch):
    adj = tmp_path / "oom_score_adj"
    monkeypatch.setattr(train, "OOM_SCORE_ADJ", adj)

    assert train._as_expendable(os.getpid(), int, "7") == 7
    assert not adj.exists(), "the server itself must keep its score"
    assert train._as_expendable(os.getpid() + 1, int, "7") == 7
    assert adj.read_text() == "1000"

    monkeypatch.setattr(train, "OOM_SCORE_ADJ", tmp_path / "missing" / "adj")
    assert train._as_expendable(os.getpid() + 1, int, "7") == 7, "a denied write is ignored"


def test_a_dedicated_verdict_without_its_file_is_not_served_by_the_pooled_model(
        tmp_path, caplog):
    """What the killed 0.4.2 train left behind: the pooled model aliased under a
    dedicated verdict, failing on missing columns every cycle."""
    from occupancy_forecast import predict as predict_mod

    with (_trained()[0] / train.POOLED_NAME).open("rb") as fh:
        pooled = pickle.load(fh)
    horizon = TEST_HORIZONS[0]
    pooled["metrics"][horizon] = {**pooled["metrics"][horizon],
                                  "kind": "dedicated", "ships": True}
    with (tmp_path / train.POOLED_NAME).open("wb") as fh:
        pickle.dump(pooled, fh)

    caplog.set_level("WARNING")
    models = predict_mod.load_models(tmp_path)
    assert horizon not in models
    assert any(f"+{horizon}h" in r.getMessage() for r in caplog.records)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """Six cores, and whatever memory the test writes."""
    monkeypatch.setattr(train.os, "sched_getaffinity", lambda pid: set(range(6)),
                        raising=False)
    monkeypatch.setattr(train, "MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(train, "CGROUP", tmp_path / "cgroup")
    (tmp_path / "cgroup").mkdir()

    def write(available_gb: float, limit_gb: float | None = None, used_gb: float = 0):
        (tmp_path / "meminfo").write_text(
            f"MemTotal: 6000000 kB\nMemAvailable: {int(available_gb * 1e6)} kB\n")
        (tmp_path / "cgroup" / "memory.max").write_text(
            "max\n" if limit_gb is None else f"{int(limit_gb * 1e9)}\n")
        (tmp_path / "cgroup" / "memory.current").write_text(f"{int(used_gb * 1e9)}\n")
    return write


def _fits(free_gb: float, per_worker: float) -> int:
    """What the budget allows, spelled out: a share of free memory over the
    data a worker holds plus the interpreter it needs to hold it."""
    budget = free_gb * 1e9 * train.MEMORY_SHARE
    return int(budget // (per_worker + train.WORKER_BASELINE_BYTES))


def test_workers_are_capped_by_free_memory(machine):
    machine(available_gb=3.0)
    assert train.worker_count() == 5, "no estimate: cores less one"
    assert train.worker_count(per_worker=0.1e9) == 5, "cores stay the ceiling"
    assert train.worker_count(per_worker=0.75e9) == _fits(3.0, 0.75e9) == 2
    with pytest.raises(train.OutOfMemory, match="for one worker"):
        train.worker_count(per_worker=10e9)


def test_a_worker_is_priced_with_the_interpreter_it_needs(machine):
    """A worker costs an interpreter before it touches data, so a narrow frame
    must not read as a free one."""
    machine(available_gb=0.4)
    assert train.worker_count(per_worker=0.03e9) == 1


def test_a_cgroup_limit_counts_as_less_free_memory(machine):
    machine(available_gb=5.0, limit_gb=2.0, used_gb=0.5)
    assert train.worker_count(per_worker=0.5e9) == _fits(1.5, 0.5e9) == 1


def test_unreadable_memory_falls_back_to_cores(machine, tmp_path):
    machine(available_gb=3.0)
    (tmp_path / "meminfo").unlink()
    assert train.worker_count(per_worker=10e9) == 5


def test_the_blocked_melt_matches_the_one_frame_melt(tmp_path):
    """`write_long` melts a block of origin rows at a time, so the whole long
    table never exists in the server. Row order matters too, because
    `load_long` does not sort what it reads."""
    # Several blocks, and more than one subject per timestamp, or the block
    # boundaries and the sort keys go untested.
    wide = train.read_wide(_feature_table(days=12))
    horizons = (1, HORIZON, 24)
    path = tmp_path / train.LONG_NAME
    rows, size = train.write_long(wide, path, horizons=horizons, block_rows=64)

    columns = train.pooled_columns()
    expected = train.to_long(wide, horizons=horizons)[columns]
    pd.testing.assert_frame_equal(train.load_long(path), expected)
    assert rows == len(expected)
    assert size > 0


def test_the_long_table_is_written_in_the_order_load_long_trusts(tmp_path):
    """`load_long` does not sort, so the file has to arrive in exactly the
    order the pooled code reads. Pinned here rather than inferred from
    `long_frame`'s own sort, which is in another module."""
    wide = train.read_wide(_feature_table(days=12))
    path = tmp_path / train.LONG_NAME
    train.write_long(wide, path, horizons=(1, HORIZON), block_rows=64)

    frame = train.load_long(path)
    keys = ["time", features.HORIZON_COLUMN, "subject"]
    assert frame[keys].equals(frame[keys].sort_values(keys).reset_index(drop=True))


def test_a_melt_block_never_splits_a_timestamp(tmp_path):
    """A timestamp split across two blocks would be written out of order, and
    `load_long` does not sort."""
    wide = train.read_wide(_feature_table(days=4))
    subjects = wide["subject"].nunique()
    assert subjects > 1, "a single subject cannot show the hazard"

    times = wide["time"].to_numpy()
    for start, stop in train._melt_blocks(wide, block_rows=7):
        if stop < len(wide):
            assert times[stop - 1] != times[stop], "a timestamp spans two blocks"


def test_the_melt_is_refused_when_it_would_not_fit(machine, tmp_path):
    """The budget is asked before the melt, so a refusal costs nothing."""
    machine(available_gb=0.001)
    wide = train.read_wide(_feature_table(days=12))
    windows, _ = train.shared_windows(wide)
    with pytest.raises(train.OutOfMemory, match="previous models stay in use"):
        train.train_pooled(wide, windows, scratch=tmp_path, horizons=(1, HORIZON))
    assert not (tmp_path / train.LONG_NAME).exists()


def test_the_melt_estimate_is_the_size_of_one_horizon_times_the_count():
    wide = train.read_wide(_feature_table(days=12))
    one = train.frame_bytes(train._melt_block(wide, (1,)))
    assert train.long_bytes(wide, (1, HORIZON, 24)) == one * 3
    assert train.long_bytes(wide.iloc[:0], (1,)) == 0


def test_the_peak_resident_set_is_read_or_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "STATUS", tmp_path / "status")
    (tmp_path / "status").write_text("Name:\tpython\nVmHWM:\t  2048 kB\nVmRSS:\t 1 kB\n")
    assert train.peak_rss() == 2048 * 1024
    (tmp_path / "status").unlink()
    assert train.peak_rss() is None


def test_a_phase_keeps_its_largest_peak(monkeypatch):
    readings = iter([100, 300, 200])
    monkeypatch.setattr(train, "peak_rss", lambda: next(readings))
    phases = train.Phases()
    for _ in range(3):
        with phases("melt"):
            pass
    assert phases.peaks["melt"] == 300
    phases.note_peak("worker", None)
    assert "worker" not in phases.peaks
    phases.note_peak("worker", 1_500_000)
    assert phases.peaks_as_dict()["worker"] == 1.5


def test_the_summary_records_the_peak_separately_from_the_seconds(tmp_path):
    phases = train.Phases()
    phases.note_peak("pooled melt", 400_000_000)
    train.write_summary({}, tmp_path, phases={"pooled melt": 1.0},
                        peak_mb=phases.peaks_as_dict())
    summary = json.loads(train.summary_path(tmp_path).read_text())
    assert summary["peak_mb"] == {"pooled melt": 400.0}
    assert summary["phases"] == {"pooled melt": 1.0}


def test_publishing_leaves_the_melts_scratch_file_behind(tmp_path):
    """The staging directory doubles as the melt's scratch, so `_publish` must
    move artifacts only."""
    staging = train._staging(tmp_path / "models")
    staging.mkdir(parents=True)
    (staging / "occupancy_pooled.pkl").write_bytes(b"artifact")
    (staging / "metrics.json").write_text("{}")
    (staging / train.LONG_NAME).write_bytes(b"scratch")

    models_dir = tmp_path / "models"
    train._publish(staging, models_dir)
    assert (models_dir / "occupancy_pooled.pkl").read_bytes() == b"artifact"
    assert (models_dir / "metrics.json").exists()
    assert not (models_dir / train.LONG_NAME).exists()
    assert (staging / train.LONG_NAME).exists()


def test_an_unsorted_feature_table_is_refused_rather_than_melted(tmp_path):
    """`_melt_blocks` walks timestamps with `searchsorted`, which on unsorted
    input returns an edge that does not advance."""
    wide = train.read_wide(_feature_table(days=12))
    shuffled = wide.sort_values(["subject", "time"]).reset_index(drop=True)
    with pytest.raises(ValueError, match="not in time order"):
        train.write_long(shuffled, tmp_path / train.LONG_NAME, horizons=(1,))


def test_nothing_is_copied_when_every_column_was_observed():
    """The fill is a no-op on a table where every column has values, so
    `transform` hands back the frame it was given."""
    from occupancy_forecast import estimators
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.nan]})
    step = estimators.UnobservedToConstant().fit(frame)
    assert step.transform(frame) is frame

    blank = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    filled = estimators.UnobservedToConstant().fit(blank).transform(blank)
    assert filled is not blank
    assert list(filled["b"]) == [0.0, 0.0]


def test_a_phase_the_server_timed_reaches_the_summary(tmp_path):
    """`train_all` writes the summary before the server's own phases are done,
    so the server adds its keys to what the trainer already wrote."""
    phases = train.Phases()
    phases.seconds["features"] = 12.0
    phases.note_peak("features horizons", 190_000_000)
    train.write_summary({"1": {"ships": True}}, tmp_path,
                        phases={"read": 1.0}, peak_mb={"read": 700.0})

    train.stamp_run(99.0, tmp_path, phases.as_dict(), phases.peaks_as_dict())
    summary = json.loads(train.summary_path(tmp_path).read_text())
    assert summary["phases"] == {"read": 1.0, "features": 12.0}
    assert summary["peak_mb"] == {"read": 700.0, "features horizons": 190.0}
    assert summary["duration_s"] == 99.0


def test_the_resident_set_is_read_or_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "STATUS", tmp_path / "status")
    (tmp_path / "status").write_text("VmHWM:\t 4096 kB\nVmRSS:\t 2048 kB\n")
    assert train.rss() == 2048 * 1024
    assert train.peak_rss() == 4096 * 1024
    (tmp_path / "status").unlink()
    assert train.rss() is None


def test_a_declared_file_is_read_without_a_second_sort(tmp_path):
    """`build_to` writes in order and says so, and sorting it again would cost
    a second copy of the whole table."""
    from occupancy_forecast.tests.test_features import _built_source

    source, start, stop = _built_source(days=8)
    declared = tmp_path / "declared.parquet"
    features.build_to(source, declared, start, stop, target_bytes=1_000_000)
    assert train.declares_time_order(declared)

    plain = tmp_path / "plain.parquet"
    features.write(features.build(source, start, stop), plain)
    assert not train.declares_time_order(plain)

    skipped, sorted_ = train.read_wide(declared), train.read_wide(plain)
    assert skipped["time"].is_monotonic_increasing
    # The same rows either way. The subject order within one timestamp does
    # differ, because the sorted path leaves it to an unstable sort.
    keys = ["time", "subject"]
    pd.testing.assert_frame_equal(skipped.sort_values(keys).reset_index(drop=True),
                                  sorted_.sort_values(keys).reset_index(drop=True))
    assert list(skipped[keys].itertuples(index=False)) == sorted(
        skipped[keys].itertuples(index=False)), "a declared file reads back in order"


def test_a_table_out_of_time_order_is_still_sorted(tmp_path):
    """A file written before the order was declared gets the sort it needs."""
    from occupancy_forecast.tests.test_features import _built_source

    source, start, stop = _built_source(days=6)
    table = features.build(source, start, stop)
    shuffled = table.sort_values(["subject", "time"]).reset_index(drop=True)
    path = tmp_path / "subject_major.parquet"
    features.write(shuffled, path)

    assert not train.declares_time_order(path)
    assert train.read_wide(path)["time"].is_monotonic_increasing


def test_the_sample_does_not_depend_on_the_order_the_rows_arrive_in():
    """A worker decides a row group at a time, so the draw has to be a function
    of the row's key rather than its position."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "subject": rng.choice(list(config.all_slugs()), 4000),
        "time": pd.Timestamp("2026-01-01", tz="UTC")
        + pd.to_timedelta(rng.integers(0, 40000, 4000), unit="m"),
        features.HORIZON_COLUMN: rng.integers(1, 49, 4000).astype(float),
    }).drop_duplicates(subset=["subject", "time", features.HORIZON_COLUMN])

    keys = ["subject", "time", features.HORIZON_COLUMN]
    chosen = frame[train.sampled_rows(frame, 0.25, seed=3)]
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    again = shuffled[train.sampled_rows(shuffled, 0.25, seed=3)]

    assert set(map(tuple, chosen[keys].to_numpy())) == set(
        map(tuple, again[keys].to_numpy()))


def test_the_sample_keeps_about_the_rate():
    """Uniform enough that a rate means what it says, or the budget is wrong."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "subject": rng.choice(list(config.all_slugs()), 60000),
        "time": pd.Timestamp("2026-01-01", tz="UTC")
        + pd.to_timedelta(np.arange(60000), unit="min"),
        features.HORIZON_COLUMN: rng.integers(1, 49, 60000).astype(float),
    })
    for rate in (0.25, 0.125, 0.0625):
        kept = train.sampled_rows(frame, rate, seed=0).mean()
        assert abs(kept - rate) < 0.01, (rate, kept)
    assert train.sampled_rows(frame, 1.0, seed=0).all(), "a full rate keeps all"


def test_a_folds_scored_rows_are_never_sampled_away(tmp_path):
    """The gate sees all forty-eight horizons, so the window a fold scores on
    comes back whole however low the rate goes."""
    from occupancy_forecast.tests.test_features import _built_source

    source, start, stop = _built_source(days=10)
    wide = train.read_wide(_feature_table(days=10))
    long_path = tmp_path / train.LONG_NAME
    train.write_long(wide, long_path, horizons=(1, HORIZON))

    whole = train.load_long(long_path)
    times = whole["time"]
    window = (times.iloc[len(times) // 2], times.iloc[-1] + pd.Timedelta("1min"))
    in_window = int(((times >= window[0]) & (times < window[1])).sum())

    sampled = train.load_long_sampled(long_path, 0.0625, seed=0, window=window)
    scored = ((sampled["time"] >= window[0]) & (sampled["time"] < window[1])).sum()
    assert int(scored) == in_window
    assert len(sampled) < len(whole), "outside the window it did sample"


def test_the_leaf_floor_follows_the_sample_rate():
    """The floor counts person-days, so it has to move with how many rows an
    origin contributes."""
    assert train.min_samples_leaf(12 / 48) == 100
    assert train.min_samples_leaf(6 / 48) == 50
    assert train.min_samples_leaf(1 / 48) == 20, "a floor of one row is not a floor"


def test_the_sample_rate_steps_down_to_what_fits(machine):
    """A tight box trains on fewer horizons per origin rather than not at all."""
    estimate, folds = 1e9, 26
    rungs = [p / len(config.HORIZONS_H) for p in train.HORIZONS_PER_ORIGIN_RUNGS]

    machine(available_gb=8.0)
    assert train.pooled_rate(estimate, folds) == train.TRAIN_SAMPLE_RATE

    # Each rung has to be affordable at its own price and not at the one above.
    seen = []
    for free in (8.0, 1.6, 1.2, 0.8, 0.6, 0.45, 0.2):
        machine(available_gb=free)
        rate = train.pooled_rate(estimate, folds)
        assert rate is None or rate in rungs, rate
        seen.append(rate)
    assert seen[0] == train.TRAIN_SAMPLE_RATE
    assert seen[-1] is None, "nothing fits, so it refuses"
    tried = [r for r in seen if r is not None]
    assert tried == sorted(tried, reverse=True), "less memory must not buy more rows"
    assert len(set(tried)) > 2, "the rungs have to step"


def test_a_rate_that_does_not_fit_refuses_rather_than_tries(machine, tmp_path):
    """The refusal names the table's size, which is the figure the reader can
    act on."""
    machine(available_gb=0.05)
    wide = train.read_wide(_feature_table(days=12))
    windows, _ = train.shared_windows(wide)
    with pytest.raises(train.OutOfMemory, match="not enough free memory"):
        train.train_pooled(wide, windows, scratch=tmp_path, horizons=(1, HORIZON))


def test_the_sampled_read_hands_every_row_group_back(tmp_path, monkeypatch):
    """Both passes over the row groups have to release, or the read grows by
    the whole file."""
    import pyarrow.parquet as pq

    long_path = tmp_path / "long.parquet"
    rows, _ = train.write_long(_feature_table(days=20), long_path,
                               horizons=TEST_HORIZONS, block_rows=256)
    groups = pq.ParquetFile(long_path).metadata.num_row_groups
    assert groups > 1, "one row group cannot show a per-group leak"

    calls = []
    monkeypatch.setattr(features, "release_arrow", lambda: calls.append(1))
    frame = train.load_long_sampled(long_path, 1.0, 0)
    assert len(frame) == rows
    assert len(calls) == 2 * groups, "the keys pass and the fill pass, each group"


def test_the_melt_hands_its_blocks_back(tmp_path, monkeypatch):
    """The melt writes a row group per block, and every one of them releases."""
    import pyarrow.parquet as pq

    long_path = tmp_path / "long.parquet"
    calls = []
    monkeypatch.setattr(features, "release_arrow", lambda: calls.append(1))
    train.write_long(_feature_table(days=20), long_path,
                     horizons=TEST_HORIZONS, block_rows=256)
    groups = pq.ParquetFile(long_path).metadata.num_row_groups
    assert len(calls) >= groups


def test_a_fold_is_priced_with_the_window_it_scores(machine):
    """A fold holds its sample and the whole window it scores, so the price has
    to cover both."""
    assert train.worker_share(1 / 48, 26) == pytest.approx(1 / 48 + 1 / 26)
    assert train.worker_share(0.9, 2) == 1.0, "never more than the whole table"
    assert train.pooled_bytes(1e9, 1 / 48, 26) > train.pooled_bytes(1e9, 1 / 48, 1000)

    # Wide scoring windows cost rows: the same box steps down a rung for them.
    machine(available_gb=1.4)
    assert train.pooled_rate(1e9, 1000) > train.pooled_rate(1e9, 26)


def test_the_fan_out_is_priced_after_the_melt(tmp_path, monkeypatch, machine):
    """The melt spends memory the workers then have to fit into, so the price
    is read off the melted table and not off the estimate."""
    machine(available_gb=8.0)
    order = []
    real_rate, real_melt = train.pooled_rate, train.write_long

    def priced(long_bytes, n_folds):
        order.append("price")
        return real_rate(long_bytes, n_folds)

    def melt(*args, **kwargs):
        order.append("melt")
        answer = real_melt(*args, **kwargs)
        machine(available_gb=0.05)      # what the melt just cost
        return answer

    monkeypatch.setattr(train, "pooled_rate", priced)
    monkeypatch.setattr(train, "write_long", melt)
    path = _feature_table(days=20)
    windows, _ = train.shared_windows(train.read_wide(path))
    with pytest.raises(train.OutOfMemory, match="not enough free memory"):
        train.train_pooled(path, windows, scratch=tmp_path,
                           horizons=TEST_HORIZONS)
    assert order == ["price", "melt", "price"]
