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
    """`train_all` hands each ladder worker only `baseline.columns_for(h)`, so a
    rung reading a column not named there scores on a frame missing it --
    silently, if the column merely goes NaN."""
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
