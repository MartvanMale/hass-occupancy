"""How long until they are home, given where they are and which way they are going.

This answers a different question from the rest of the stack, and the split
matters. `train`/`predict` answer **"will they be home at t+h?"** -- a
probability, on an hourly grid. This module answers **"if they are on their way,
how many minutes?"** -- a duration, at minute resolution.

Both are needed and neither substitutes for the other:

  * The occupancy model cannot resolve better than its horizon grid, so the best
    it can ever say is "home within the next hour". For pre-heating that is the
    difference between a warm house and an hour of wasted gas.
  * This module is CONDITIONAL ON ARRIVING. It is trained only on samples that
    were actually followed by an arrival, so it has no opinion on whether
    somebody is coming home at all -- ask it about a person sitting at their
    desk and it will cheerfully tell you how long the drive would take. The
    probability has to come from the occupancy model.

Trained on the raw proximity series rather than the 30-minute feature table,
because quantising an ETA to half an hour throws away most of its value.

MEASURED on a two-person household with a few hundred recorded arrivals each.
The two people's usual journeys differed enough in length that they need
separate models -- a single pooled "minutes per km" was wrong for both.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config, evaluate, log

_log = log.get(__name__)

MODELS_DIR = config.MODELS_DIR

# Inside this radius of home, they have arrived. zone.home has a 50 m radius;
# 200 m allows for GPS scatter without letting a neighbouring street count.
ARRIVED_M = 200

# Only model journeys that start beyond this. Below it the answer is "minutes"
# whatever you do, and the samples are dominated by GPS jitter in the driveway.
MIN_JOURNEY_KM = 1.0

# How far ahead an arrival may be and still be attributed to the current
# position. Beyond this the link is not a journey, it is a coincidence -- and
# training on it teaches the model that being at work predicts arriving in
# four hours, which is a calendar fact, not a travel time.
MAX_LEAD_MIN = 180

# Window used to estimate closing speed from the distance trace.
SPEED_WINDOW_MIN = 15

# Below this closing speed, no ETA is served at all.
#
# **This enforces at SERVING TIME the condition `MAX_LEAD_MIN` imposes on
# TRAINING, and without it the sensor is confidently wrong most of the day.**
# Training only ever sees moments within 180 minutes of an arrival, so the model
# cannot represent a longer wait -- and nothing in the features distinguishes
# "stationary at 32 km at 17:50" from "stationary at 32 km at 12:10", which are
# two hundred minutes apart. Asked the second, it answered 169 minutes: the top
# of its range, for somebody who had not left her desk.
#
# MEASURED against uncensored truth (minutes to the real next arrival, however
# far off) over ~19,500 moments for one person and ~7,600 for the other:
#
#   stationary or moving away   13-27% are truly within 180 min   median 585 min
#   closing > 1 km/h            38-51%
#   closing > 5 km/h            57-58%                            median 30-58 min
#
# 5 km/h rather than any positive value because the fraction plateaus there and
# because it is the line between walking about at work and actually travelling.
# The cost is real -- the sensor is now silent for most of the day -- but it was
# wrong for most of that time, and where it does answer it is very good indeed:
# MAE 4.3 and 5.4 minutes while closing, against 11-15 while stationary.
#
# The 15-minute window this is measured over is what keeps it steady: a stop at
# a traffic light does not zero a quarter-hour of approach.
MIN_CLOSING_KMH = 5.0

FEATURES = ["distance_km", "closing_kmh", "dir_towards", "dir_away", "hour", "dow"]

# ETA is per PERSON, never for `house`. A house does not travel; its arrival is
# whichever person gets back first, so serving derives it as the min of the
# people rather than fitting a third model on a mixture of commutes that may be
# very different lengths.
def eta_subjects() -> tuple[str, ...]:
    """People with any distance signal at all -- real or synthesised.

    Never the house: a house does not travel, and its arrival is whichever
    person gets back first, which serving computes as a min() rather than a
    third model over a mixture of different commutes.
    """
    return tuple(p.slug for p in config.PEOPLE)

# Fold geometry, deliberately looser than evaluate.TEST_DAYS.
#
# These samples are journeys, not slots: a person who travels less can yield
# only ~1200 usable rows across a whole history, so the occupancy folds' 200-row
# minimum produced ZERO folds for them. 14-day windows with a 50-row floor
# give ~12.
# Lower floors mean noisier per-fold numbers, which is why the ship gate below
# also demands a real effect size and not just a majority.
FOLD_DAYS = 14
FOLD_MIN_ROWS = 50

# Minimum improvement over constant-speed for the model to be served. An ETA
# that is 5% better than "distance divided by average speed" does not change a
# pre-heating decision and is not worth the moving parts.
MIN_SKILL_PCT = 15.0


@dataclass
class EtaMetrics:
    subject: str
    n_samples: int
    n_arrivals: int
    n_folds: int
    mae_min: float
    median_ae_min: float
    p90_ae_min: float
    baseline_mae_min: float
    skill_pct: float
    folds_beating_baseline: int
    sign_test_p: float
    ships: bool
    implied_kmh: float
    per_fold: list = field(default_factory=list)


def _distance_entity(subject: str) -> tuple[str | None, str | None]:
    """(distance, direction) entity ids for one person, real or synthesised.

    Never raises for an unconfigured person -- it used to be a bare dict lookup
    and threw KeyError, which callers then had to catch by exception type rather
    than by asking.
    """
    from .discover import synthetic_distance_entity
    try:
        person = config.subject(subject)
    except KeyError:
        return None, None
    distance = person.distance_entity or (
        synthetic_distance_entity(person.slug) if person.is_person else None)
    return distance, person.direction_entity


def traces(source, subject: str, start: str, stop: str | None = None) -> pd.DataFrame:
    """Raw distance and direction for one subject, on the union of their events.

    No resampling: the proximity sensor already fires every 0.5-3 minutes while
    somebody is moving, and it is the moving samples this module cares about.
    """
    distance_entity, direction_entity = _distance_entity(subject)
    if distance_entity is None:
        return pd.DataFrame(columns=["distance_km", "direction"])

    dist = source.numeric(distance_entity, start, stop)
    if not dist:
        return pd.DataFrame(columns=["distance_km", "direction"])
    frame = pd.DataFrame(
        {"distance_km": [v / 1000.0 for _, v in dist]},
        index=pd.to_datetime([t for t, _ in dist], utc=True, format="ISO8601"),
    ).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    direction = (source.seeded_states(direction_entity, start, stop)
                 if direction_entity else [])
    if direction:
        d = pd.Series([v for _, v in direction],
                      index=pd.to_datetime([t for t, _ in direction], utc=True,
                                           format="ISO8601")).sort_index()
        d = d[~d.index.duplicated(keep="last")]
        frame["direction"] = d.reindex(frame.index.union(d.index)).ffill().reindex(frame.index)
    else:
        frame["direction"] = None
    return frame


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The model's inputs for every observation. No target, no filtering.

    Shared by training and serving on purpose: if the two computed
    `closing_kmh` even slightly differently, the served number would be drawn
    from a distribution the model never saw, and nothing would say so.
    """
    if frame.empty:
        return pd.DataFrame(columns=FEATURES)

    # Everything below works in epoch nanoseconds. Mixing a tz-aware index with
    # the tz-naive datetime64 that .values and .rolling() hand back is a
    # TypeError waiting to happen, and the arithmetic is clearer this way.
    stamp = frame.index.asi8.astype("float64")

    # Closing speed: km lost per hour over the trailing window. Positive means
    # getting closer, which a raw distance column cannot express.
    seconds = stamp / 1e9
    km = frame["distance_km"].to_numpy()
    left = np.searchsorted(seconds, seconds - SPEED_WINDOW_MIN * 60, side="left")
    elapsed_h = (seconds - seconds[left]) / 3600
    # np.where evaluates BOTH branches, so the denominator has to be safe even
    # where the mask discards it.
    safe_h = np.where(elapsed_h > 0.02, elapsed_h, np.nan)
    closing = np.nan_to_num((km[left] - km) / safe_h, nan=0.0, posinf=0.0, neginf=0.0)

    local = frame.index.tz_convert(config.TIMEZONE)
    out = pd.DataFrame({
        "distance_km": km,
        "closing_kmh": np.clip(closing, -200, 200),
        "dir_towards": (frame["direction"] == "towards").astype(float).to_numpy(),
        "dir_away": (frame["direction"] == "away_from").astype(float).to_numpy(),
        "hour": local.hour + local.minute / 60,
        "dow": local.dayofweek,
    }, index=frame.index)
    return out


def build_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Training rows: features plus minutes-until-the-next-arrival.

    Rows with no arrival inside `MAX_LEAD_MIN` are dropped -- which is exactly
    the conditional-on-arriving caveat in the module docstring, and the reason
    this cannot be read as "are they coming home".
    """
    out = feature_frame(frame)
    if out.empty:
        return out

    home = (frame["distance_km"] * 1000 < ARRIVED_M).to_numpy()
    stamp = frame.index.asi8.astype("float64")
    # Nanosecond stamp of the next arrival at or after each row.
    next_arrival = pd.Series(np.where(home, stamp, np.nan)).bfill().to_numpy()
    out["minutes_to_home"] = (next_arrival - stamp) / 1e9 / 60

    keep = (
        out["minutes_to_home"].notna()
        & (out["minutes_to_home"] > 0)
        & (out["minutes_to_home"] <= MAX_LEAD_MIN)
        & (out["distance_km"] >= MIN_JOURNEY_KM)
    )
    return out[keep]


def current_row(source, subject: str, lookback_h: int = 6) -> pd.DataFrame | None:
    """The newest feature row for one subject, or None if there is nothing usable.

    None in two cases, and both mean "the model was never trained on this".

    **Already home** (inside `MIN_JOURNEY_KM`): "how long until you get home"
    has no meaning.

    **Not travelling** (closing slower than `MIN_CLOSING_KMH`): the model is
    conditional on being ON a journey home, and `MAX_LEAD_MIN` enforces that in
    training by discarding anything more than three hours out. Serving without
    the same condition asks it a question it has never seen and gets an answer
    near the top of its range -- see the constant for the measurement.
    """
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_h))
    frame = traces(source, subject, start.strftime("%Y-%m-%dT%H:%M:%SZ"))
    rows = feature_frame(frame)
    if rows.empty:
        return None
    last = rows.iloc[[-1]]
    if float(last["distance_km"].iloc[0]) < MIN_JOURNEY_KM:
        return None
    if float(last["closing_kmh"].iloc[0]) < MIN_CLOSING_KMH:
        return None
    return last


def _estimator() -> HistGradientBoostingRegressor:
    """Small, and fitted on log-minutes.

    Log because the target spans 1 to 180 minutes and the errors that matter are
    proportional -- being ten minutes out on a two-hour drive is fine, on a
    twelve-minute one it is the whole answer. Squared error on raw minutes would
    optimise almost entirely for the long journeys.
    """
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0,
    )


def _fit(frame: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = _estimator()
    model.fit(frame[FEATURES], np.log1p(frame["minutes_to_home"]))
    return model


def predict_minutes(model, rows: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(model.predict(rows[FEATURES])), 0.0, MAX_LEAD_MIN)


def train_one(source, subject: str, start: str | None = None,
              stop: str | None = None) -> tuple[object, EtaMetrics]:
    from .features import history_start
    frame = traces(source, subject, start or history_start(source), stop)
    samples = build_samples(frame).sort_index()
    if len(samples) < 500:
        raise ValueError(f"{subject}: only {len(samples)} usable samples")

    times = pd.Series(samples.index, index=range(len(samples)))
    folds = evaluate.calendar_folds(times, embargo=pd.Timedelta(hours=6),
                                    test_days=FOLD_DAYS, min_test_rows=FOLD_MIN_ROWS)
    if not folds:
        raise ValueError(f"{subject}: no folds from {len(samples)} samples")

    # Baseline: constant speed, the implied km/h of the training half. This is
    # the "distance divided by how fast we usually get home" answer, and the
    # model has to beat it or it is not earning its keep.
    fold_mae, base_mae, per_fold = [], [], []
    for fold in folds:
        tr, te = samples.iloc[fold.train_idx], samples.iloc[fold.test_idx]
        kmh = (tr["distance_km"] / (tr["minutes_to_home"] / 60)).median()
        model = _fit(tr)
        pred = predict_minutes(model, te)
        base = np.clip(te["distance_km"] / kmh * 60, 0, MAX_LEAD_MIN)
        truth = te["minutes_to_home"].to_numpy()
        fold_mae.append(float(np.mean(np.abs(pred - truth))))
        base_mae.append(float(np.mean(np.abs(base - truth))))
        per_fold.append({"n": int(len(te)), "mae_min": fold_mae[-1],
                         "baseline_mae_min": base_mae[-1]})

    mae, bmae = float(np.mean(fold_mae)), float(np.mean(base_mae))
    beat = sum(1 for a, b in zip(fold_mae, base_mae) if a < b)
    p = evaluate.sign_test(beat, len(folds))

    model = _fit(samples)
    pred_all = predict_minutes(model, samples)
    err = np.abs(pred_all - samples["minutes_to_home"].to_numpy())
    home = (frame["distance_km"] * 1000 < ARRIVED_M)

    metrics = EtaMetrics(
        subject=subject,
        n_samples=int(len(samples)),
        n_arrivals=int((home & ~home.shift(1, fill_value=False)).sum()),
        n_folds=len(folds),
        mae_min=mae,
        median_ae_min=float(np.median(err)),
        p90_ae_min=float(np.percentile(err, 90)),
        baseline_mae_min=bmae,
        skill_pct=100.0 * (1.0 - mae / bmae) if bmae else float("nan"),
        folds_beating_baseline=beat,
        sign_test_p=p,
        ships=bool(mae < bmae and beat * 2 > len(folds)
                   and 100.0 * (1.0 - mae / bmae) >= MIN_SKILL_PCT),
        implied_kmh=float((samples["distance_km"] / (samples["minutes_to_home"] / 60)).median()),
        per_fold=per_fold,
    )
    return model, metrics


def save(model, metrics: EtaMetrics, models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"eta_{metrics.subject}.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"model": model, "subject": metrics.subject,
                     "features": FEATURES, "metrics": asdict(metrics)}, fh)
    tmp.replace(path)
    return path


def load_models(models_dir: Path = MODELS_DIR) -> dict[str, dict]:
    out = {}
    for subject in eta_subjects():
        path = models_dir / f"eta_{subject}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                out[subject] = pickle.load(fh)
    return out


def train_all(source, models_dir: Path = MODELS_DIR) -> dict:
    summary = {}
    for subject in eta_subjects():
        try:
            model, metrics = train_one(source, subject)
        except Exception as err:  # noqa: BLE001
            _log.warning("eta %s: skipped -- %s", subject, err)
            continue
        save(model, metrics, models_dir)
        summary[subject] = asdict(metrics)
    (models_dir / "eta_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the arrival-ETA models")
    parser.add_argument("--models", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args(argv)

    from . import runtime
    _, _, source = runtime.bootstrap()
    summary = train_all(source, args.models)
    print(f"\n{'subject':<8} {'samples':>8} {'arrivals':>9} {'MAE':>7} {'baseline':>9} "
          f"{'skill':>7} {'folds':>7} {'p':>7}  ships")
    for subject, m in summary.items():
        print(f"{subject:<8} {m['n_samples']:>8} {m['n_arrivals']:>9} "
              f"{m['mae_min']:>6.1f}m {m['baseline_mae_min']:>8.1f}m "
              f"{m['skill_pct']:>6.1f}% {m['folds_beating_baseline']:>3}/{m['n_folds']:<3} "
              f"{m['sign_test_p']:>7.3f}  {'YES' if m['ships'] else 'no'}")


if __name__ == "__main__":
    main()
