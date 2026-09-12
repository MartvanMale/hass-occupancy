"""Is this person going out today, and when?

"Out" means a day spent in a zone the household configured; `zone_other` does
not count. Not "will they leave the house", which is nearly constant. Everything
is causal, for the reason `departure._causal` gives.
"""

from __future__ import annotations

import json

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baseline, config, departure, evaluate, features

# Below this many earlier same-weekdays, that weekday's rate is NaN rather than a
# number backed by one observation. HistGradientBoosting reads NaN natively.
MIN_WEEKDAY_SAMPLES = 3

# The trailing window beside the per-weekday view, so a change of pattern shows
# up sooner than an expanding mean allows.
RECENT_DAYS = 28

# Days of history before a fit is attempted at all.
MIN_TRAIN_DAYS = 60

# 15 rather than the occupancy family's 5: the geometry here is looser, so a
# fold majority is weaker evidence.
MIN_SKILL_PCT = 15.0

# Calendar width for the fold record. The predictions are made once and grouped
# afterwards, so there is no window to choose after seeing the answer.
BUCKET_DAYS = 14


def out_columns() -> list[str]:
    """The zone columns that count as "somewhere that matters", derived from the
    configured zones so a different installation needs no code change.
    """
    return [c for c in features.zone_columns() if c != "zone_other"]


def label_out_days(table: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Add `out` to labelled days: did they reach a configured zone. Candidacy
    is decided in `departure.label_days`, so this cannot drift from it.
    """
    columns = [c for c in out_columns() if c in table.columns]
    if not columns:
        days = days.copy()
        days["out"] = np.nan
        return days

    local = table["time"].dt.tz_convert(config.tzinfo())
    keyed = table.assign(_date=local.dt.date)
    present = keyed[columns].fillna(0.0).max(axis=1)
    went = (keyed.assign(_in=present)
            .groupby(["subject", "_date"])["_in"].max() > 0)
    went = went.rename("out").reset_index().rename(columns={"_date": "date"})
    went["date"] = pd.to_datetime(went["date"])

    # Home again: the first slot at `HOME_THRESHOLD` after the last in a zone,
    # the line `departure.label_days` draws; same local day only, so a return
    # after midnight yields no hour rather than wrapping to 00:30.
    per_hour = features.slots_per_hour()
    keyed = keyed.assign(_in=present)
    local_all = keyed["time"].dt.tz_convert(config.tzinfo())
    keyed = keyed.assign(_slot=features.slot_of_day(local_all))

    returns = []
    for (subject, date), day in keyed.groupby(["subject", "_date"], sort=False):
        day = day.sort_values("_slot")
        inside = day.loc[day["_in"] > 0, "_slot"]
        if inside.empty:
            continue
        after = day[(day["_slot"] > int(inside.max()))
                    & (day["home_frac"] >= evaluate.HOME_THRESHOLD)]
        if after.empty:
            continue                    # home only after midnight, or not seen
        returns.append({"subject": subject, "date": pd.Timestamp(date),
                        "out_return_hour": int(after["_slot"].iloc[0]) / per_hour})
    exits = pd.DataFrame(returns, columns=["subject", "date", "out_return_hour"])

    merged = days.merge(went, on=["subject", "date"], how="left")
    merged = merged.merge(exits, on=["subject", "date"], how="left")
    merged["out"] = merged["out"].astype(float)
    return merged


def _causal(days: pd.DataFrame) -> pd.DataFrame:
    """That person's own record of going out, as of the morning of each row."""
    days = days.sort_values(["subject", "date"]).copy()

    by_weekday = days.groupby(["subject", "dow"], sort=False)
    days["wday_rate"] = by_weekday["out"].transform(
        lambda s: s.shift().expanding().mean())
    days["wday_n"] = by_weekday["out"].transform(
        lambda s: s.shift().expanding().count())
    days.loc[days["wday_n"] < MIN_WEEKDAY_SAMPLES, "wday_rate"] = np.nan

    recent, since, yesterday = [], [], []
    for _subject, part in days.groupby("subject", sort=False):
        indexed = part.set_index("date")
        shifted = indexed["out"].shift()
        recent.append(shifted.rolling(f"{RECENT_DAYS}D", min_periods=3).mean())
        yesterday.append(shifted)
        # How long since they last went in. A run of these is leave or illness,
        # which is the case a weekday rate is most confidently wrong about.
        went = indexed.index.where(indexed["out"] > 0)
        previous = pd.Series(went, index=indexed.index).shift().ffill()
        since.append((indexed.index - pd.DatetimeIndex(previous)).days
                     .to_series(index=indexed.index))
    days["recent_rate"] = pd.concat(recent).to_numpy()
    days["out_yesterday"] = pd.concat(yesterday).to_numpy()
    days["days_since_out"] = pd.concat(since).to_numpy()

    days["all_rate"] = days.groupby("subject", sort=False)["out"].transform(
        lambda s: s.shift().expanding().mean())
    return days


def _partner(days: pd.DataFrame) -> pd.DataFrame:
    """What the OTHER people did, as of the same morning. Shifted like
    everything else: yesterday's is knowable at 04:00 today, today's is not.
    """
    wide = days.pivot_table(index="date", columns="subject",
                            values="out_yesterday", aggfunc="first")
    for slug in config.all_slugs():
        column = f"other_{slug}_out"
        if slug in wide.columns:
            mapped = days["date"].map(wide[slug])
            # Never a person's own column mirrored back at them.
            days[column] = np.where(days["subject"] == slug, np.nan, mapped)
        else:
            days[column] = np.nan
    return days


def feature_columns() -> list[str]:
    """What the model reads. Derived, never spelled at a use site."""
    return [
        # A PLAIN INTEGER, for the reason `departure.feature_columns` gives.
        "dow", "is_weekend", "is_holiday",
        "wday_rate", "wday_n", "recent_rate",
        "out_yesterday", "days_since_out",
        *(f"other_{slug}_out" for slug in config.all_slugs()),
    ]


def feature_frame(days: pd.DataFrame) -> pd.DataFrame:
    """Labelled days plus the features, one row per person-day."""
    days = _partner(_causal(days))
    days["is_weekend"] = features.is_weekend(days["dow"])
    days["is_holiday"] = features.holiday_flags(pd.DatetimeIndex(days["date"]))
    return days


def _estimator():
    """Deliberately tiny: the independent unit is one person-day, and there are
    about a hundred and fifty of them."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=15, l2_regularization=1.0, random_state=0)


def _fit_rate_shrink(train: pd.DataFrame) -> tuple[float, float]:
    """How far to pull the weekday rate toward the person's overall rate. A
    weekday seen three times says 0.00 or 0.33, and an unshrunk baseline
    emitting confident zeros is easy to beat for reasons unrelated to skill.
    """
    base = float(train["out"].mean())
    raw = train["wday_rate"].to_numpy(dtype=float)
    truth = train["out"].to_numpy(dtype=float)
    keep = ~np.isnan(raw)
    if not keep.any():
        return 1.0, base
    losses = [float(np.mean((baseline.shrink(raw[keep], w, base) - truth[keep]) ** 2))
              for w in baseline.SHRINK_GRID]
    return baseline.SHRINK_GRID[int(np.argmin(losses))], base


def prequential(days: pd.DataFrame) -> pd.DataFrame:
    """Predict each day from a model that has only seen earlier ones: one fit
    per day, for the reason `departure.prequential` records.
    """
    embargo = departure.label_embargo()
    out = []
    for subject, part in days.groupby("subject", sort=False):
        part = part[part["candidate"] & part["out"].notna()]
        part = part.sort_values("date").reset_index(drop=True)
        columns = [c for c in feature_columns() if c in part.columns]
        for i, row in enumerate(part.itertuples()):
            train = part[part["date"] < row.date - embargo]
            if len(train) < MIN_TRAIN_DAYS:
                continue
            weight, base = _fit_rate_shrink(train)
            record = part.iloc[i]

            model = _estimator().fit(train[columns], train["out"].astype(int))
            p_model = float(model.predict_proba(part.iloc[[i]][columns])[0, 1])

            rate = record["wday_rate"]
            rate = base if pd.isna(rate) else float(rate)
            p_base = float(baseline.shrink(np.array([rate], dtype=float),
                                           weight, base)[0])
            out.append({
                "subject": subject, "date": record["date"], "dow": record["dow"],
                "out": bool(record["out"]),
                "p_model": p_model, "p_baseline": p_base,
                "shrink_weight": weight, "shrink_base": base,
            })
    return pd.DataFrame(out)


@dataclass
class OutMetrics:
    """What the model scored, and whether it earns its place."""

    subject: str
    n_days: int
    n_out: int
    n_scored: int
    n_buckets: int
    base_rate: float
    brier: float
    baseline_brier: float
    skill_pct: float
    buckets_beating: int
    sign_test_p: float
    ships: bool
    shrink_weight: float
    shrink_base: float


def score_subject(subject: str, days: pd.DataFrame,
                  scored: pd.DataFrame) -> OutMetrics | None:
    from . import train as train_mod

    mine = scored[scored["subject"] == subject]
    if mine.empty:
        return None
    span = (mine["date"] - mine["date"].min()).dt.days // BUCKET_DAYS

    truth = mine["out"].astype(float).to_numpy()
    brier = float(np.mean((mine["p_model"] - truth) ** 2))
    base_brier = float(np.mean((mine["p_baseline"] - truth) ** 2))

    beat = buckets = 0
    for _key, part in mine.groupby(span):
        buckets += 1
        y = part["out"].astype(float).to_numpy()
        beat += (np.mean((part["p_model"] - y) ** 2)
                 < np.mean((part["p_baseline"] - y) ** 2))

    skill = 100.0 * (1.0 - brier / base_brier) if base_brier else float("nan")
    all_days = days[(days["subject"] == subject) & days["candidate"]]
    return OutMetrics(
        subject=subject,
        n_days=len(all_days),
        n_out=int(all_days["out"].fillna(0).sum()),
        n_scored=len(mine), n_buckets=buckets,
        base_rate=float(truth.mean()),
        brier=brier, baseline_brier=base_brier, skill_pct=skill,
        buckets_beating=int(beat),
        sign_test_p=evaluate.sign_test(int(beat), buckets),
        ships=bool(brier < base_brier
                   and train_mod.fold_record_allows(int(beat), buckets)
                   and skill >= MIN_SKILL_PCT),
        shrink_weight=float(mine["shrink_weight"].iloc[-1]),
        shrink_base=float(mine["shrink_base"].iloc[-1]),
    )


# --- the routine, which is what is actually served -------------------------
# No model, and that is the finding: under a permutation null holding the
# weekday rate the gate fired on RANDOM labels, so calibrated arithmetic ships.

ROUTINE_NAME = "out_routine.json"

# History before a routine is published at all. Below this the per-weekday
# medians are single observations wearing a median's clothes.
MIN_ROUTINE_DAYS = 45


def fit_routine(days: pd.DataFrame) -> dict:
    """One table of numbers per person: how often, and at what hours. Fitted
    over all history rather than causally: causality is a property of the
    EVALUATION, and this is the artifact being served forward.
    """
    frame = feature_frame(days)
    fitted_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    out: dict[str, dict] = {}
    for subject, part in frame.groupby("subject", sort=False):
        if subject == config.HOUSE_SLUG:
            continue                       # a house does not go anywhere
        usable = part[part["candidate"] & part["out"].notna()]
        if len(usable) < MIN_ROUTINE_DAYS:
            continue
        weight, base = _fit_rate_shrink(usable)
        went = usable[usable["out"] > 0]

        by_weekday: dict[str, dict] = {}
        for dow, group in usable.groupby("dow"):
            here = group[group["out"] > 0]
            depart, depart_sd, n_depart = departure.summarise(here["departure_hour"])
            back, back_sd, _ = departure.summarise(
                here.get("out_return_hour", pd.Series(dtype=float)))
            by_weekday[str(int(dow))] = {
                "n": int(len(group)), "n_out": int(len(here)),
                "rate": float(group["out"].mean()),
                "departure_hour": depart, "departure_sd": depart_sd,
                "departure_n": n_depart,
                "return_hour": back, "return_sd": back_sd,
            }

        depart, depart_sd, _ = departure.summarise(went["departure_hour"])
        back, back_sd, _ = departure.summarise(
            went.get("out_return_hour", pd.Series(dtype=float)))
        out[subject] = {
            "subject": subject,
            "fitted_at": fitted_at,
            "n_days": int(len(usable)),
            "n_out": int(len(went)),
            "base_rate": float(usable["out"].mean()),
            "shrink_weight": weight,
            "shrink_base": base,
            "by_weekday": by_weekday,
            "overall": {"departure_hour": depart, "departure_sd": depart_sd,
                        "return_hour": back, "return_sd": back_sd},
        }
    return out


def save_routine(routine: dict, models_dir=None):
    from pathlib import Path

    models_dir = Path(models_dir or config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / ROUTINE_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(routine, indent=2))
    tmp.replace(path)
    return path


def load_routine(models_dir=None) -> dict:
    """JSON rather than a pickle: there is no estimator in it, and it must be
    checkable with `cat`.
    """
    from pathlib import Path

    path = Path(models_dir or config.MODELS_DIR) / ROUTINE_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def today(routine: dict, subject: str, at: pd.Timestamp | None = None) -> dict | None:
    """What to expect of this person today: how likely, and at what hours. Every
    number comes with the count behind it and the spread around it; a median off
    four Fridays and one off thirty are different claims.
    """
    entry = routine.get(subject)
    if not entry:
        return None
    at = at or pd.Timestamp.now(tz="UTC")
    local = at.tz_convert(config.tzinfo())
    dow = str(int(local.dayofweek))
    weekday = (entry.get("by_weekday") or {}).get(dow) or {}
    overall = entry.get("overall") or {}

    thin = weekday.get("n", 0) < MIN_WEEKDAY_SAMPLES
    rate = entry["shrink_base"] if thin else weekday.get("rate", entry["shrink_base"])
    probability = float(baseline.shrink(np.array([rate], dtype=float),
                                        entry["shrink_weight"],
                                        entry["shrink_base"])[0])

    def hours(key: str) -> tuple[float | None, float | None, str]:
        # A weekday seen often enough with NO days out is an ANSWER, not a gap;
        # the overall median there would be an hour nobody earned.
        if (weekday.get("n", 0) >= MIN_WEEKDAY_SAMPLES
                and weekday.get("n_out", 0) == 0):
            return None, None, "never"
        enough = weekday.get("departure_n", 0) >= MIN_WEEKDAY_SAMPLES
        if enough and weekday.get(f"{key}_hour") is not None:
            return weekday[f"{key}_hour"], weekday.get(f"{key}_sd"), "weekday"
        return overall.get(f"{key}_hour"), overall.get(f"{key}_sd"), "overall"

    departure_h, departure_sd, departure_from = hours("departure")
    return_h, return_sd, return_from = hours("return")
    return {
        "probability": round(probability, 4),
        "weekday": int(local.dayofweek),
        "n_weekday": int(weekday.get("n", 0)),
        "n_out_weekday": int(weekday.get("n_out", 0)),
        "departure_hour": departure_h,
        "departure_sd": departure_sd,
        "departure_from": departure_from,
        "return_hour": return_h,
        "return_sd": return_sd,
        "return_from": return_from,
        "fitted_at": entry.get("fitted_at"),
    }


def at_hour(at: pd.Timestamp, hour: float | None) -> str | None:
    """A fractional local hour as an ISO timestamp on `at`'s local date. A time
    already past is still published: hiding it would make the sensor unreadable
    exactly when someone is checking whether it was right.
    """
    if hour is None:
        return None
    tz = config.tzinfo()
    local = at.tz_convert(tz)
    # WALL-CLOCK arithmetic, then localise: `midnight + Timedelta` on a tz-aware
    # stamp is an hour wrong on the two transition days.
    minutes = int(round(hour * 60))
    wall = (pd.Timestamp(local.date()) + pd.Timedelta(minutes=minutes))
    # A time inside the skipped hour lands on the far side of it; a time inside
    # the repeated hour takes its first occurrence.
    return wall.tz_localize(tz, ambiguous=True, nonexistent="shift_forward").isoformat()
