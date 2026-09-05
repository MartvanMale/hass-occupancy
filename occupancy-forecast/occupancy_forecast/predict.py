"""Serve the forecast to Home Assistant over MQTT discovery.

ADVISORY ONLY. This process never calls a Home Assistant service, never sets a
setpoint and never changes a mode. Anything that acts on these sensors must
check `predicted_at` first and ignore a stale one, so that a container outage,
a bad deploy or a broker hiccup degrades to "the house behaves as it does
today" rather than to a house that thinks nobody is coming home.

Two things here are worth knowing before changing anything:

**Model or nothing.** `train` decides per horizon whether the model actually
beat the best baseline, and records it as `metrics["ships"]`. A horizon is
published if and only if it ships AND the model produced a value for this row.
Everywhere else -- typically the longest horizons, and early on all of them --
NOTHING is published: the key is absent from `curve`, the sensor reads
`unknown`, and the chart has a gap.

This used to serve the calibrated baseline instead and label it in a `sources`
attribute. Nothing downstream read that attribute. Home Assistant saw a
percentage, the automations saw a percentage, the dashboard drew a line; the
add-on was publishing persistence dressed as a forecast and calling it honest
because a field somewhere said otherwise. The baselines have not gone away --
they are still fitted, still scored on the same folds, still the bar the gate
measures against. They stopped being an answer. They are still the yardstick.

**The MQTT client id and topic prefix must be unique on the broker.** MQTT
requires the broker to kick an existing session when a second client connects
with the same id, so sharing one with any other publisher would silently
disconnect it every time this ran. Both are derived from the add-on's own slug
-- see `config.topic_prefix`.
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from . import config, eta as eta_mod, features, log, nowcast
from . import outing as outing_mod, train

_log = log.get(__name__)

MODELS_DIR = config.MODELS_DIR
DISCOVERY_PREFIX = "homeassistant"
def state_prefix() -> str:
    return config.topic_prefix()


def client_id() -> str:
    return f"{state_prefix()}-predictor"

# How much history a serving build needs behind it: the deepest in-table window
# (`features.deepest_lookback_days`), plus the longest horizon, plus a day of
# slack. Derived rather than hand-set so widening any window over there cannot
# silently start serving NaN over here.
LOOKBACK_DAYS = (features.deepest_lookback_days()
                 + max(config.HORIZONS_H) // 24 + 2)

# The line that says whether somebody is home RIGHT NOW, which is a different
# question from thresholding a forecast and is deliberately not configurable.
# `state_now` is a fraction of the last five minutes spent at home
# (nowcast.presence_fraction), not a probability: binarising an observation at
# the user's `departure_threshold` of 0.7 would report "not home" for somebody
# who has been in the house for three of the last five minutes. Half the window
# is the only defensible cut for a time fraction.
OBSERVED_HOME_THRESHOLD = 0.5


def _load_artifact(path: Path) -> dict | None:
    """Unpickle one artifact, refusing anything this build cannot serve.

    The version check is load-bearing now rather than belt-and-braces. Before,
    an artifact from an older build merely wanted a feature that had been
    renamed and `predict_rows` degraded it to a baseline. The pooled model
    changed the artifact's SHAPE -- one file with a dict of metrics where there
    were 48 files with one each -- so an old pickle does not degrade, it raises
    somewhere less obvious. Refusing it here means a stale `/data/models`
    publishes nothing and says so, which is the behaviour the status page
    describes.
    """
    if not path.exists():
        return None
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    if artifact.get("version") != train.MODEL_VERSION:
        # Counted, not announced. A version bump makes this true for all 48
        # horizons at once, and 48 identical lines is a wall that hides the one
        # line worth reading; `load_models` says it once with the count.
        _stale.append((path.name, artifact.get("version")))
        return None
    return artifact


# Filled by `_load_artifact` during one `load_models` pass; reported once.
_stale: list[tuple[str, str | None]] = []


def load_models(models_dir: Path = MODELS_DIR) -> dict[int, dict]:
    """The per-horizon view the rest of the package expects, from two families.

    `train` fits a dedicated model per horizon AND one pooled model over all of
    them, then gates per horizon on which actually won. This hides that: every
    horizon maps to whichever artifact serves it, tagged with `kind` so
    `_model_curve` knows whether to hand it a wide row or a melted one. The
    pooled model is unpickled ONCE and aliased into every horizon it won.

    Everything downstream -- the ship gate, `server._status`'s `served_by`,
    `explore.horizon_recipe`, the panel, the API contract test -- indexes models
    by horizon and none of it has to care which family answered.
    """
    models: dict[int, dict] = {}
    _stale.clear()

    # Both artifacts carry EVERY horizon's verdict, not just the ones they won,
    # so a horizon arrives here with its `ships` flag whichever file holds it --
    # which is what `_model_curve` gates on and what `server._status` reads to
    # say which horizons are published. `metrics["kind"]` says which family
    # actually answers.
    pooled = _load_artifact(models_dir / train.POOLED_NAME)
    if pooled is not None:
        for horizon, metrics in (pooled.get("metrics") or {}).items():
            models[int(horizon)] = {
                "model": pooled["model"], "kind": metrics.get("kind"),
                "version": pooled["version"], "horizon_h": int(horizon),
                "metrics": metrics, "features": pooled.get("features")}

    for horizon in config.HORIZONS_H:
        artifact = _load_artifact(
            models_dir / train.DEDICATED_NAME.format(horizon=horizon))
        if artifact is None:
            continue
        stored = artifact.get("metrics") or {}
        metrics = stored.get(horizon) or stored.get(str(horizon))
        if not metrics:
            # Written by the worker before the gate had spoken.
            continue
        # Only take over a horizon this family actually won. Otherwise the
        # pooled entry stands, and the verdict is the same object either way.
        if metrics.get("kind") == "dedicated" or horizon not in models:
            models[horizon] = {
                "model": artifact["model"], "kind": metrics.get("kind"),
                "version": artifact["version"], "horizon_h": horizon,
                "metrics": metrics, "features": artifact.get("features")}

    if _stale:
        built = sorted({v or "unknown" for _, v in _stale})
        _log.warning("ignoring %d model file(s) built by %s -- this build is "
                     "%s. Retrain to use them; nothing is published until then.",
                     len(_stale), "/".join(built), train.MODEL_VERSION)
    _log.info("loaded %d model(s): %d dedicated, %d pooled, %d not served",
              len(models),
              sum(1 for a in models.values() if a.get("kind") == "dedicated"),
              sum(1 for a in models.values() if a.get("kind") == "pooled"),
              sum(1 for a in models.values() if a.get("kind") is None))
    return models


def current_rows(source, at: pd.Timestamp | None = None) -> pd.DataFrame:
    """The newest fully-formed row per subject, with its origin moved to `at`.

    Row SELECTION still happens on the grid value: the `dropna` runs before the
    nowcast, so a subject with no usable slot falls back to an older row rather
    than getting a live state pinned onto nothing behind it.

    `at` exists so a test can pin the clock; serving passes nothing.
    """
    at = at or pd.Timestamp.now(tz="UTC")
    start = at - pd.Timedelta(days=LOOKBACK_DAYS)
    table = features.build(source, start=start.strftime("%Y-%m-%dT%H:%M:%SZ"))
    table = table.dropna(subset=["state_now"])
    if table.empty:
        raise RuntimeError(
            "no row with a usable state_now in the last "
            f"{LOOKBACK_DAYS} days -- are the person trackers reporting?")
    newest = table.sort_values("time").groupby("subject", as_index=False).tail(1)
    return nowcast.apply(newest, source, at)


def arrival_etas(eta_models: dict[str, dict], source=None) -> dict[str, float | None]:
    """Minutes until home, per subject, or None where the question is meaningless.

    CONDITIONAL ON ARRIVING. `eta` is trained only on journeys that ended at
    home, so it answers "if they are on their way, how long" and has no opinion
    on whether they are coming at all -- ask it about somebody sitting at their
    desk and it will tell you how long the drive would take. Pair it with the
    occupancy probability, which is what `home_state.py` does.

    `house` is the first person home, not a fourth model: a house does not
    travel.
    """
    out: dict[str, float | None] = {}
    for subject in eta_mod.eta_subjects():
        artifact = eta_models.get(subject)
        out[subject] = None
        if artifact is None or not artifact.get("metrics", {}).get("ships"):
            continue
        try:
            row = eta_mod.current_row(source, subject)
        except Exception:
            continue
        if row is None:
            continue
        out[subject] = round(float(eta_mod.predict_minutes(artifact["model"], row)[0]), 1)

    people = [v for v in out.values() if v is not None]
    out[config.HOUSE_SLUG] = min(people) if people else None
    return out


def _model_curve(models: dict[int, dict], row: pd.Series) -> dict[int, float]:
    """Every horizon a model can answer for this row, by family.

    The pooled horizons go through in ONE call: the row is exploded across them
    with `features.long_frame`, the same function the fit used, so a
    horizon-relative column cannot mean one thing there and another here. The
    dedicated horizons keep the wide row and one call each, as they always did.

    Each family is tried independently and neither raising takes the other down.
    A missing value is not an error, it is a horizon that goes unpublished --
    the caller logs it, because a stale sensor deleting an hour of the forecast
    is a fault worth finding rather than something to paper over.

    This function IS the serving rule: it returns exactly the horizons that
    ship and answered, which is exactly what may be published.
    """
    shipping = [h for h in config.HORIZONS_H
                if (models.get(h) or {}).get("metrics", {}).get("ships")]
    pooled = [h for h in shipping if models[h].get("kind") == "pooled"]
    dedicated = [h for h in shipping if models[h].get("kind") != "pooled"]

    out: dict[int, float] = {}
    if pooled:
        try:
            frame = features.long_frame(row.to_frame().T, horizons=tuple(pooled))
            values = train.predict_pooled(models[pooled[0]]["model"], frame)
            out.update(zip(frame[features.HORIZON_COLUMN].astype(int), values))
        except Exception:
            pass
    if dedicated:
        frame = row.to_frame().T
        for horizon in dedicated:
            try:
                out[horizon] = float(train.predict_dedicated(
                    models[horizon]["model"], frame, horizon)[0])
            except Exception:
                continue
    return out


def _next_change(routine: dict | None, subject: str, observed_at: pd.Timestamp,
                 departure_h: int | None, arrival_h: int | None) -> dict:
    """One answer, from the two halves each is good at.

    **The model decides WHETHER, the routine decides WHEN.** The curve is a
    calibrated probability that passed its ship gate, so its crossing is the
    verdict that a change is coming and in which direction. The hour is another
    matter: measured on this household the routine's is 0.45-0.54 h out against
    the crossing's 3.14 h, and the crossing runs a median 1.5 h late when it
    speaks at all. Showing both put two estimates 2.4 h apart on one row, which
    reads as a fault however it is worded.

    **The routine is read for the day the change FALLS ON, not for today.** A
    departure sixteen hours out lands tomorrow, and one person here never goes
    out on a Thursday while going out on 88% of Fridays -- today's routine would
    be the wrong question. `outing.today` already keys on the weekday of the
    timestamp it is handed, so this passes the crossing's own moment.

    Falls back to the crossing's hour where that day has no measured one, and
    `at_from` records which, so nothing downstream has to guess whether it is
    looking at a measured time or a rounded one.
    """
    direction = ("leaving" if departure_h is not None
                 else "arriving" if arrival_h is not None else None)
    if direction is None:
        return {"direction": None, "in_hours": None, "at": None, "at_from": None}

    hours = int(departure_h if direction == "leaving" else arrival_h)
    when = observed_at + pd.Timedelta(hours=hours)
    day = outing_mod.today(routine or {}, subject, when) or {}
    hour = (day.get("departure_hour") if direction == "leaving"
            else day.get("return_hour"))
    if hour is None:
        return {"direction": direction, "in_hours": hours,
                "at": when.isoformat(), "at_from": "crossing"}
    return {"direction": direction, "in_hours": hours,
            "at": outing_mod.at_hour(when, hour), "at_from": "routine"}


def predict_rows(models: dict[int, dict], rows: pd.DataFrame,
                 etas: dict[str, float | None] | None = None,
                 out_routine: dict | None = None) -> list[dict]:
    """One record per subject: P(home) at every horizon a model earned.

    `curve` is SPARSE, and its keys are the record of what was served. A
    horizon is in it only where the model ships and produced a value; there is
    no other way for a number to get in here. Nothing is published for the rest
    -- not a baseline, not a cold-start hedge, not a shrunk daily lag.

    The record itself is still produced when `curve` is empty, which is the
    normal state for the first weeks after installation. That is deliberate and
    it is the fix for a real bug: an early build skipped the subject entirely
    when it had nothing to forecast, so a fresh install published no entities
    at all for seven weeks and no explanation. `current`, `current_at`,
    `predicted_at` and the proximity ETA are OBSERVATIONS -- the ETA has its
    own independent ship gate and may well be serving when no occupancy model
    is -- and suppressing them because a different component has nothing to say
    is the same category error as publishing a baseline, pointed the other way.
    The entities exist from the first minute and read `unknown`, which is true,
    rather than not existing, which is unanswerable.
    """
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    results = []
    for _, row in rows.iterrows():
        curve = {h: round(float(np.clip(v, 0.0, 1.0)), 4)
                 for h, v in _model_curve(models, row).items()
                 if v is not None and not np.isnan(v)}

        # A horizon that ships but produced nothing is a hole for a different
        # reason than the gate, and it is the only one that is a FAULT: a
        # feature the model wants is missing from this row, usually a sensor
        # that has stopped reporting. It used to be papered over with a
        # plausible baseline number, which is why nobody ever found one.
        # `_model_curve` swallows both of its branches with a bare `except`, so
        # without this line nothing anywhere would say it happened.
        missing = sorted(h for h in config.HORIZONS_H
                         if (models.get(h) or {}).get("metrics", {}).get("ships")
                         and h not in curve)
        if missing:
            _log.warning(
                "%s: %d shipping horizon(s) produced no value and were not "
                "published (%s) -- a feature the model wants is missing from "
                "this row", row["subject"], len(missing), missing)

        record = {
            "subject": row["subject"],
            "observed_at": pd.Timestamp(row["time"]).isoformat(),
            # Three clocks, and they mean different things. `observed_at` is
            # the left edge of the 30-minute slot the lag and climatology
            # features are anchored on; `current_at` is when the presence
            # reading behind `current` and `state_now` was taken, seconds ago;
            # `predicted_at` is when the arithmetic ran. `.get` because
            # `predict_rows` is called with hand-built frames in the tests.
            "current_at": row.get("current_at"),
            "predicted_at": now,
            "current": round(float(row["state_now"]), 4),
            # This person's routine for TODAY, or None. Not a forecast and
            # not from the model: it is what their own history says about this
            # weekday, calibrated, and it carries the counts and the spread
            # behind it so a median off four Fridays cannot be mistaken for one
            # off thirty. `outing.py` records why nothing more than arithmetic
            # ships here.
            "out": outing_mod.today(out_routine or {}, row["subject"]),
            "curve": curve,
            # Passed rather than read inside `_crossing`, so that the
            # departure/arrival asymmetry is visible here and the function stays
            # pure enough for the tests to sweep without calling `configure()`.
            "next_departure_h": _crossing(
                row, curve, going_home=False,
                threshold=config.DEPARTURE_THRESHOLD,
                min_hours=config.CROSSING_MIN_HOURS),
            "next_arrival_h": _crossing(
                row, curve, going_home=True,
                threshold=config.ARRIVAL_THRESHOLD,
                min_hours=config.CROSSING_MIN_HOURS),
            "eta_minutes": (etas or {}).get(row["subject"]),
            "model_version": train.MODEL_VERSION,
        }
        # One answer built from both: the crossing says a change is coming, the
        # routine says at what time. See `_next_change`.
        record["next_change"] = _next_change(
            out_routine, row["subject"], pd.Timestamp(row["time"]),
            record["next_departure_h"], record["next_arrival_h"])
        results.append(record)
    return results


def _crossing(row: pd.Series, curve: dict[int, float], going_home: bool,
              threshold: float, min_hours: int) -> int | None:
    """First horizon at which the forecast crosses `threshold` and STAYS across.

    Resolution is the horizon grid, which is hourly out to 48 h -- so this is
    good to the nearest hour, not to the nearest minute. Read it as "home within
    the next 6 h", not "home at 14:12": the underlying models are fitted on
    30-minute slots but only ever asked whole-hour questions.

    `min_hours` is why this is not simply the first hour past the line. A curve
    sitting at 0.8 all day that dips to 0.49 for one hour has not forecast a
    departure, it has wobbled -- and it cannot have forecast a short one either,
    because the target is the fraction of a 30-minute slot spent home, so an
    absence under about an hour has no representation in this model at all. The
    old behaviour reported that dip as a departure. See config.Settings.

    Three boundary decisions, all deliberate:

      * **A hole breaks the run**, and holes are now the normal case rather
        than a rare NaN. `curve` holds only the horizons a model earned, so a
        household whose far horizons do not ship has no forecast out there at
        all. Stepping over that gap would assert agreement across hours nobody
        forecast. The visible consequence is that both crossings return `None`
        far more often than they used to; that is the honest answer, and
        `home_state.py` already reads `unknown` as "no opinion".
      * **The run must fit inside the horizon grid, not inside the curve.**
        Requiring it to fit within `max(curve)` would report "leaving at +15 h"
        for a household whose models stop at +15 h, when what happened is that
        the forecast ran out -- the sensor would then move whenever a `ships`
        flag flipped at a retrain, for reasons that have nothing to do with the
        household. Measuring against `max(config.HORIZONS_H)` instead keeps the
        old tail exemption, which exists so a genuine +47 h departure is not
        silently turned into "unknown", and applies it only at the real end of
        the grid.
      * **An empty curve has no crossing.** A fresh install forecasts nothing
        at all, and `max({})` raises.
    """
    if not curve:
        return None
    at_home_now = float(row["state_now"]) >= OBSERVED_HOME_THRESHOLD
    if going_home == at_home_now:
        return None
    grid_end = max(config.HORIZONS_H)
    for start in sorted(curve):
        run = [h for h in range(start, start + min_hours) if h <= grid_end]
        if all(h in curve and (curve[h] >= threshold) == going_home for h in run):
            return start
    return None


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------

def connect(client: str | None = None, availability: bool = True) -> mqtt.Client:
    """Connect to the broker.

    `client_id` must be unique per connection: MQTT requires the broker to kick
    the existing session when a second client connects with the same id, so a
    collision is silent and permanent. Hence the instance suffix.
    """
    settings = config.mqtt_settings()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=client or client_id())
    if settings["username"]:
        client.username_pw_set(settings["username"], settings["password"])
    if availability:
        client.will_set(f"{state_prefix()}/availability", "offline", retain=True, qos=1)
    client.connect(settings["host"], settings["port"], keepalive=60)
    client.loop_start()
    if availability:
        client.publish(f"{state_prefix()}/availability", "online", retain=True, qos=1)
    return client


class Broker:
    """A lazily-connected, self-healing MQTT client.

    The connection used to be made once during application startup, which meant
    a broker that happened to be down at boot stopped the whole add-on from
    starting -- so `/health` could not report the one thing you needed to know.
    Now a dead broker degrades to "no entities published yet", the status page
    says so, and the next publish reconnects.
    """

    def __init__(self):
        self._client: mqtt.Client | None = None
        self.last_error: str | None = None

    def client(self) -> mqtt.Client | None:
        if self._client is not None:
            return self._client
        try:
            self._client = connect()
            # Transitions only. This is called every cycle, so logging the
            # happy path unconditionally would be a line every five minutes
            # saying nothing had changed.
            if self.last_error is not None:
                _log.info("MQTT reconnected")
            self.last_error = None
        except Exception as err:  # noqa: BLE001
            if str(err) != self.last_error:
                _log.warning("MQTT unavailable: %s. Entities will not update "
                             "until it is back.", err)
            self.last_error = str(err)
            self._client = None
        return self._client

    @property
    def connected(self) -> bool:
        return self._client is not None

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.publish(f"{state_prefix()}/availability", "offline",
                                 retain=True, qos=1)
            self._client.loop_stop()
            self._client.disconnect()
        finally:
            self._client = None


def _device_label(prefix: str) -> str:
    """`occupancy_forecast_edge` -> "Occupancy Forecast Edge".

    This feeds the MQTT device NAME, and Home Assistant builds entity ids from
    the device name -- so a casing change here silently orphans every entity the
    add-on already owns. There used to be an acronym table beside this, because
    `.title()` turned the old slug's "ml" into "Ml"; the current slug has no
    acronym in it and the table matched nothing, so it is gone. Reintroduce one
    before putting an acronym in a slug, not after.
    """
    return prefix.replace("_", " ").title()


def _discovery_payloads(subject: str) -> list[tuple[str, dict]]:
    """HA MQTT-discovery configs. Retained, so entities survive a restart.

    Note HA slugifies `device name + name` and ignores `object_id`, so the
    device is named for the entity id we actually want, not for how it reads in
    the UI. Get this wrong and the entities land under a slugified device name
    rather than the object_id they asked for, which is not fixable afterwards
    without renaming every entity by hand.
    """
    # The device NAME carries the prefix too, not just the identifiers. Home
    # Assistant builds the entity id from `device name + entity name`, so two
    # installations naming their devices identically would both slugify to
    # `sensor.occupancy_forecast_<person>_...` and the second would silently become
    # `..._2`. Deriving it from the prefix keeps stable and edge apart in the
    # entity registry, not merely on the broker.
    label = _device_label(state_prefix())
    device = {
        "identifiers": [f"{state_prefix()}_{subject}"],
        "name": f"{label} {subject.replace('_', ' ').title()}",
        # A literal, and deliberately so, against everything above it. The
        # identifiers and the name are IDENTITY and must derive from the slug,
        # or stable and edge collide. The manufacturer is BRANDING, shared by
        # both builds: deriving it would list them on the device page as two
        # different vendors of the same thing.
        "manufacturer": "Occupancy Forecast",
        "model": "home-occupancy forecaster",
    }
    base = f"{state_prefix()}/{subject}"
    availability = [{"topic": f"{state_prefix()}/availability"}]
    payloads: list[tuple[str, dict]] = []

    def sensor(key: str, name: str, template: str, **extra) -> None:
        payloads.append((
            f"{DISCOVERY_PREFIX}/sensor/{state_prefix()}_{subject}_{key}/config",
            {
                "name": name,
                "unique_id": f"{state_prefix()}_{subject}_{key}",
                "object_id": f"{subject}_{key}",
                "state_topic": f"{base}/state",
                "json_attributes_topic": f"{base}/attributes",
                "value_template": template,
                "availability": availability,
                "device": device,
                **extra,
            },
        ))

    sensor("home_probability", "Home probability",
           "{{ value_json.p_home_1h }}", unit_of_measurement="%",
           state_class="measurement")
    for horizon in config.SENSOR_HORIZONS_H:
        payloads.append((
            f"{DISCOVERY_PREFIX}/sensor/{state_prefix()}_{subject}_p{horizon}h/config",
            {
                "name": f"Home probability +{horizon}h",
                "unique_id": f"{state_prefix()}_{subject}_p_home_{horizon}h",
                "object_id": f"{subject}_home_probability_{horizon}h",
                "state_topic": f"{base}/state",
                "value_template": f"{{{{ value_json.p_home_{horizon}h }}}}",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "availability": availability,
                "device": device,
            },
        ))
    sensor("next_departure", "Hours until away",
           "{{ value_json.next_departure_h }}", unit_of_measurement="h")
    sensor("next_arrival", "Hours until home",
           "{{ value_json.next_arrival_h }}", unit_of_measurement="h")
    # Minutes, from the proximity trace -- a much sharper answer than the
    # hourly curve can give, but only meaningful while they are actually
    # travelling. See arrival_etas.
    sensor("eta_minutes", "Minutes until home",
           "{{ value_json.eta_minutes }}", unit_of_measurement="min",
           device_class="duration")

    # Going out. Timestamps rather than a fractional hour, because a
    # moment is what an automation and a person both want -- "08:12 today", not
    # "8.2". A time already past still publishes: it is what was expected, and
    # blanking it would empty the sensor exactly when someone is checking
    # whether it was right.
    #
    # `None` for anyone without a routine, which HA renders as `unknown`. That
    # is the same rule the rest of this module follows: say nothing rather than
    # publish a number that was not earned.
    # The DISPLAY name is what Home Assistant builds the entity id from, not
    # `object_id` -- so these three names are what produce `<person>_out_today`,
    # `_out_departure` and `_out_return`.
    sensor("out_today", "Out today",
           "{{ value_json.out_today }}", unit_of_measurement="%",
           state_class="measurement")
    sensor("out_departure", "Out departure",
           "{{ value_json.out_departure }}", device_class="timestamp")
    sensor("out_return", "Out return",
           "{{ value_json.out_return }}", device_class="timestamp")
    # The combined answer: the model's verdict that a change is coming, timed by
    # the routine. A moment rather than "in 3 h", because an automation cannot
    # act on a relative number without doing this arithmetic itself.
    sensor("next_change_at", "Next change at",
           "{{ value_json.next_change_at }}", device_class="timestamp")
    return payloads


def publish(results: list[dict], client: mqtt.Client) -> None:
    for result in results:
        subject = result["subject"]
        base = f"{state_prefix()}/{subject}"

        for topic, payload in _discovery_payloads(subject):
            client.publish(topic, json.dumps(payload), retain=True, qos=1)

        # EVERY horizon gets a key, and an unpublished one gets an explicit
        # null. This is not tidiness. Omitting the key renders the discovery
        # template `{{ value_json.p_home_24h }}` as the empty string, and Home
        # Assistant IGNORES an empty MQTT payload -- so the sensor would hold
        # whatever number it last had, forever, with a fresh `predicted_at`
        # beside it saying the forecast was current. A stale number is the one
        # thing worse than no number. `null` renders as "None", which HA maps
        # to `unknown` via `PAYLOAD_NONE`; same mechanism as the two crossings
        # below, which have relied on it since they were written.
        state = {f"p_home_{h}h": (None if (v := result["curve"].get(h)) is None
                                  else round(100 * v, 1))
                 for h in config.HORIZONS_H}
        state["current"] = round(100 * result["current"], 1)
        out = result.get("out") or {}
        at = pd.Timestamp(result["predicted_at"])
        state["out_today"] = (
            None if not out else round(100 * out["probability"], 1))
        state["out_departure"] = outing_mod.at_hour(at, out.get("departure_hour"))
        state["out_return"] = outing_mod.at_hour(at, out.get("return_hour"))
        state["next_change_at"] = (result.get("next_change") or {}).get("at")
        # None becomes null, which HA renders as "unknown" -- correct for "they
        # are home and not forecast to leave within 48 h".
        state["next_departure_h"] = result["next_departure_h"]
        state["next_arrival_h"] = result["next_arrival_h"]
        state["eta_minutes"] = result["eta_minutes"]
        client.publish(f"{base}/state", json.dumps(state), retain=True, qos=1)

        attributes = {
            "observed_at": result["observed_at"],
            "current_at": result.get("current_at"),
            "predicted_at": result["predicted_at"],
            "model_version": result["model_version"],
            # The observed present, so a chart can anchor the curve at t0 rather
            # than starting it an hour out. Same 0-1 scale as `curve`.
            "current": result["current"],
            "eta_minutes": result["eta_minutes"],
            "eta_is_conditional_on_arriving": True,
            # The cuts the two "hours until" numbers were read off: it makes
            # them checkable against `curve` without knowing what the panel is
            # set to.
            "crossing": {
                "departure_threshold": config.DEPARTURE_THRESHOLD,
                "arrival_threshold": config.ARRIVAL_THRESHOLD,
                "min_hours": config.CROSSING_MIN_HOURS,
            },
            # SPARSE, unlike `state` above, and deliberately so: this is the
            # record of what was served, so a chart drawing it must break its
            # line over a missing hour rather than bridge it. `state` has to
            # carry a null for every horizon because a discovery template
            # cannot express "no key"; an attribute can simply not be there.
            "curve": {str(h): v for h, v in result["curve"].items()},
            # The counts and the spread behind the three "out" sensors. A
            # median departure built from four Fridays and one from thirty read
            # identically on the sensor and are different claims; this is where
            # the difference is visible.
            "out": result.get("out"),
            # Which half timed it: `routine` is a measured hour for the day the
            # change falls on, `crossing` is the model's own rounded hour where
            # that day has none. A consumer that cares can tell them apart.
            "next_change": result.get("next_change"),
        }
        client.publish(f"{base}/attributes", json.dumps(attributes), retain=True, qos=1)


def run_cycle(models: dict[int, dict], client: mqtt.Client | None, source,
              eta_models: dict[str, dict] | None = None,
              out_routine: dict | None = None) -> list[dict]:
    """The single serving path, used by both the CLI and the HTTP server.

    A missing client is not an error: the forecast is still computed and
    returned, it simply is not published. That keeps a broker outage from
    looking like a modelling failure.
    """
    rows = current_rows(source)
    results = predict_rows(models, rows, arrival_etas(eta_models or {}, source),
                           out_routine or {})
    if client is not None:
        publish(results, client)
    return results


def main() -> None:
    models = load_models()
    if not models:
        raise SystemExit("no models in /models -- run `python -m occupancy_forecast.train`")
    client = connect()
    try:
        for result in run_cycle(models, client):
            served = f"{len(result['curve'])}/{len(config.HORIZONS_H)} horizons"
            print(f"{result['subject']:<8} now {result['current']:.2f}  "
                  f"curve {result['curve']}  serving {served}", flush=True)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
