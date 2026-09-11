"""Serve the forecast to Home Assistant over MQTT discovery.

ADVISORY ONLY: nothing here calls a Home Assistant service; whatever acts on
these sensors must ignore a stale `predicted_at`. Model or nothing: a horizon
that does not ship reads `unknown`. The broker silently kicks a duplicate
client id, so the id and topic prefix derive from `config.topic_prefix`.
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from . import config, eta as eta_mod, evaluate, features, log, nowcast
from . import outing as outing_mod, train

_log = log.get(__name__)

MODELS_DIR = config.MODELS_DIR
DISCOVERY_PREFIX = "homeassistant"

# Who created these entities, in the form MQTT discovery asks for. BRANDING,
# shared by both builds like `manufacturer`: the identity lives in the ids.
ORIGIN = {
    "name": "Occupancy Forecast",
    "sw_version": train.MODEL_VERSION,
    "support_url": "https://github.com/MartvanMale/hass-occupancy",
}
def state_prefix() -> str:
    return config.topic_prefix()


def client_id() -> str:
    return f"{state_prefix()}-predictor"

# Derived from `features.deepest_lookback_days` plus the longest horizon plus
# slack, so widening a window there cannot silently start serving NaN here.
LOOKBACK_DAYS = (features.deepest_lookback_days()
                 + max(config.HORIZONS_H) // 24 + 2)

# `state_now` is a time fraction, not a probability, so half the window is the
# only defensible cut; one constant with `evaluate`, so the two cannot drift.
OBSERVED_HOME_THRESHOLD = evaluate.HOME_THRESHOLD


def _load_artifact(path: Path) -> dict | None:
    """Unpickle one artifact, or None for one this build cannot serve: an old
    artifact has a different SHAPE and would raise somewhere less obvious.
    """
    if not path.exists():
        return None
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    if artifact.get("version") != train.MODEL_VERSION:
        # Counted, not announced: a version bump stales all 48 at once.
        _stale.append((path.name, artifact.get("version")))
        return None
    return artifact


# Filled by `_load_artifact` during one `load_models` pass; reported once.
_stale: list[tuple[str, str | None]] = []


def stale_artifacts() -> list[tuple[str, str | None]]:
    """Model files the last `load_models` refused for their version; non-empty
    tells the worker to retrain now rather than publish nothing for a week.
    """
    return list(_stale)


def load_models(models_dir: Path = MODELS_DIR) -> dict[int, dict]:
    """Every horizon mapped to whichever family's artifact serves it, tagged
    with `kind`; the pooled model is unpickled once and aliased into each.
    """
    models: dict[int, dict] = {}
    _stale.clear()
    mismatched: list[str] = []

    def fitted_for_this_house(artifact: dict, wanted: list[str], name: str) -> bool:
        # The feature list is config-DERIVED: a person or zone removed changes
        # it with no MODEL_VERSION bump, so refuse here once, not every cycle.
        stored = artifact.get("features")
        if stored is None or list(stored) == list(wanted):
            return True
        mismatched.append(name)
        return False

    # Both artifacts carry EVERY horizon's verdict, so `ships` arrives whichever
    # file holds the horizon; `metrics["kind"]` says which family answers.
    pooled = _load_artifact(models_dir / train.POOLED_NAME)
    if pooled is not None and not fitted_for_this_house(
            pooled, train.base_features(), train.POOLED_NAME):
        pooled = None
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
        if not fitted_for_this_house(
                artifact, train.features_for(horizon),
                train.DEDICATED_NAME.format(horizon=horizon)):
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
    if mismatched:
        _log.warning("ignoring %d model file(s) fitted for a different set of "
                     "people or zones (%s%s). Retrain to serve those horizons "
                     "again.", len(mismatched), ", ".join(mismatched[:3]),
                     " ..." if len(mismatched) > 3 else "")
    _log.info("loaded %d model(s): %d dedicated, %d pooled, %d not served",
              len(models),
              sum(1 for a in models.values() if a.get("kind") == "dedicated"),
              sum(1 for a in models.values() if a.get("kind") == "pooled"),
              sum(1 for a in models.values() if a.get("kind") is None))
    return models


def current_rows(source, at: pd.Timestamp | None = None) -> pd.DataFrame:
    """The newest fully-formed row per subject, its origin moved to `at` (now).

    Selected before the nowcast, so a live state is never pinned onto nothing.
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
    """Minutes until home per subject, or None; `house` is the first one home.

    CONDITIONAL ON ARRIVING: no opinion on whether they are coming at all.
    """
    out: dict[str, float | None] = {}
    for subject in eta_mod.eta_subjects():
        artifact = eta_models.get(subject)
        out[subject] = None
        if artifact is None or not artifact.get("metrics", {}).get("ships"):
            continue
        try:
            row = eta_mod.current_row(source, subject)
            if row is None:
                continue
            out[subject] = round(float(eta_mod.predict_minutes(artifact["model"], row)[0]), 1)
        except Exception as err:  # noqa: BLE001
            # Said out loud, or a stale feature list is a silent `unknown`.
            _log.warning("eta %s: no answer this cycle -- %s", subject, err)
            continue

    people = [v for v in out.values() if v is not None]
    out[config.HOUSE_SLUG] = min(people) if people else None
    return out


def _model_curve(models: dict[int, dict], row: pd.Series) -> dict[int, float]:
    """Every horizon that ships and answered for this row: this IS the serving
    rule. Pooled horizons are melted by `features.long_frame`, as in the fit;
    each family fails independently and logs why, as the caller sees a hole.
    """
    shipping = [h for h in config.HORIZONS_H
                if (models.get(h) or {}).get("metrics", {}).get("ships")]
    pooled = [h for h in shipping if models[h].get("kind") == "pooled"]
    dedicated = [h for h in shipping if models[h].get("kind") != "pooled"]
    subject = row.get("subject", "?")

    out: dict[int, float] = {}
    if pooled:
        try:
            frame = features.long_frame(row.to_frame().T, horizons=tuple(pooled))
            values = train.predict_pooled(models[pooled[0]]["model"], frame)
            out.update(zip(frame[features.HORIZON_COLUMN].astype(int), values))
        except Exception as err:  # noqa: BLE001
            _log.warning("%s: the pooled model failed at all %d of its horizons "
                         "-- %s: %s", subject, len(pooled), type(err).__name__, err)
    if dedicated:
        frame = row.to_frame().T
        failed: list[tuple[int, Exception]] = []
        for horizon in dedicated:
            try:
                out[horizon] = float(train.predict_dedicated(
                    models[horizon]["model"], frame, horizon)[0])
            except Exception as err:  # noqa: BLE001
                failed.append((horizon, err))
        if failed:
            # One line, not one per horizon: a missing column fails all 48 the
            # same way, and 48 identical lines hide the one worth reading.
            horizon, err = failed[0]
            _log.warning("%s: %d dedicated model(s) failed, first at +%dh -- %s: %s",
                         subject, len(failed), horizon, type(err).__name__, err)
    return out


def _next_change(routine: dict | None, subject: str, observed_at: pd.Timestamp,
                 departure_h: int | None, arrival_h: int | None) -> dict:
    """The model decides WHETHER a change is coming, the routine decides WHEN,
    read for the day the change FALLS ON rather than today. Falls back to the
    crossing's hour, and `at_from` records which was used.
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
    """One record per subject; `curve` is SPARSE, its keys what was served.
    The record exists even when `curve` is empty: `current` and the ETA are
    observations, and the entities must exist from the first minute.
    """
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    results = []
    for _, row in rows.iterrows():
        curve = {h: round(float(np.clip(v, 0.0, 1.0)), 4)
                 for h, v in _model_curve(models, row).items()
                 if v is not None and not np.isnan(v)}

        # A shipping horizon with no value is the one hole that is a FAULT:
        # a feature the model wants is missing, usually a sensor gone quiet.
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
            # `observed_at` is the slot the features anchor on, `current_at`
            # when the presence reading was taken, `predicted_at` this run.
            "current_at": row.get("current_at"),
            "predicted_at": now,
            "current": round(float(row["state_now"]), 4),
            # This person's routine for TODAY, or None: not from the model,
            # and it carries its own counts and spread.
            "out": outing_mod.today(out_routine or {}, row["subject"]),
            "curve": curve,
            # Passed in, so the departure/arrival asymmetry is visible here and
            # `_crossing` is testable without `configure()`.
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
        record["next_change"] = _next_change(
            out_routine, row["subject"], pd.Timestamp(row["time"]),
            record["next_departure_h"], record["next_arrival_h"])
        results.append(record)
    return results


def _crossing(row: pd.Series, curve: dict[int, float], going_home: bool,
              threshold: float, min_hours: int) -> int | None:
    """First hour at which the forecast crosses `threshold` and STAYS across for
    `min_hours`: a shorter dip is a wobble, as a 30-minute-slot target cannot
    represent an absence under about an hour.
    """
    if not curve:
        return None
    at_home_now = float(row["state_now"]) >= OBSERVED_HOME_THRESHOLD
    if going_home == at_home_now:
        return None
    # A hole in `curve` breaks the run, which is clipped at the grid's end, not
    # at `max(curve)`, or the sensor would move whenever a `ships` flag flips.
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
    """Connect to the broker. `client_id` must be unique per connection: MQTT
    kicks the existing session on a collision, silently and permanently.
    """
    settings = config.mqtt_settings()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=client or client_id())
    if settings["username"]:
        client.username_pw_set(settings["username"], settings["password"])
    # Supervisor says whether its broker wants TLS; a TLS-only broker answered
    # a plaintext CONNECT with a closed socket and "MQTT unavailable".
    if settings.get("ssl"):
        client.tls_set()
    if availability:
        client.will_set(f"{state_prefix()}/availability", "offline", retain=True, qos=1)
    client.connect(settings["host"], settings["port"], keepalive=60)
    client.loop_start()
    if availability:
        client.publish(f"{state_prefix()}/availability", "online", retain=True, qos=1)
    return client


class Broker:
    """A lazily-connected, self-healing MQTT client: a dead broker degrades to
    "no entities published yet" rather than stopping the add-on.
    """

    def __init__(self):
        self._client: mqtt.Client | None = None
        self.last_error: str | None = None
        self.last_error_public: str | None = None

    def client(self) -> mqtt.Client | None:
        if self._client is not None:
            return self._client
        # No client until the add-on knows its own name: the default prefix
        # would put this build's client id and retained states on another's.
        if not config.topic_prefix_resolved():
            reason = config.topic_prefix_error() or "slug not yet read from Supervisor"
            if reason != self.last_error:
                _log.warning("MQTT withheld: %s. Nothing is published until the "
                             "add-on's own slug is known.", reason)
            # Written here rather than by a library, so the status page keeps it.
            self.last_error = self.last_error_public = reason
            return None
        try:
            self._client = connect()
            # Transitions only; this runs every cycle.
            if self.last_error is not None:
                _log.info("MQTT reconnected")
            self.last_error = self.last_error_public = None
        except Exception as err:  # noqa: BLE001
            # Compared on the full text, so this fires on a CHANGED error.
            if str(err) != self.last_error:
                _log.warning("MQTT unavailable: %s. Entities will not update "
                             "until it is back.", err)
            self.last_error = str(err)
            self.last_error_public = log.SEE_THE_LOG
            self._client = None
        return self._client

    @property
    def connected(self) -> bool:
        """Whether the MQTT session is actually up, not whether a client exists;
        paho keeps the real state, so ask it.
        """
        return self._client is not None and self._client.is_connected()

    def close(self) -> None:
        if self._client is None:
            return
        try:
            info = self._client.publish(f"{state_prefix()}/availability", "offline",
                                        retain=True, qos=1)
            # Let the network thread send it before it is stopped, or HA shows
            # the entities available under stale retained values.
            wait = getattr(info, "wait_for_publish", None)
            if wait is not None:
                try:
                    wait(timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass
            self._client.loop_stop()
            self._client.disconnect()
        finally:
            self._client = None


def _discovery_payloads(subject: str) -> list[tuple[str, dict]]:
    """HA MQTT-discovery configs, retained so entities survive a restart. HA
    builds entity ids from `device name + name` and ignores `object_id`, so a
    mistake here costs renaming every entity by hand.
    """
    # The device NAME carries the prefix too, or two builds' devices slugify
    # alike and the second's entities silently become `..._2`.
    label = config.display_name()
    device = {
        "identifiers": [f"{state_prefix()}_{subject}"],
        "name": f"{label} {subject.replace('_', ' ').title()}",
        # A literal: the identifiers and name are IDENTITY and derive from the
        # slug; the manufacturer is BRANDING, shared by both builds.
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
                "origin": ORIGIN,
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
                "origin": ORIGIN,
            },
        ))
    sensor("next_departure", "Hours until away",
           "{{ value_json.next_departure_h }}", unit_of_measurement="h")
    sensor("next_arrival", "Hours until home",
           "{{ value_json.next_arrival_h }}", unit_of_measurement="h")
    # Only meaningful while they are travelling; see `arrival_etas`.
    sensor("eta_minutes", "Minutes until home",
           "{{ value_json.eta_minutes }}", unit_of_measurement="min",
           device_class="duration")

    sensor("out_today", "Out today",
           "{{ value_json.out_today }}", unit_of_measurement="%",
           state_class="measurement")
    # Timestamps, not a fractional hour: a moment is what an automation wants.
    sensor("out_departure", "Out departure",
           "{{ value_json.out_departure }}", device_class="timestamp")
    sensor("out_return", "Out return",
           "{{ value_json.out_return }}", device_class="timestamp")
    # A moment rather than "in 3 h": an automation cannot act on a relative
    # number without doing this arithmetic itself.
    sensor("next_change_at", "Next change at",
           "{{ value_json.next_change_at }}", device_class="timestamp")
    return payloads


def retract(subject: str, client: mqtt.Client) -> int:
    """Clear everything retained for a removed subject, or HA keeps its entities
    forever; an EMPTY retained payload is how MQTT deletes one. Returns how many
    topics were cleared.
    """
    base = f"{state_prefix()}/{subject}"
    topics = [topic for topic, _ in _discovery_payloads(subject)]
    topics += [f"{base}/state", f"{base}/attributes"]
    for topic in topics:
        client.publish(topic, "", retain=True, qos=1)
    return len(topics)


def publish(results: list[dict], client: mqtt.Client) -> None:
    for result in results:
        subject = result["subject"]
        base = f"{state_prefix()}/{subject}"

        for topic, payload in _discovery_payloads(subject):
            client.publish(topic, json.dumps(payload), retain=True, qos=1)

        # EVERY horizon gets a key, an unpublished one an explicit null: a
        # missing key renders as "", HA IGNORES an empty payload, and the sensor
        # would hold its last number forever under a fresh `predicted_at`.
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
        info = client.publish(f"{base}/state", json.dumps(state), retain=True, qos=1)
        # The one return code checked: paho queues while disconnected, so this
        # says the forecast never reached the broker (`getattr` for test fakes).
        rc = getattr(info, "rc", None)
        if rc is not None and rc != mqtt.MQTT_ERR_SUCCESS:
            _log.warning("%s: state not published (rc=%s, %s); the sensors keep "
                         "their previous values until the broker is back",
                         subject, rc, mqtt.error_string(rc))

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
            # The cuts the two "hours until" numbers were read off, so they are
            # checkable against `curve`.
            "crossing": {
                "departure_threshold": config.DEPARTURE_THRESHOLD,
                "arrival_threshold": config.ARRIVAL_THRESHOLD,
                "min_hours": config.CROSSING_MIN_HOURS,
            },
            # SPARSE, unlike `state`: a chart drawing this must break its line
            # over a missing hour rather than bridge it.
            "curve": {str(h): v for h, v in result["curve"].items()},
            # A median off four Fridays and one off thirty read identically on
            # the sensor; this is where the difference is visible.
            "out": result.get("out"),
            # Which half timed it: a measured `routine` hour or the `crossing`.
            "next_change": result.get("next_change"),
        }
        client.publish(f"{base}/attributes", json.dumps(attributes), retain=True, qos=1)


def run_cycle(models: dict[int, dict], client: mqtt.Client | None, source,
              eta_models: dict[str, dict] | None = None,
              out_routine: dict | None = None) -> list[dict]:
    """The single serving path, for the CLI and the HTTP server. With no client
    the forecast is still computed, so a broker outage is not a modelling fault.
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
