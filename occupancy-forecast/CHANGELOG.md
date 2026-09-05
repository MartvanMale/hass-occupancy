# Changelog

All notable changes to the Occupancy Forecast add-on are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions are [semantic](https://semver.org/spec/v2.0.0.html): `X.Y.Z`. There
is no upstream project whose version this tracks — the add-on and the
`occupancy_forecast` package are the same thing, released together.

- **MAJOR** — something you have to act on: a renamed or removed MQTT topic,
  entity or `unique_id`; a removed or renamed option in `config.yaml`; a
  `/data/history.db` migration you cannot roll back from.
- **MINOR** — additive: new published entities, new options, new endpoints, a
  new signal the forecast can use.
- **PATCH** — fixes and internals, with no change to any surface above.

`version:` in `config.yaml` moves only on a promotion, so every promotion adds a
section here and nothing else does.

The Edge add-on keeps no release history. Its `CHANGELOG.md` is the queue of what
has landed on edge and has not been promoted yet; `scripts/promote.sh` is the
moment those entries move into this file under a version number.

## 0.1.0 - 2026-09-05

First release. Everything is under `### Added`: there is no earlier published
version for anything to be a change to, and describing the add-on as it stands is
more use to a reader than the order it was built in.

### Added

- **A 48-hour occupancy forecast, per person and for the house.** One
  `sensor.*_home_probability` carrying the whole curve in its attributes, plus a
  flat sensor per horizon at +1, 2, 3, 6, 12, 24, 36 and 48 h. Published over
  MQTT discovery, so they are ordinary Home Assistant entities — renameable,
  grouped into a device, usable anywhere. There is no custom integration.

- **A horizon is served by a model only where the model was measured to beat
  that horizon's own baseline, and otherwise publishes nothing.** The ladder
  (persistence, same-slot-yesterday, a weekday-by-slot climatology) is scored on
  the same rolling-origin folds as the model, per horizon, and the sensor reads
  `unknown` where nothing cleared the bar. A horizon can also be handed back:
  the baselines improve with history, so one the model won at two weeks may lose
  at two months. `served_by` on the status page says which is which.

- **`hours_until_away` and `hours_until_home`** — when the curve is expected to
  cross in either direction, from the shipped reduction rule rather than a
  second model.

- **`next_change_at`** — the same verdict as a moment rather than a wait: the
  model says a change is coming and the routine times it. A timestamp because an
  automation cannot act on "in 3 h" without doing that arithmetic itself.

- **`minutes_until_home`, and only while somebody is actually travelling.** A
  separate model over the raw proximity trace, far sharper than the hourly curve
  can be — MAE around 5 minutes while closing. It is silent when the person is
  stationary or moving away, because its training window cannot represent a wait
  longer than three hours and answering anyway produced confident nonsense for
  somebody sitting at a desk. `unknown` here means "not travelling", not broken;
  `hours_until_home` is the one that answers all day.

- **`out_today`, `out_departure` and `out_return`** — how often this person goes
  out to a tracked zone on this weekday, and the hours they usually leave and
  come back, as timestamps. Plain causal arithmetic over their own history, not
  a model. A weekday they have been observed on and never once gone out on
  publishes no hour at all rather than falling back to an overall median: "we
  know they do not" and "we cannot say" are different answers, and only the
  second justifies a fallback.

- **An Ingress panel.** Overview is who is home, what is expected to change and
  the 48-hour curves; Data walks through what the add-on has collected and what
  it built from it; Setup is where people, zones and the history source are
  ticked. Everything is served from the add-on itself — no CDN, no font service,
  nothing fetched from outside the container.

  The panel **states** numbers and `DOCS.md` explains them. A page you leave
  open should not argue for its own figures every time you glance at it, so
  cards carry a timestamp, a count or a percentage and little else, and the
  reasoning — the ship gate, the two model families, how a crossing time is
  arrived at, the training cadence, what night shading does and does not do —
  lives in a `## Reading the panel` section instead. Empty and error states are
  the deliberate exception and keep their sentence of why, because that is the
  moment somebody is deciding whether the add-on is broken.

  Every chart carries two descriptions, not one: a short printed caption and a
  full `aria-label`. They were a single string, which meant shortening the
  visible text would silently have shortened the only description available to
  the one reader who cannot fall back on the marks.

- **Forecast verification.** Every published forecast is recorded and later
  scored against what actually happened, kept for 30 days and charted in the
  panel. It is the only thing that will tell you the add-on has quietly stopped
  working.

- **A watchdog that can see a blocked worker.** The serve loop marks each phase,
  and three missed cycles dumps every thread's stack to the log, drops the MQTT
  client and raises `worker.stalled` on `/api/status` and in the panel. It exists
  because `except Exception` around a cycle plus a `last_error` field cannot see
  a thread that is not raising but waiting — a real outage published nothing for
  11.5 hours with every health signal green. A retrain is exempt, since a train
  legitimately holds the loop for minutes.

- **Its own history archive.** Home Assistant's recorder keeps about ten days
  and long-term statistics do not cover presence at all, so the add-on appends
  its own SQLite archive under `/data` from the moment it is installed — about
  2 MB a year, never purged. Training starts at ten days. If you already archive
  to **InfluxDB**, `source: influx` trains from that history instead and is
  properly trained on the first run.

- **Training is scheduled, parallel and accounted for.** Daily while the history
  is short, weekly once it has stopped changing the answer. The fold fan-out,
  the baseline ladder and the final pooled refit share one pool with the longest
  tasks queued first, and `metrics.json` records elapsed seconds per phase — so
  where the time goes is measured rather than assumed. It was assumed wrong
  once: the ~1,900 pandas joins that build the horizon columns look like the
  expensive part and are about 8 s of a 180 s train.

- **Optional night shading** on the forecast chart, driven by a `schedule.*`
  entity you already keep, so a dip at 03:00 reads differently from a dip at
  15:00.

- **It runs on a `person` entity alone.** Proximity, tracked zones, a person
  group, a phone alarm and a country set in Home Assistant each add something
  and none is required; a missing signal is an empty column, never an error, and
  the status page shows which are active.

- **Two add-ons, installable side by side.** Stable and Edge derive their MQTT
  topic root, MQTT client id and Home Assistant device names from their own
  add-on slug, so they never collide and there is nothing to configure. If the
  slug cannot be read from Supervisor the add-on falls back to stable's prefix
  and logs a warning — which is itself the collision, so that line matters.

- **`log_level`**, from `trace` to `fatal`. At `info` the add-on logs
  transitions and one heartbeat an hour, so silence longer than an hour means it
  is not working. Chatty protocol libraries are pinned at `WARNING` at every
  setting: `websockets` logs every frame at DEBUG, and the first frame of a
  connection is the authentication handshake.

- **Advisory only.** Nothing here changes anything in your house. It publishes
  sensors and, at most, a persistent notification. If you wire the forecast to
  your heating, check `predicted_at` and ignore a stale one, so that an outage
  degrades to your previous behaviour rather than to a cold house.

- `aarch64` and `amd64`. scikit-learn publishes no armv7 wheels, so on a 32-bit
  Pi the add-on would compile for forty minutes and then fail; better not to
  offer it. It builds from source on install — pandas, numpy, pyarrow,
  scikit-learn — so the first install takes several minutes and a few hundred
  MB.
