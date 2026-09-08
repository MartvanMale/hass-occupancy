## 0.1.1 - 2026-09-05

### Added

- A third household in the test data, which comes and goes far less tidily than
  the two before it. Test-suite only; nothing the add-on does changes.
- A build check that catches a stale Ingress panel before it can ship.

### Fixed

- The add-on installed but would not start on a Raspberry Pi 4, dying
  immediately with `Illegal instruction` and no other explanation. A dependency
  shipped an ARM build that a Pi 4's processor cannot run; it is updated. (#1)

- Installing the add-on from the repository URL failed to build, with a Docker
  error that named neither the cause nor the file. The Ingress panel's compiled
  bundle was missing from the repository and is now included.

## 0.1.0 - 2026-09-05

First release, so everything is listed as added.

### Added

- **A 48-hour occupancy forecast, per person and for the house.** One
  `sensor.*_home_probability` carries the whole curve in its attributes, plus a
  flat sensor per horizon at +1, 2, 3, 6, 12, 24, 36 and 48 h. They arrive over
  MQTT discovery as ordinary Home Assistant entities — renameable, grouped into
  a device, usable anywhere. There is no custom integration to install.

- **A horizon publishes only where the forecast was measured to beat a simple
  baseline.** Where nothing beat it, the sensor reads `unknown` instead of
  guessing. A horizon can also be handed back later, because the baselines get
  better as your history grows — `served_by` on the status page says which
  horizons the model is currently serving.

- **`hours_until_away` and `hours_until_home`** — when the curve is expected to
  cross in either direction.

- **`next_change_at`** — the same answer as a timestamp rather than a wait, so
  an automation can act on it without doing the arithmetic itself.

- **`minutes_until_home`, while somebody is actually travelling.** A separate
  and much sharper model, accurate to about 5 minutes while somebody is closing
  in. It stays `unknown` when the person is stationary or moving away: that
  means "not travelling", not broken. `hours_until_home` is the one that answers
  all day.

- **`out_today`, `out_departure` and `out_return`** — how often this person goes
  out on this weekday, and the hours they usually leave and come back, as
  timestamps. A weekday they have never once gone out on publishes no hour at
  all rather than inventing an average one.

- **An Ingress panel.** Overview is who is home, what is expected to change and
  the 48-hour curves; Data walks through what the add-on has collected and what
  it built from it; Setup is where people, zones and the history source are
  ticked. Everything is served by the add-on itself — no CDN, no font service,
  nothing fetched from outside your house. The panel states the numbers and
  `DOCS.md` explains them, under `## Reading the panel`.

- **Forecast verification.** Every published forecast is recorded and later
  scored against what actually happened, kept for 30 days and charted in the
  panel. It is the only thing that will tell you the add-on has quietly stopped
  working.

- **A watchdog that can see a stuck worker.** If the add-on stops producing
  forecasts without raising an error, it now says so — on the status page, in
  the panel and in the log. An early outage published nothing for eleven hours
  with every health signal green.

- **Its own history archive.** Home Assistant's recorder keeps about ten days
  and long-term statistics do not cover presence at all, so the add-on keeps its
  own archive under `/data` from the moment it is installed — about 2 MB a year,
  never purged. Training starts at ten days. If you already archive to
  **InfluxDB**, `source: influx` trains from that history instead and is
  properly trained on the very first run.

- **Training is scheduled.** Daily while your history is short, weekly once more
  history has stopped changing the answer. A train takes a few minutes and
  forecasting carries on while it runs.

- **Optional night shading** on the forecast chart, driven by a `schedule.*`
  entity you already keep, so a dip at 03:00 reads differently from a dip at
  15:00.

- **It runs on a `person` entity alone.** Proximity, tracked zones, a person
  group, a phone alarm and a country set in Home Assistant each add something
  and none of them is required. A missing signal is never an error, and the
  status page shows which ones are active.

- **Two add-ons, installable side by side.** Stable and Edge take their entity
  names and MQTT topics from their own add-on slug, so they never collide and
  there is nothing to configure.

- **`log_level`**, from `trace` to `fatal`. At `info` the add-on logs
  transitions and one heartbeat an hour, so silence for longer than an hour
  means it is not working.

- **Advisory only.** Nothing here changes anything in your house. It publishes
  sensors and, at most, a persistent notification. If you wire the forecast to
  your heating, check `predicted_at` and ignore a stale one, so that an outage
  degrades to your previous behaviour rather than to a cold house.

- `aarch64` and `amd64`. A 32-bit Pi is not supported: one of the add-on's
  dependencies has no 32-bit ARM build, so it would compile for forty minutes
  and then fail. The add-on builds from source on install, so the first install
  takes several minutes and a few hundred MB.
