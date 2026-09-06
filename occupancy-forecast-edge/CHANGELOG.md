# Changelog — Edge

`## Unreleased` is the queue: what has landed on edge and has not been promoted
to stable yet. Everything below it has shipped, and is word for word what
stable's own `CHANGELOG.md` carries — `scripts/promote.sh` is the moment a block
crosses over, and it is copied rather than moved, so this file keeps the whole
record of what edge has run.

File each entry under the same `### Added` / `### Changed` / `### Fixed` /
`### Removed` headings the stable changelog uses, and write it the way the stable
changelog is written: one to three plain sentences saying what changed for
somebody running the add-on. The block lands verbatim in the Changelog tab every
store user sees. The mechanism, the measurement and the alternative that was
rejected go in the commit message instead.

## Unreleased

## 0.2.0 - 2026-09-06

### Added

- An AppArmor profile confining the add-on's writes to its own data directory,
  which also takes Supervisor's security rating to its cap of 8. It enforces
  rather than warns: the add-on refuses to start if the profile is wrong, which
  is the intended trade.
- **`admin_users`** — the Home Assistant users allowed to change the
  configuration or start a retrain. Ingress proved who the caller was and
  nothing then decided whether they were allowed to retrain the house. Empty by
  default, which is exactly how every existing install behaves today.
- Supervisor now restarts the add-on if its worker gets stuck or it could not
  reach Home Assistant at start-up. It used to report itself healthy in both
  cases.
- The device page in Home Assistant now shows the add-on's name, version and
  support link.
- The options on the Configuration tab have proper names and descriptions
  instead of raw keys.
- Edge is marked advanced and experimental, so it is only offered to users who
  have switched advanced mode on.
- README sections on what the add-on stores — and that it travels in your
  backups — and on giving it a read-only, bucket-scoped InfluxDB token.

### Changed

- The add-on no longer runs as root.
- `scripts/test.sh` now checks that the committed panel bundle was built from
  the source beside it. Nothing on any path a person actually took verified that
  pair before.
- `scripts/deploy-edge.sh` stamps an uncommitted tree with a hash of its
  contents, so two different edits can no longer deploy under the same version
  string.
- `scripts/deploy-stable.sh` is removed. It targeted a local stable add-on that
  no longer exists, and created a directory on the box that collided with the
  store-installed one before failing. Stable deploys by `git push`.
- Internal tidying, with no change to any published surface: one shared spelling
  of the slot-of-day calendar, one of the panel's number formatting, and one of
  the run-length accumulator its charts use.
- `departure.py` keeps its unshipped model half, now headed with what it
  measured and why it did not ship, so the record travels with the code.
- The Dockerfile copies the panel bundle before the Python, so ordinary code
  edits no longer rebuild the panel layer.

### Fixed

- `config.yaml` described the add-on's Home Assistant access as read-only. It
  never has been: the add-on creates and dismisses its own progress
  notification.
- The panel's per-horizon quality card failed to load for some horizons and
  showed an error instead. A figure that does not exist for a horizon now shows
  as a dash.
- A configuration save that is rejected no longer changes anything. A save with
  nobody ticked was refused, but had already emptied the list of people — so the
  add-on collected nobody's history from the next cycle until it was restarted.
- If Supervisor could not be asked which add-on this is at start-up — still
  booting after a host restart, say — the edge build fell back to stable's
  entity names and MQTT topics and kept them until it was restarted, which is
  precisely the collision that fallback warns about. It now keeps retrying, and
  publishes nothing until it knows.
- The stall watchdog could not see a training run that had itself hung; a train
  now has its own one-hour deadline. The status page also reports the MQTT
  connection accurately, and a good cycle clears the error a bad one left
  behind.
- `out_departure`, `out_return` and `next_change_at` were an hour out on the two
  clock-change days.
- One of the two 02:xx observations on the autumn clock-change day was thrown
  away rather than averaged.
- Removing a person no longer breaks the forecasts until the next scheduled
  train, which could be a week away. Changing the people or zones now starts a
  retrain straight away, and a removed person's entities leave Home Assistant
  instead of holding their last forecast forever.
- The add-on asked Home Assistant for far more history than it needed on every
  five-minute cycle — up to 400 days of it in the worst case. Each entity is now
  asked only for the stretch it is actually missing.
- A horizon could be served by a leftover model from an earlier training run
  after the latest run had dropped it.
- A model trained before a configuration change is now refused at load, with a
  line saying to retrain, instead of failing silently on every cycle.
- A latent bug in the dedicated model family that would have discarded its
  warm-up rows as soon as a new published field was added.
- The decision to serve a horizon is now a fair comparison. The model and its
  baselines were scored on slightly different sets of rows, and a fold the model
  could not score at all counted as a fold it lost.
- The add-on no longer refuses to start when there is nothing to forecast yet —
  no `person` entity, or Home Assistant unreachable. It starts idle with the
  reason on the Setup tab, which is the page you need in order to fix it. Its
  configuration file is also written safely, so a power cut cannot leave an
  empty one behind.
- The history archive is no longer read and written through one shared database
  connection that was never closed.
- The "still learning" notification came back within five minutes of being
  dismissed, for as long as seven weeks. It is now sent only when something has
  actually changed: once a day while collecting, once when training starts, and
  dismissed once something ships.
- A TLS-only MQTT broker works now. Supervisor says whether the broker wants
  TLS and the add-on ignored it, connecting in plaintext.
- In the panel, the dropdown no longer clears what you have typed every ten
  seconds while it is open, and the "was it right?" and entity cards recover
  from an error without a page reload.
- A configuration change now takes effect on the live trigger subscription
  instead of waiting for a restart.
- A clean shutdown waits for the "offline" message to be sent, so the entities
  cannot be left showing as available under stale values.
- The training schedule is read on your household's clock rather than the
  container's, and a timezone the add-on cannot use is logged once instead of
  being quietly treated as UTC.
- The status page's zone scan no longer starts a second scan on every poll that
  lands during the first, and retries a failed one after a minute rather than a
  quarter of an hour.
- The explore endpoints clamp their day range consistently, and return a proper
  404 for a horizon that does not exist.
- A partially built panel no longer crashes start-up; it is skipped with a
  warning.
- Appending to the history archive no longer runs two full table scans per
  insert.
- Four API fields the panel reads were missing from the contract test.

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
