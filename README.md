# Occupancy Forecast

A Home Assistant add-on that forecasts **who will be home, hour by hour, for the
next two days** — and, once somebody is on their way, **how many minutes until
they actually arrive**.

Entities are published over MQTT discovery, so they are ordinary Home Assistant
sensors: renameable, grouped into devices, and usable anywhere. There is no
custom integration to install.

## Install

1. Settings → Add-ons → Add-on Store → ⋮ → **Repositories**, and add:

   ```
   https://github.com/MartvanMale/hass-occupancy
   ```

2. Install **Occupancy Forecast**, start it, and open its panel from the sidebar.
3. Confirm the people it has already ticked for you. Everything else is optional.

Requires the **Mosquitto broker** add-on (or any MQTT broker) and the MQTT
integration, which ships with Home Assistant. Without a broker the add-on still
runs, collects and trains — it just cannot publish entities.

`aarch64` and `amd64` only. scikit-learn publishes no armv7 wheels, so on a
32-bit Pi the add-on would spend forty minutes compiling and then fail.

The add-on builds from source on install — pandas, numpy, pyarrow and
scikit-learn — so the first install takes several minutes and a few hundred MB.
There is no prebuilt image.

### Two add-ons: stable and edge

The repository ships **Occupancy Forecast** and **Occupancy Forecast Edge**, and they are
designed to run *at the same time* so you can compare a change against what you
already trust before promoting it.

They keep out of each other's way automatically. The MQTT topic root, the MQTT
client id and the Home Assistant device names are all derived from the add-on's
own slug (`occupancy-forecast/occupancy_forecast/config.py`, `topic_prefix`), so stable owns
`sensor.occupancy_forecast_*` and edge owns `sensor.occupancy_forecast_edge_*`. There is
nothing to configure — which matters more than it sounds, because two MQTT
clients sharing an id do not error: the broker hands the id to whoever connected
last and silently disconnects the other, forever.

The slug is read from Supervisor at startup. If that call fails the add-on falls
back to the default prefix and **logs a warning**, because for a second instance
that fallback is exactly the collision above.

**Edge is where work happens; stable is generated from it.** Development lands
in `occupancy-forecast-edge/`, and `occupancy-forecast/` changes only when a change has
earned its way into the add-on you trust:

```sh
./scripts/promote.sh       # edge -> stable, everything except config.yaml,
                           # DOCS.md and CHANGELOG.md
```

That direction is the point. Generating edge *from* stable — which is what this
repository did first — puts every edit into the stable add-on's directory
before edge ever sees it, which leaves nowhere to put work in progress and
makes "test it on edge before promoting" impossible.

## What it needs, and what it merely likes

Only one thing is required: at least one `person` entity.

| signal | where it comes from | if you have it | if you do not |
|---|---|---|---|
| `person.*` | the Person integration, one per household member | required | the add-on refuses to start |
| the tracker behind a person | the Companion app, which reports GPS | a distance to home can be worked out at all | a router or ping tracker only ever says home or not-home, and there is no distance |
| Proximity | the **Proximity** integration, one per person, against `zone.home` | distance and direction **with full history** — the largest single feature win measured here, +17 % Brier at 1 h | distance is synthesised from the person's GPS instead, accurate to 4–10 m but accumulating only from install |
| zones | Settings → Areas & zones | tick any place worth knowing about — work, a second office, the supermarket; the model works out what each means for each person, and the `out_today` sensors exist for them | the columns collapse to "away" and the model leans on presence alone |
| a person group | a `group` whose members are people | the house's occupancy directly | derived as "anyone is home" |
| country set in HA | Settings → System → General, overridable in the panel | public holidays as a feature | no holiday feature |
| next alarm | Companion app → Manage sensors → **Next alarm**, off by default | nothing yet — it is collected and not served, so turning it on now is what makes the history exist when it ships | one less thing waiting |
| a `schedule` | any schedule you already keep for waking hours | the night greyed out on the 48-hour chart | no grey bands. It is decoration either way — no feature, no model, no entity |

A missing signal is never an error. The column is left empty,
HistGradientBoosting handles it natively, and the ship gate prices what remains.
The add-on's status page shows which signals are active so you can see what
turning one on would buy you.

**Proximity and next-alarm sensors are matched on the person's slug appearing in
the sensor id**, which is how the Proximity integration and the Companion app
name them: `person.alice` finds `sensor.home_alice_distance` and
`sensor.alices_phone_next_alarm`, and does not find `sensor.pixel_7_next_alarm`.
Name the device after the person and there is nothing to configure.

### What it does not read

**Presence state, which zone that state names, distance and direction of travel,
and the calendar.** That is the whole list of inputs, and the target is the
fraction of a 30-minute slot spent at home.

So motion sensors, door contacts, `media_player`, illuminance and standalone
`device_tracker` entities are **not read** — no device class is consulted
anywhere. A house wired for occupancy detection contributes nothing here beyond
what its `person` entities already say. This forecasts presence over the next
two days, which is a different question from whether a room is occupied now, and
the sensors you already have answer the second one better.

## The first few weeks

**Home Assistant cannot give you training history.** Its recorder keeps 10 days
by default, and long-term statistics do not cover presence at all. Measured on a
recorder configured for 100 days, the history API reached back 21.

So the add-on keeps its own archive, appending from the moment it is installed —
about 2 MB a year, and it never purges. From the first day it publishes
**baselines**: persistence for the near hours, same-slot-yesterday further out.
Those are real forecasts, not placeholders — persistence alone scores a Brier of
0.053 at one hour.

It starts training at **10 days**, which is the point where there is enough
history to hold out a test window at all. Early models are weak, and most
horizons will not beat their baseline at first. That is the intended outcome
rather than a disappointment: a horizon is served by the model **only** where
the model measurably beat that horizon's own baseline, and otherwise the
baseline keeps serving. Training early therefore cannot make your forecasts
worse — it can only add skill where skill has been demonstrated.

While history is short the add-on retrains daily, so you can watch horizons
switch over one at a time. `served_by` on the status page always says which is
which, and `horizons_shipping` counts them.

**Horizons can switch back, too.** The baselines get better with more history —
a weekday-by-slot climatology needs a couple of dozen samples per cell before it
means anything — so a horizon the model won at two weeks can be handed back to
the baseline at two months. That is the gate working, not a regression.

### Where that history should live

`source` decides where a year of presence history accumulates, which makes it
the one setting that is awkward to revisit.

**`store`, the default, is right if this add-on is the only thing that will ever
read it.** That is the archive above: a SQLite table at `/data/history.db`,
appended from install and never purged. Nothing to install, and nothing that can
be configured wrong.

**`influx` is right if that history is going to feed anything else** — a second
forecaster, a notebook, Grafana, a heating model. Presence, zone membership and
distance-to-home are useful well beyond this add-on, and in a bucket they are
queryable by anything that speaks Flux instead of sitting inside one add-on's
`/data`. Set `source: influx` along with the URL, org, bucket and token, and it
trains from that archive immediately rather than accumulating its own — months
instead of days, and a properly trained model on the first run.

Three things to get right before choosing it:

- **The add-on only reads Influx; it never writes to it.** Getting Home
  Assistant's states into a bucket is Home Assistant's own **InfluxDB**
  integration, set up separately and first.
- **InfluxDB v2**, because the source speaks Flux. Missing the URL, org or token
  refuses to start rather than falling back quietly to the store.
- **Check the bucket's retention policy.** The store never purges; a bucket very
  often does, and a 30-day retention silently caps the training history at 30
  days — the opposite of the reason to switch.

And **install Proximity if you go this way**: the synthesised distance is
written to the local store, an `influx` install has no local store, so that
fallback never runs and the column stays empty.

## What it publishes

Per person, and for the house as a whole:

```
sensor.occupancy_forecast_<who>_home_probability        %    P(home) in 1 h, full 48 h curve in attributes
sensor.occupancy_forecast_<who>_home_probability_<N>h   %    N in 1,2,3,6,12,24,36,48
sensor.occupancy_forecast_<who>_minutes_until_home      min  while travelling
sensor.occupancy_forecast_<who>_hours_until_home        h
sensor.occupancy_forecast_<who>_hours_until_away        h
sensor.occupancy_forecast_<who>_out_today               %    chance of a day out to a tracked zone
sensor.occupancy_forecast_<who>_out_departure           ts   the hour they usually leave on such a day
sensor.occupancy_forecast_<who>_out_return              ts   and the hour they usually get back
sensor.occupancy_forecast_<who>_next_change_at          ts   the model says a change is coming, the routine times it
```

**`minutes_until_home` is conditional on arriving.** It is trained only on
journeys that ended at home, so for somebody sitting at their desk all afternoon
it reports how long the drive *would* take. Pair it with `home_probability` if
you are going to act on it — one says *when*, the other says *whether*.

## Advisory only

Nothing here changes anything in your house. It publishes sensors and, at most,
a persistent notification. If you wire the forecast to your heating, check
`predicted_at` and ignore a stale one, so that an outage degrades to your
previous behaviour rather than to a cold house.

## Development

Edit under `occupancy-forecast-edge/`. Then:

```sh
./scripts/test.sh          # tsc --noEmit, then pytest in python:3.13-slim
./scripts/build-panel.sh   # Vite in a container; the Dockerfile only COPYs dist/
./scripts/check-panel.sh   # is the committed bundle built from the current source?
./scripts/promote.sh       # edge -> stable
```

**The panel's compiled bundle is committed**, which is unusual and deliberate.
Nothing compiles a frontend on the Home Assistant box — the Dockerfile has no
node stage, because no other add-on on that box builds one either — and an
add-on installed from this repository's URL is a git clone and nothing more. An
ignored `panel/dist` is therefore an add-on that cannot be installed from the
store at all: the build fails at the `COPY`, saying nothing about the panel.

So `node_modules/` is ignored and `dist/` is not. Edit the panel, run
`build-panel.sh`, and commit the bundle with the source. `build-panel.sh` stamps
`panel/dist/.source-hash` with a hash of its inputs and the pre-commit hook
compares it, so committing one without the other is refused rather than shipped.

470 tests, no network, no Home Assistant, no broker. They run against a synthetic
household that matches nobody's real installation, which is what stops one
particular set of entity ids creeping back into the code.

Those three scripts work anywhere Docker does. `deploy-edge.sh`,
`deploy-stable.sh` and `backfill-store-from-influx.sh` are **author-local**:
they rsync to `HOST=ha`, an ssh alias for one particular Home Assistant box, so
they will not do anything useful in a fresh clone. Point them somewhere else or
copy the add-on directory into your own `/addons/` by hand.

Note `requirements-dev.txt`, not `requirements.txt`: pytest lives only in the
former, because the shipped image deliberately does not carry a test framework.

### Deploying

Here, both add-ons are installed **locally** — copied into `ha:/addons/<slug>/`,
where Supervisor builds them — rather than from this repository's URL. So
deploying is a file copy plus `ha addons rebuild`, and needs neither a version
bump nor a push. `deploy-edge.sh` stamps the deployed `config.yaml` with the
current commit sha, so the add-on page in Home Assistant says exactly which
build is running. (For anyone installing from the store the version *is* the
mechanism: an add-on installed from a repository URL offers an update only when
`config.yaml`'s `version:` changes.)

`scripts/deploy-stable.sh` does the same for the stable add-on, but refuses
unless `occupancy-forecast/` is committed *and* is a clean promotion of edge.

A `.githooks/pre-commit` refuses commits that touch the generated `occupancy-forecast/`
tree, and commits that change panel source without the rebuilt bundle; enable it
once with `git config core.hooksPath .githooks`.

### Versions and the changelog

Versions are documentation here rather than a mechanism: stable is semver and
moves only on a promotion, edge is `<next-stable>-dev`.

[`occupancy-forecast/CHANGELOG.md`](occupancy-forecast/CHANGELOG.md) is the release history,
in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) form, and its
preamble says what a MAJOR, MINOR or PATCH bump means for this add-on. Edge keeps
no history: [`occupancy-forecast-edge/CHANGELOG.md`](occupancy-forecast-edge/CHANGELOG.md) is
the queue of what has landed on edge and not yet been promoted, and a promotion
moves that block into the stable file under a version number and a date.
