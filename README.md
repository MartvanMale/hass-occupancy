# Occupancy Forecast

A Home Assistant add-on that forecasts **who will be home, hour by hour, for the
next two days** — and once somebody is on their way, how many minutes until they
actually arrive.

It learns from the presence history you already have. scikit-learn answers "will
somebody be home?"; a smaller piece of arithmetic turns a person's distance and
direction into a travel time. Entities are published over MQTT, so they are
ordinary Home Assistant sensors — renameable, grouped into devices, usable in any
automation. There is no custom integration to install.

**Nothing here changes anything in your house.** It publishes sensors and, at
most, a persistent notification. Acting on a forecast is your automations' job.

## Install

1. Settings → Add-ons → Add-on Store → ⋮ → Repositories, and add:

   ```
   https://github.com/MartvanMale/hass-occupancy
   ```

2. Install **Occupancy Forecast**, start it, and open its panel from the sidebar.
3. Confirm the people it has already ticked for you. Everything else is optional.

The add-on builds from source on install — pandas, numpy, pyarrow and
scikit-learn — so the first install takes several minutes and a few hundred MB.
`aarch64` and `amd64` only: scikit-learn publishes no armv7 wheels.

A broker is optional, but without one the entities are never discovered by Home
Assistant, which is most of the point.

## What you need

**One thing is required: at least one `person` entity.** Everything else the
add-on will work without, and it tells you on its own status page what turning
each one on would buy you.

The things that help most, roughly in order: a GPS tracker behind each person
(the Companion app), the **Proximity** integration against `zone.home` — the
largest single measured win — and any zones worth knowing about, like work or a
second office. [The setup guide](occupancy-forecast/DOCS.md#setting-up) walks
through each one.

### If you already archive to InfluxDB

By default the add-on keeps its own SQLite archive under `/data`. Set
`source: influx` instead and it trains from an existing bucket immediately —
months of history on the first run rather than days. Three things to get right:

- **It only ever reads Influx.** Getting Home Assistant's states into a bucket is
  the InfluxDB integration's job, set up separately and first. So give the add-on
  a **read-only token scoped to that one bucket**: the only endpoint it calls is
  `/api/v2/query`, and a token that can do more than that is a token that can do
  more than it needs to.
- **InfluxDB v2**, because the source speaks Flux. A missing URL, org or token
  refuses to start rather than quietly falling back to the local store.
- **Check the bucket's retention.** The local store never purges; a bucket very
  often does, and a 30-day retention silently caps training history at 30 days —
  the opposite of the reason to switch.

Install Proximity if you go this way: the synthesised distance is written to the
local store, and an `influx` install has none, so that fallback never runs.

### What it does not read

Presence state, which zone that state names, distance and direction of travel,
and the calendar. **That is the whole list.**

So motion sensors, door contacts, `media_player`, illuminance and standalone
`device_tracker` entities are *not* read — no device class is consulted anywhere.
A house wired for occupancy detection contributes nothing here beyond what its
`person` entities already say. This forecasts presence over the next two days,
which is a different question from whether a room is occupied right now, and the
sensors you already have answer the second one better.

### The first few weeks are honest, not impressive

Home Assistant cannot give you training history — its recorder keeps about 10
days. So the add-on starts its own archive the moment you install it, and from
day one it publishes **baselines** rather than nothing.

It starts training at 10 days, and a horizon is served by the model **only**
where the model measurably beat that horizon's own baseline. Early models are
weak and most horizons will not qualify at first. That is the design, not a
disappointment: training early cannot make your forecasts worse, only better
where skill has been demonstrated. Horizons can switch back, too, as the
baselines improve.

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
journeys that ended at home, so for somebody at their desk all afternoon it
reports how long the drive *would* take. Pair it with `home_probability` if you
are going to act on it — one says *when*, the other says *whether*.

If you wire any of this to your heating, **check `predicted_at` and ignore a
stale forecast**, so that an outage degrades to your previous behaviour rather
than to a cold house.

## Your data

Everything lives under the add-on's own `/data` and nowhere else — no `/config`,
no `/share`, nothing on the LAN. It is personal data, so it is worth knowing what
is there: months of each person's presence and which zone they were in; a
distance to home (the GPS coordinates themselves are not stored); the entities
you ticked and your home's latitude and longitude, which the panel shows to
anyone who can open it; and model artifacts derived from those.

All of it is included in Home Assistant backups, so it goes wherever your backups
go. The only things the add-on sends anywhere are the MQTT sensors and one
persistent notification. It reads InfluxDB when asked to, and never writes to it.

## Two add-ons: stable and edge

The repository ships **Occupancy Forecast** and **Occupancy Forecast Edge**, and
they are designed to run *at the same time* so a change can be compared against
what you already trust. Their MQTT topics and device names derive from their
slugs, so they keep out of each other's way with nothing to configure.

Install stable unless you have a reason not to. Edge is where development lands;
stable changes only when a change has earned its way in.

## Where to find things

| | |
|---|---|
| **Setting it up, every option, reading the panel, troubleshooting** | [`occupancy-forecast/DOCS.md`](occupancy-forecast/DOCS.md) — also the Documentation tab of the add-on itself |
| **What edge is, and running both** | [`occupancy-forecast-edge/DOCS.md`](occupancy-forecast-edge/DOCS.md) |
| **Working on the code** | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| **Running the demo, screenshotting the panel** | [`docs/demo-instance.md`](docs/demo-instance.md) |
| **What changed** | [`occupancy-forecast/CHANGELOG.md`](occupancy-forecast/CHANGELOG.md) |

## Licence

MIT — see [LICENSE](LICENSE).
