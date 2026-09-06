# Occupancy Forecast

See the [repository README](https://github.com/MartvanMale/hass-occupancy) for
what this does. This page is how to set it up, and the reference for the
options, the sensors that need explaining, and the panel.

## Setting up

Three steps, and only the last of them is required. This section is what the
other two are worth.

### A broker, first

Entities are published over MQTT discovery, so the add-on wants the **Mosquitto
broker** add-on — or any broker — and the **MQTT** integration, which ships
with Home Assistant. There is no custom integration to install and nothing to
write in YAML: the sensors arrive as ordinary Home Assistant entities.

`mqtt:want` in the manifest is *want*, not *need*. Without a broker the add-on
still starts, still collects history and still trains. It simply publishes
nothing, and `mqtt` in `/health` says which it is.

### Where the history should live

`source` is the one decision that is awkward to revisit later, because it
decides where a year of presence history accumulates.

**`store` is right if this add-on is the only thing that will ever read that
history.** It is a table of `(entity_id, ts, value)` in SQLite at
`/data/history.db`, appended from the moment you install, about 2.3 MB a year,
and it is **never purged**. Nothing to install, and nothing that can be
configured wrong.

**`influx` is right if that history is going to feed anything else** — a second
forecaster, a notebook, Grafana, a heating model. Presence, zone membership and
distance-to-home are useful well beyond this add-on, and in a bucket they are
queryable by anything that speaks Flux rather than sitting inside one add-on's
`/data` where only this add-on can reach them. Training from an archive that
already goes back months is the other half of the bargain: a properly trained
model on the first run instead of after ten days.

Three things to get right before choosing it:

- **The add-on only ever reads Influx. It never writes to it.** Getting Home
  Assistant's states into a bucket is Home Assistant's own **InfluxDB**
  integration, set up separately and first; this add-on then reads what that
  integration has been writing. "Use Influx" is two pieces of plumbing and the
  add-on is the second one.
- **InfluxDB v2.** The source speaks Flux and needs the URL, org, bucket and
  token together. `source: influx` with any of the URL, org or token missing
  refuses to start rather than falling back quietly to the store.
- **Check the bucket's retention policy.** The local store never purges; a
  bucket very often does, and a 30-day retention silently caps the training
  history at 30 days — the opposite of the reason to switch. This is the one
  that bites.

Two consequences worth expecting rather than reporting. An `influx` install
keeps **no local archive**, so the first two cards on the Data tab say so
instead of drawing an empty table. And **`source` in the Configuration tab wins**
over whatever is stored in `/data/config.json`, so the Configuration tab is
where to change it.

**On `influx`, install the Proximity integration too.** Distance-to-home is
otherwise synthesised from GPS and written to the local store — and an `influx`
install has no local store, so that fallback never runs and the distance column
stays empty. On `store` the fallback works and Proximity is merely better; on
`influx` it is the only source of the single most valuable feature the add-on
has.

### Confirm the people

Open the panel from the sidebar. On first run the add-on has already proposed a
configuration from what Home Assistant has — **every `person` entity ticked**,
the first person group if there is one, and any Proximity sensors it could match
— so this step is usually confirming it and unticking anyone who should not be
forecast. **It is the only required step**, and a `person` entity is the one
thing the add-on refuses to start without.

Everything below is optional, and a missing signal is never an error: the column
is left empty, the model reads it as *unknown* rather than as zero, and the ship
gate prices what remains. The Setup tab's "What this installation has" card
lists the same signals as the table below, with `active` or `not available`
beside each, so the two can be read side by side.

### What to turn on, and what each buys

| signal | where it comes from | what it buys |
|---|---|---|
| **people** | the Person integration — one per household member, each with a device tracker attached | required; occupancy is what all of this is about |
| **the tracker behind a person** | the Home Assistant Companion app, which reports GPS. A router or ping tracker only ever says home or not-home | GPS is what makes a synthesised distance possible at all |
| **Proximity** | the **Proximity** integration, one per person, against `zone.home` | the largest single feature win measured here: **+17 % Brier at 1 h**. It also has *history*, so it trains from day one. Without it the distance is synthesised from the person's GPS — accurate to 4–10 m, but only forward from install |
| **zones** | Settings → Areas & zones. Work, a second office, a school, the supermarket | the `zone_*` columns, and the three `out_today` sensors, which exist only for zones you ticked. Zones are roleless on purpose — which one means "work" for which person is learned, not declared |
| **a person group** | a `group` whose members are people | the house's own occupancy, measured. Without one it is derived as "anyone is home" |
| **holiday calendar** | the `holidays` package, seeded from the country set in Home Assistant | the `is_holiday` feature. Set it to the calendar the household actually *keeps*, which is not always where the house is |
| **next alarm** | Companion app → Manage sensors → **Next alarm**, which is off by default | nothing yet — it is collected and not served. Turn it on now so the history exists when it ships |
| **night shading** | any `schedule` entity you already keep | display only: it greys the hours outside the schedule on the 48-hour chart. No feature, no model, no published entity |

### Two names that have to line up

**Proximity and next-alarm sensors are matched on the person's slug appearing in
the sensor id.** `person.alice` finds `sensor.home_alice_distance` and
`sensor.alices_phone_next_alarm`; it does not find `sensor.pixel_7_next_alarm`,
and nothing will announce the miss beyond that row reading `not available`.
Naming the device after the person is the whole fix.

**Do not rename a zone once it is ticked.** Home Assistant writes a zone's
*friendly name* into the person's state, so that name is the only per-person
zone signal history contains — and a rename strands every earlier row in
`zone_other`. The zones row on the Setup page counts away-states in history that
match no ticked zone for exactly this reason: a non-zero count there, quoting a
name you no longer use, is what a rename looks like from the inside.

### What the model is actually fed

The target is **`home_frac`** — the fraction of a 30-minute slot spent at home,
time-weighted, rather than whatever the tracker happened to say at the slot
boundary. That is not fussiness: measured on a real installation, 19 % of one
person's presence episodes are shorter than five minutes, and a ninety-second
GPS blip landing on a grid point would otherwise become an empty house.

What is read in order to predict it: **presence state, which zone that state
names, distance and direction of travel, and the calendar.** That is the whole
list.

Which means, explicitly, that **motion sensors, door contacts, `media_player`,
illuminance and standalone `device_tracker` entities are not read** — no device
class is consulted anywhere in the add-on. A house wired for occupancy detection
contributes nothing here beyond what its `person` entities already say. This
forecasts presence over the next two days, which is a different question from
whether a room is occupied now, and the second one is answered better by the
sensors you already have.

Those inputs turn into rather a lot of columns, grouped into families, and the
Data tab lists every family with a sentence saying what it is. That listing is
the summary of record: it is generated from the feature build itself, so unlike
a list on this page it cannot quietly go stale.

## Options

| option | meaning |
|---|---|
| `log_level` | standard add-on log level; see `## The log` below |
| `source` | `store` (default) accumulates history from Home Assistant. `influx` reads an existing InfluxDB v2 archive instead. Which to pick is `### Where the history should live` above |
| `influx_url` / `influx_org` / `influx_bucket` / `influx_token` | required when `source` is `influx` |

Everything else — which people, which zones, which group — is configured on the
add-on's own panel, because Supervisor's options form has no entity picker and
typing `person.alice` into YAML is not a user interface. `## Setting up` above
is what to pick there and what each one is worth.

## The three timestamps on a forecast

A published forecast carries three, and they are not the same thing:

| | what it means |
|---|---|
| `current_at` | when the presence reading behind `current` was taken — seconds ago |
| `observed_at` | the left edge of the 30-minute slot the lag and climatology features are anchored on — up to 30 minutes ago |
| `predicted_at` | when the arithmetic ran |

`observed_at` looks stale and is not: a slot runs *forward* from its left edge,
so the newest row is the one in progress. Anything acting on a forecast should
still check `predicted_at`, which is the one that says whether the add-on is
alive.

## When a crossing counts

`sensor.*_hours_until_away` and `sensor.*_hours_until_home` are the 48-hour curve
reduced to a single number, and the Setup page says how that reduction is done: a
probability cut per direction, and how many consecutive hours the forecast has to
stay past it. The defaults are 50 %, 50 % and a two-hour run.

Two things this is **not**. It does not change the forecast — the curve is
identical whatever these are set to, and no retrain is involved; the next serve
cycle picks the change up. And it cannot make the sensors notice a shorter
absence: the model's target is the fraction of a 30-minute slot spent at home,
so a walk round the block never appears as a departure at any setting. These
sensors answer "gone for a while", and a household that also wants "popped
out" should read presence directly.

## Minutes until home, and when it says nothing

`sensor.…_<person>_minutes_until_home` is the sharpest arrival answer the add-on
has — **MAE 4–5 minutes while somebody is driving home** — and it is silent most
of the day on purpose.

It is conditional on being *on a journey home*. Training enforces that by
discarding any sample more than three hours from an arrival, and serving
enforces the same thing: no number is published unless the distance to home is
closing faster than 5 km/h. Without that check the model was asked about a person
sitting at a desk 32 km away and answered **169 minutes** — the top of its
trained range — when they were six hours from home. It has no way to express a
longer wait, and nothing in its inputs separates "stationary at 32 km at 12:10"
from "stationary at 32 km at 17:50".

Measured against uncensored truth over ~19,500 and ~7,600 moments away from home:
while stationary or moving away only **13–27%** of moments are genuinely within
the model's three-hour range; above 5 km/h of closing speed that rises to **57%**
and the median true wait falls from 585 minutes to 30–58.

So `unknown` on this sensor means "not travelling", not "broken". The hourly
`hours_until_home` beside it is the one that answers all day.

## Out today

Three sensors per person, and they are **not** the model's output:

| sensor | what it is |
|---|---|
| `sensor.…_<person>_out_today` | how often this person goes out to a tracked zone on this weekday, calibrated |
| `sensor.…_<person>_out_departure` | the median time they leave home when they do, as a timestamp today |
| `sensor.…_<person>_out_return` | the median time they leave that zone, ~20 min before they are home |

**"Out" means a day spent in a zone you configured** — a workplace, a school, a
stable, whatever you told the add-on matters. Nothing extra needs configuring and
no threshold needs tuning. `zone_other` does not count: it means "in some zone
nobody named", which is the opposite of a declaration. A household that
configures somewhere it drops into briefly and often — a gym — would want a
per-person zone setting instead; that is a setting to add when somebody has one.

They exist because the forecaster answers a different question well and this one
badly. "When will they leave the house" mixes the commute with the gym and the
school run: measured on a real archive, the first departure of any kind has a
standard deviation of **3.50 h**, and departures on a day out alone **0.74 h**.
Splitting them by the zone turns an unanswerable question into a half-hour one.

**A model was tried here and refused.** Against a permutation null — shuffling
the label within (person, weekday), which holds the weekday rate and destroys
everything else — it scored p = 0.12 for both subjects, and the 15% ship gate
fired on 4–12% of *randomly shuffled* labels. At this sample size the null has a
standard deviation of 13 points, so nothing short of a very large effect could be
distinguished from chance. What is published is the calibrated arithmetic, which
is good in absolute terms: Brier 0.128 / 0.147 against a flat rate's ~0.21.

On the panel these appear as a second line under each person on "Next
expected change", beside the forecast's own answer for the same person.

Each sensor's attributes carry the counts and the spread behind it. A median
departure built from four Fridays and one built from thirty read identically on
the sensor face and are different claims; `out.n_out_weekday` and
`out.departure_sd` are where the difference is visible, and `departure_from` says
whether the number is this weekday's own, the fallback across all days out, or
`never` — a weekday seen often enough with no days out at all, where no hour is
published because there is nothing to state. Nothing is published until 45 usable
days exist.

## "Was it right?" — and what the add-on remembers

Every other score in the panel is measured at *training* time: rolling-origin
cross-validation over the feature table, answering "how would a model fitted on
folds `[0,k)` have scored on fold `k`". It is a real number, and it is not a
measurement of what the add-on actually did. It cannot see the nowcast pin, a
tracker that went quiet at 07:00, or the ship gate deciding a horizon is not
worth publishing.

So the add-on records what it publishes, in a `forecasts` table in
`/data/history.db` beside the archive: one row per subject, per served horizon,
per cycle, keyed by the slot the forecast was *about*. **A horizon nothing was
published for writes no row** — the absence is the record, and it is what puts
the gap on the chart.

Two consequences worth expecting rather than reporting:

- **The card is empty for the first `horizon` hours after a deploy.** A +6 h
  forecast made at 09:00 is about 15:00, and there is nothing to compare it to
  until 15:00. It says so rather than drawing an empty chart.
- **It stays empty for a horizon the model never earns.** That is the same fact
  the horizon strip states, seen over time instead of right now.

Unlike the archive, this table is pruned — 30 days
(`config.FORECAST_RETENTION_DAYS`), about 200k rows and a few megabytes. The
archive is training history and is kept forever; this is a chart of the recent
past and nothing refits from it.

The live Brier under the chart and the backtest Brier in step five are therefore
**different quantities, and are meant to be read side by side**. A large gap
between them is a finding about the deployment, not a bug in the card.

## Reading the panel

The panel states numbers and does not explain them. This section is the
explanation.

### Which horizons publish, and why some do not

Both model families are fitted at every one of the 48 horizons, and a horizon
**publishes only where the model beat its own best baseline** by enough, in
enough of the folds, to be worth the risk. Where it did not, nothing is
published: the sensor reads `unknown` and the 48-hour chart has a gap. That is
the gate working, not a failure — the baseline is the bar, never the answer.

The two families are:

| | what it reads |
|---|---|
| **dedicated** | one horizon's own features; a separate fit per horizon |
| **pooled** | one model fitted over every horizon at once |

Typically one wins the near horizons and the other the far ones. **Where they
cross is measured on this household's own history, not chosen** — which is why
the horizon strip on Overview shows two colours and where they hand over is the
most interesting thing on it. The `other family` column in the Data tab's
metrics table is whichever one lost at that horizon.

The strip is about the MODELS, not about the last forecast. A horizon that ships
can still come out empty on the chart if a feature that model wanted was missing
from that particular row; the add-on logs a warning when it happens.

### Next expected change

The time on this card is **the first hour the forecast crosses the cut and stays
crossed** for the configured run — see `## When a crossing counts` above for
the two cuts and the run length, which are settings.

It is hourly, so read `18:00` as "within that hour", not as "at 18:12". The
chip beside it says `within the hour` rather than `in 0 h` for the same reason.

**The model decides WHETHER a change is coming; the time comes from that
person's own routine for the day it falls on.** These are two different
mechanisms and it matters when they disagree: a card that said "expected back in
3 h" over "expected back around 18:00" would be one event with two answers 2.4 h
apart. Only one time is shown.

`On the way, 12 min out.` appears only while somebody is demonstrably
travelling — see `## Minutes until home` above for what "demonstrably" means and
why the number is absent most of the day.

### Training cadence

The Training card says `Weekly` or `Daily`, and the rule behind it is: **daily
while the history is still short enough that another day changes the answer,
weekly once it has stopped.** Nothing is published at all until there is enough
history to hold out a test window — the sensors exist and read `unknown` until
then, which on a fresh install is the normal state for the first 10 days.

A retrain is idempotent and non-destructive: it either replaces the models or
fails and leaves the old ones serving. During one, the forecast carries on being
served from the models the add-on already has. That is why the four buttons have
no confirmation dialog — the honest signals are the disabled buttons and the
elapsed time.

### Night shading

**Display only.** Picking a schedule greys out the hours outside it on the
48-hour chart, which is what makes a dip at 03:00 read differently from a dip at
15:00. It changes no feature, no model and no published entity.

It is read from **the schedule's own last week**, because Home Assistant will
not say what a schedule is going to do next. A schedule changed today shows its
new shape on the chart a week from today.

### A feature family with no columns

On Setup, a signal listed as *not available* is **not an error**. The column is
left empty, the model ignores it, and the forecast is a little less sharp. On
the Data tab the same fact appears as a family with zero columns: it is a signal
this house does not have, not a fault.

The one state worth acting on is different and is called out in red or orange:
an entity that is configured and has **never produced a row**.

### The Data tab

The panel's second tab reads the add-on's own data back out — the raw archive,
the feature table, what each horizon is allowed to use, and how well the models
scored. It is read-only and nothing on it is polled.

| Endpoint | What it answers |
|---|---|
| `GET /api/explore/archive` | every entity in `/data/history.db`, its row count and span, and whether anything reads it |
| `GET /api/explore/entity?entity_id=&days=` | one entity's raw transitions, and the `home_frac` the feature builder derives from them |
| `GET /api/explore/features` | the feature table by column family, from the parquet footer — never its data |
| `GET /api/explore/feature-series?subject=&column=&days=` | one column, for one subject, over time |
| `GET /api/explore/horizon/{h}` | which columns horizon `h` fits on, and which daily lags it may not touch |
| `GET /api/explore/metrics` and `/metrics/{h}` | per-horizon scores, the per-fold spread and the calibration curve |

Each answers `{"available": false, "reason": "…"}` rather than a 404 when the
layer it reads does not exist yet, which on a fresh install is the normal state
for most of them. An installation whose `source` is `influx` keeps no local
archive, so the first two say so.

An entity listed as **unused** is harmless — left over from an earlier
configuration, and ignored by the feature build. An entity listed as
**configured with no rows** is worth acting on: something on the Setup tab names
an entity that has never reported, and the model is training without it.

**Step 4, what a horizon is withheld.** Each horizon gets its own model and its
own slice of the feature table, and the interesting part is what is taken away:
**a lag that reaches past the moment being predicted would train beautifully and
could never be served.** The column is still in the table — the table is built
once for every horizon at the same time — and the gate is applied when each
model picks its features, which is why the step shows the same daily lags marked
`used` or `not allowed` depending on where the slider is.

The folds are held apart for the same reason. `embargo_hours` on that card is
the gap between the training rows and the test rows, so the training rows cannot
see the test answer through a lag that straddles the boundary.

**Step 5, reading the metrics table.** Brier is a squared error on a
probability, so lower is better and the `baseline` column is the number to beat;
`skill` is the percentage below it. `other family` is whichever of the two fits
lost at that horizon — see `### Which horizons publish` above. A slot counts as
not observed, and is drawn as a break rather than as a zero, when nothing
reported or less than the minimum share of the slot was covered; a missing
feature value is read by the model as *unknown*, never as zero.

### Forecasts only update every five minutes

Expected, unless the status page says **Live updates: connected**. The add-on
subscribes to Home Assistant over the WebSocket API so that an arrival is
noticed rather than waited for; if that subscription is down it falls back to
the five-minute poll, which still works and is not an outage. `listener` in
`/health` carries the reason.

## The log

The add-on writes to its **Log** tab in the shape Supervisor uses,
`[HH:MM:SS] LEVEL: message`, in local time.

At the default `info` level it reports **transitions** — models loaded, MQTT up
or down, the trigger subscription, training start and finish — plus **one
heartbeat line an hour** whether or not anything is wrong:

    [09:31:48] INFO: ready: 48 model(s), 2 person(s), source=store, log level info
    [10:31:52] INFO: alive: 12 cycles, 42/48 horizons shipping, mqtt up, listener up, last predict 2026-09-02T08:31:50+00:00

That heartbeat exists so that **silence means something**. If the last line is
more than an hour old, the add-on is not working, and the watchdog will have
written a stall line and a full thread dump saying where it stopped.

Set **log level** to `debug` in the Configuration tab for a line per five-minute
cycle with timings, and for the HTTP and MQTT libraries' own output. Set it to
`warning` for problems only.

The WebSocket library is deliberately excluded from this: it logs every frame,
including the authentication one, so no log level will make it print a token.

## Endpoints

The panel is the intended way in; these exist for debugging.

| | |
|---|---|
| `GET /health` | everything: history span, which horizons are served, MQTT state, last error |
| `GET /api/status` | what the panel renders |
| `GET /api/forecast` | the forecast as last published to MQTT, rather than recomputed |
| `GET /api/candidates` | entities discovered as trackable people, zones and groups |
| `GET /api/config` | the current selection |
| `POST /api/config` | change the selection. This is what the panel writes |
| `POST /collect` | pull new history now |
| `POST /predict` | recompute and republish |
| `POST /train` | rebuild the feature table and refit. Refuses with 409 below 10 days of history |
| `POST /reload` | re-read the model files without retraining |

## Troubleshooting

**No entities appear.** Check `mqtt` in `/health`. The add-on runs fine without a
broker — it still collects and trains — but it cannot publish. Install Mosquitto.

**Most horizons read `unknown`.** Expected for the first 10 days, when there is
not yet enough history to train at all, and normal for a while after that: a
horizon publishes only where the model beat that horizon's own baseline. Where it
did not, nothing is published rather than something unearned. The panel shows the
days remaining and how many horizons the model currently wins. Horizons can also
be handed *back* as history grows and the baselines improve.

**`minutes_until_home` is unknown.** It only has a value while somebody is
actually travelling — see `## Minutes until home` above — and it needs ~500
recorded journeys before it will ship at all.

**Nothing has been published for hours and the log is silent.** The heartbeat
line is hourly, so silence older than that means the worker has stopped. The
watchdog dumps every thread's stack to the log and raises `worker.stalled` on
`/api/status`; that dump is what says where it stopped.
