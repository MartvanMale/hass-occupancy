# Occupancy Forecast Edge

The development build of [Occupancy Forecast](https://github.com/MartvanMale/hass-occupancy).
Same code, newer. If you only want the thing to work, install **Occupancy Forecast**
instead and ignore this one.

**Everything about what the add-on does, what it publishes and how to read the
panel is in [the stable add-on's
documentation](https://github.com/MartvanMale/hass-occupancy/blob/main/occupancy-forecast/DOCS.md).**
This page covers only what is different about running the edge build.

## Why it exists

It is designed to run **at the same time** as the stable add-on, so a change can
be watched against what you already trust before it is promoted. The two keep
apart on their own: the MQTT topic root, the MQTT client id and the Home
Assistant device names all derive from the add-on's own slug, so this one owns
`sensor.occupancy_forecast_edge_*` and stable keeps `sensor.occupancy_forecast_*`.

Nothing needs configuring for that. If the add-on cannot read its slug from
Supervisor it logs a warning and falls back to stable's prefix — which *is* the
collision, so that warning is worth reading.

## What is different from stable

Edge is stable plus whatever has not been promoted yet, and `CHANGELOG.md`'s
`## Unreleased` section is that queue written down. **Read it to know what this
build has that stable does not** — an empty section means the two are the same
code at different version strings. Everything below that section has already
shipped to stable.

## Running both

They keep separate histories, separate models and separate entities. That also
means edge starts from an empty archive: **it will publish nothing at all for its
first 10 days even if stable has been running for months**, which makes it
useless for comparison exactly when you most want to compare. The entities exist
from the first minute and read `unknown` until a model earns a horizon.

The fix is an InfluxDB history source. If you archive Home Assistant to
InfluxDB, edge trains from that history on its first run instead of accumulating
its own — and the two are comparable immediately. Set both add-ons to the same
source if the comparison is meant to be about the code: with stable on its own
archive and edge on InfluxDB, a difference between them is partly a difference
in training history.

## Where the settings are, which differs from stable right now

**In this build the history source and the MQTT broker are set on the Setup
tab**, not in the Configuration tab. Stable still has them as add-on options, so
stable's documentation describes them there; that is the one place the two
pages disagree until this is promoted.

The Configuration tab keeps `log_level` and `admin_users` only. Everything else
moved because Supervisor's options form cannot hide a field that does not apply:
the four InfluxDB fields were on screen whether or not you used InfluxDB, and
the five broker fields whether or not your broker was Home Assistant's own.

Your existing option values are copied over the first time this build starts,
and the old options stay in place for a release or two. **After that first
start, editing them in the Configuration tab does nothing** — use the Setup tab.

The Setup tab also has a **Check connection** button for InfluxDB, which reports
whether the server answers and which version it is, whether the token was
accepted, whether the bucket exists, and how many rows it can actually see for
your people. On InfluxDB 1.x it also explains that the token is
`username:password` and the bucket is `database/retention-policy`.

## Endpoints, everything else

Identical to stable — see [its
documentation](https://github.com/MartvanMale/hass-occupancy/blob/main/occupancy-forecast/DOCS.md).
