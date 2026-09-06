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

Whatever has not been promoted yet. This directory is where the code is written;
`occupancy-forecast/` is generated from it by `scripts/promote.sh` at the moment a
change is judged ready. So edge is stable plus the queue, and `CHANGELOG.md`'s
`## Unreleased` section is that queue written down. **Read it to know what this
build has that stable does not** — an empty section means the two are the same
code at different version strings.

## Running both

They keep separate histories, separate models and separate entities. That also
means edge starts from an empty archive: **it will publish nothing at all for its
first 10 days even if stable has been running for months**, which makes it
useless for comparison exactly when you most want to compare. The entities exist
from the first minute and read `unknown` until a model earns a horizon.

The fix is `source: influx`. If you archive Home Assistant to InfluxDB, edge
trains from that history on its first run instead of accumulating its own — and
the two are comparable immediately. Set both add-ons to the same source if the
comparison is meant to be about the code: with stable on `store` and edge on
`influx`, a difference between them is partly a difference in training history.

## Options, endpoints, everything else

Identical to stable — see [its
documentation](https://github.com/MartvanMale/hass-occupancy/blob/main/occupancy-forecast/DOCS.md).
