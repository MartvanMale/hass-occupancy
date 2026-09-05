"""Import an InfluxDB archive into the add-on's local store. Runs INSIDE the add-on.

This is the payload for backfill-store-from-influx.sh, which is what you should
run. It is a separate file rather than a heredoc so it can be linted, and it
takes its Influx connection as JSON on **stdin** rather than as arguments or
environment so the token never appears in an argv or in `ps`.

WHY THIS EXISTS. The two history sources are not symmetric in what they leave
behind. `source: influx` trains from Influx directly and never fills the local
store -- `server.do_collect()` short-circuits on it -- so an install that runs
on Influx for a while and later switches to `store` starts its archive from
zero on that day. And an install that has always been on `store` has only what
the recorder could give it, which is measured in days.

This closes both gaps: one idempotent import and the store holds the whole
Influx history, after which `source: store` is a real option rather than a
six-week wait. It is also the only way to compare the two source
implementations on the same data, which is what it was written for.

It is deliberately NOT part of the add-on. Backfilling is something you do once,
by hand, knowing what you are doing -- not something the add-on should decide to
do to itself on startup.

Safe to re-run. The store's primary key is (entity_id, ts) and `append()` is
INSERT OR IGNORE, so a repeat run inserts nothing and a partial run heals
itself on the next one.
"""

import datetime as dt
import json
import sys

sys.path.insert(0, "/app")

from occupancy_forecast import config, runtime                      # noqa: E402
from occupancy_forecast.sources import HistoryStore, InfluxSource   # noqa: E402
from occupancy_forecast.sources.store import _ms                    # noqa: E402

# Big enough that the whole import is a handful of transactions, small enough
# that the collector -- which writes to this same file every few minutes -- is
# never waiting long for the lock.
CHUNK = 20_000

options = json.load(sys.stdin)
url, org = options.get("influx_url", ""), options.get("influx_org", "")
bucket = options.get("influx_bucket") or "homeassistant"
token = options.get("influx_token", "")
if not (url and org and token):
    sys.exit("stdin did not carry influx_url / influx_org / influx_token")

settings = config.Settings.load()
if settings is None:
    sys.exit("no /data/config.json -- has this add-on ever started?")
config.configure(settings)

entities = runtime.tracked_entities(settings)

# `units` is what lets InfluxSource address the proximity distance sensors at
# all: Home Assistant files those under a measurement named after the UNIT with
# the object_id in a tag, so without it the highest-value feature group comes
# back empty and does so silently. See sources/influx.py.
influx = InfluxSource(url, token, org, bucket=bucket, units=settings.units)

# Which read to use per entity, by the shape Home Assistant writes it in.
# `numeric` covers the two shapes whose field is `value` -- the work zones (a
# person count, in their own measurement) and the distances (the unit shape
# above). Everything else is a string state. Getting this wrong is not an
# error, it is an empty column, which is why the per-entity counts are printed.
distances = {pair[0] for pair in settings.proximity.values() if pair and pair[0]}
zones = set(settings.office_zones.values())
numeric_entities = distances | zones

# Ask Influx where its history starts rather than guessing. Only the string
# entities can answer -- first_seen() filters on measurement name, which is not
# how the distance sensors are stored -- and people are the entity every install
# has, so they are the right thing to anchor to.
strings = [e for e in entities if e not in numeric_entities]
start = influx.first_seen(strings)
if not start:
    sys.exit(f"nothing in Influx for {strings} -- wrong bucket, or wrong org?")
stop = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

store = HistoryStore(config.HISTORY_DB)
# SQLite's default busy timeout is 0, which turns a perfectly normal overlap
# with the collector into "database is locked" rather than a short wait. WAL
# plus this is why the add-on does not need to be stopped for the import.
store._db.execute("PRAGMA busy_timeout = 30000")

before = store.span()
print(f"window   {start} -> {stop}")
print(f"before   {before['rows']} rows, {before['first']} .. {before['last']}\n")

rows = []
for entity in entities:
    if entity in numeric_entities:
        # The store column is TEXT; numeric() re-parses it on the way out.
        pulled = [(when, str(value))
                  for when, value in influx.numeric(entity, start, stop)]
        kind = "numeric"
    else:
        pulled = influx.states(entity, start, stop)
        kind = "state"
    print(f"  {entity:44s} {kind:8s} {len(pulled):7d}")
    rows.extend((entity, _ms(when), value) for when, value in pulled)

print(f"\npulled   {len(rows)} rows")

added = 0
for i in range(0, len(rows), CHUNK):
    added += store.append(rows[i:i + CHUNK])
print(f"inserted {added} new rows ({len(rows) - added} already present)\n")

after = store.span()
print(f"after    {after['rows']} rows, {after['first']} .. {after['last']} "
      f"({after['days']} days, {after['bytes'] / 1e6:.1f} MB)\n")


def _iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


for entity_id, n, lo, hi in store._db.execute(
        "SELECT entity_id, COUNT(*), MIN(ts), MAX(ts) FROM states "
        "GROUP BY entity_id ORDER BY entity_id"):
    print(f"  {entity_id:44s} {n:7d}  {_iso(lo)} .. {_iso(hi)}")
