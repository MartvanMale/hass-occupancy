# The local demo instance

`compose.yaml` runs the add-on's own image on a development machine, serving a
household that does not exist. No Home Assistant, no Supervisor, no broker.

```sh
cp .env.example .env && $EDITOR .env      # paths and a port for this machine
mkdir -p demo-data demo-shots             # Docker would create these as root

docker compose run --rm instance build     --out /data   # synthetic history
docker compose run --rm instance fit       --out /data   # features and models
docker compose run --rm instance forecasts --out /data   # backtest the models

docker compose up -d demo                 # the panel, on $DEMO_PORT
docker compose run --rm capture           # 12 PNGs into $SHOTS_DIR
```

## Why it exists: screenshots

**This is the only supported way to screenshot the panel.** The Config view
renders `person.*` and `zone.*` ids verbatim and the Data view lists every
archived series, so a capture of a real installation is a picture of somebody's
home — and redacting one afterwards means catching every id in an image editor.

`scripts/demo-instance.py` builds a household instead: the same Alice and Bob the
test suite uses, with holidays, a mid-history routine change and partner
coupling. It then runs the **real** pipeline over that history, so the curves, the
ship gate and the verification card all show what the add-on actually does. Only
the household is invented.

## How it decouples from Home Assistant

`scripts/demo-serve.py` is the piece that makes it work. `server.lifespan` calls
`runtime.bootstrap()`, which builds a Home Assistant client and immediately
re-reads the timezone, country and the home's latitude and longitude from it — so
the script replaces that client with one answering from the fictional household,
and repoints the five `/data` paths.

Everything else is the real add-on: the same FastAPI app, the same worker, the
same committed panel bundle. The missing broker and the event listener's failure
to connect are the documented degraded paths, so they show up on the status page
exactly as they would on a real install.

The one thing the demo does not exercise is `run.sh` — bashio has no Supervisor to
read options from, so `compose.yaml` clears the image's entrypoint and runs the
scripts directly. Nothing under `occupancy-forecast-edge/` changes to make any of
this work.

## Where it lives

`compose.yaml` and these scripts sit at the repository root, deliberately outside
both add-on directories: `promote.sh` and `deploy-edge.sh` run `rsync --delete`
between those trees, so a file living in only one of them is removed on the next
run.

The demo's own output — `demo-data/` (about 25 MB of archive, parquet and
pickles) and `demo-shots/` — is gitignored and regenerable by re-running the
`build`, `fit` and `forecasts` steps above.
