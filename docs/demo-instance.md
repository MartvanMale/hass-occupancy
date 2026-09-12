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

**This is the only supported way to screenshot the panel:** the Config and Data
views render real `person.*` and `zone.*` ids, so a capture of a real
installation is a picture of somebody's home.

`scripts/demo-instance.py` builds a household instead: the same Alice and Bob the
test suite uses, with holidays, a mid-history routine change and partner
coupling. It then runs the **real** pipeline over that history, so the curves, the
ship gate and the verification card all show what the add-on actually does. Only
the household is invented.

## How it decouples from Home Assistant

`scripts/demo-serve.py` replaces the Home Assistant client `runtime.bootstrap()`
builds with one answering from the fictional household, and repoints the `/data`
paths. Everything else is the real add-on, so the missing broker shows up on the
status page as the documented degraded path.

The one thing the demo does not exercise is `run.sh` — bashio has no Supervisor to
read options from, so `compose.yaml` clears the image's entrypoint and runs the
scripts directly.

## Where it lives

`compose.yaml` and these scripts sit at the repository root, outside both add-on
trees — see DEVELOPMENT.md's "Never add a file that only one add-on needs".

`demo-data/` and `demo-shots/` are gitignored and regenerable by re-running the
`build`, `fit` and `forecasts` steps above.
