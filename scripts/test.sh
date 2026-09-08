#!/usr/bin/env bash
# Run the suite against the edge tree, the source of truth; occupancy-forecast/
# is generated from it. In a container so the Python and the pinned deps are the
# ones the add-on ships. requirements-dev.txt, because the shipped image carries
# no test framework. No network, no Home Assistant, no broker -- the tests run
# against a synthetic household, which keeps real entity ids out of the code.
set -euo pipefail
cd "$(dirname "$0")/.."

# Panel first: the fast half. `tsc --noEmit` is the whole UI test budget, and it
# checks the same shapes test_api_contract.py does.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e npm_config_cache=/tmp/.npm \
  -v "$PWD/occupancy-forecast-edge/panel":/w -w /w \
  -v occupancy-panel-npm:/tmp/.npm \
  node:22-alpine \
  sh -c '[ -d node_modules ] || npm ci --no-audit --no-fund; npx tsc --noEmit'

# A stale bundle installs cleanly and serves last week's panel, so a green run
# has to mean it is fresh. If this fails: scripts/build-panel.sh, then re-test.
scripts/check-panel.sh occupancy-forecast-edge

exec docker run --rm \
  -v "$PWD/occupancy-forecast-edge":/w -w /w -e PYTHONPATH=/w \
  python:3.13-slim \
  sh -c "pip install -qr requirements-dev.txt && python -m pytest -q ${*:-}"
