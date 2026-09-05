#!/usr/bin/env bash
# Run the test suite against the edge tree -- which is the source of truth, so it
# is the tree worth testing. `occupancy-forecast/` is generated from it and identical.
#
# In a container rather than a local venv so that the Python version and the
# pinned dependencies are the ones the add-on actually ships with, on any machine
# and with nothing installed on the host.
#
# requirements-dev.txt, not requirements.txt: pytest lives only in the former,
# because the shipped image deliberately does not carry a test framework.
#
# No network, no Home Assistant, no broker. The tests run against a synthetic
# household that matches nobody's real installation, which is what stops one
# particular set of entity ids creeping back into the code.
set -euo pipefail
cd "$(dirname "$0")/.."

# The panel first, because it is the fast half and a type error there is the
# most likely thing to be broken. `tsc --noEmit` is the whole UI test budget: it
# catches the class of mistake that used to be caught by asserting on generated
# markup -- a renamed API field, a handler that no longer exists -- and the
# shapes it checks against are the other half of test_api_contract.py.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e npm_config_cache=/tmp/.npm \
  -v "$PWD/occupancy-forecast-edge/panel":/w -w /w \
  -v occupancy-panel-npm:/tmp/.npm \
  node:22-alpine \
  sh -c '[ -d node_modules ] || npm ci --no-audit --no-fund; npx tsc --noEmit'

exec docker run --rm \
  -v "$PWD/occupancy-forecast-edge":/w -w /w -e PYTHONPATH=/w \
  python:3.13-slim \
  sh -c "pip install -qr requirements-dev.txt && python -m pytest -q ${*:-}"
