#!/usr/bin/env bash
# Build the Ingress panel into <tree>/panel/dist, never on the Home Assistant box;
# the result is committed and the Dockerfile only COPYs it. Finishes by stamping
# dist/ with scripts/panel-source-hash.sh so a stale bundle can be caught later.
# `--user` is not optional, or node_modules/ and dist/ come back root-owned.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/container-guard.sh

TREE="${1:-occupancy-forecast-edge}"
PANEL="$PWD/$TREE/panel"
[[ -d "$PANEL" ]] || { echo "no panel in $TREE" >&2; exit 1; }

guard_start
guard_volume occupancy-panel-npm

guard_run panel \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e npm_config_cache=/tmp/.npm \
  -v "$PANEL":/w -w /w \
  -v occupancy-panel-npm:/tmp/.npm \
  node:22-alpine \
  timeout -k 30 900 \
  sh -c '[ -d node_modules ] || npm ci --no-audit --no-fund; npm run build'

# After the build, never before: vite.config.ts sets `emptyOutDir: true`. At
# dist/ root, not dist/assets, so web/__init__.py does not serve it over Ingress.
scripts/panel-source-hash.sh "$TREE" > "$TREE/panel/dist/.source-hash"

echo "built $TREE/panel/dist  (source-hash $(cut -c1-12 < "$TREE/panel/dist/.source-hash"))"
