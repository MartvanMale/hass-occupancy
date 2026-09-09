#!/usr/bin/env bash
# Build the Ingress panel into <tree>/panel/dist, never on the Home Assistant
# box; the result is committed and the Dockerfile only COPYs it. DEVELOPMENT.md
# ("The panel") says why. Finishes by stamping dist/ with a hash of its inputs so
# a stale bundle can be caught later -- scripts/panel-source-hash.sh.
#
# `--user` is not optional: without it npm writes node_modules/ and dist/ back
# root-owned and the next non-root build cannot overwrite them.
set -euo pipefail
cd "$(dirname "$0")/.."

TREE="${1:-occupancy-forecast-edge}"
PANEL="$PWD/$TREE/panel"
[[ -d "$PANEL" ]] || { echo "no panel in $TREE" >&2; exit 1; }

# Named volume for npm's cache. Created root-owned, so claim it once.
if ! docker volume inspect occupancy-panel-npm >/dev/null 2>&1; then
    docker volume create occupancy-panel-npm >/dev/null
    docker run --rm -v occupancy-panel-npm:/cache alpine \
        chown -R "$(id -u):$(id -g)" /cache
fi

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e npm_config_cache=/tmp/.npm \
  -v "$PANEL":/w -w /w \
  -v occupancy-panel-npm:/tmp/.npm \
  node:22-alpine \
  sh -c '[ -d node_modules ] || npm ci --no-audit --no-fund; npm run build'

# After the build, never before: vite.config.ts sets `emptyOutDir: true`. At
# dist/ root, not dist/assets, so web/__init__.py does not serve it over Ingress.
scripts/panel-source-hash.sh "$TREE" > "$TREE/panel/dist/.source-hash"

echo "built $TREE/panel/dist  (source-hash $(cut -c1-12 < "$TREE/panel/dist/.source-hash"))"
